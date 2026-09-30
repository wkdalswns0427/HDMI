"""Write HDMI-format motion.npz + meta.json and resample to the training fps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from .kinematics import finite_diff, angular_velocity_from_quat, enforce_quat_continuity

# Keys HDMI's MotionData expects, plus the contact flag its command reads.
MOTION_KEYS = [
    "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
    "joint_pos", "joint_vel",
]


def resample(body_pos, body_quat, joint_pos, src_fps: float, tgt_fps: float):
    """Resample poses to the target fps (lerp positions, slerp rotations)."""
    if abs(src_fps - tgt_fps) < 1e-6:
        return body_pos, body_quat, joint_pos, tgt_fps

    T = body_pos.shape[0]
    dur = (T - 1) / src_fps
    ts_src = np.arange(T) / src_fps
    ts_tgt = np.arange(0.0, dur + 1e-9, 1.0 / tgt_fps)
    ts_tgt = ts_tgt[ts_tgt <= ts_src[-1]]

    def _lerp(x):
        flat = x.reshape(T, -1)
        out = np.stack([np.interp(ts_tgt, ts_src, flat[:, c])
                        for c in range(flat.shape[1])], axis=-1)
        return out.reshape(len(ts_tgt), *x.shape[1:])

    B = body_quat.shape[1]
    quat_out = np.empty((len(ts_tgt), B, 4))
    for b in range(B):
        slerp = Slerp(ts_src, R.from_quat(body_quat[:, b][:, [1, 2, 3, 0]]))
        quat_out[:, b] = slerp(ts_tgt).as_quat()[:, [3, 0, 1, 2]]

    return _lerp(body_pos), quat_out, _lerp(joint_pos), tgt_fps


def build_motion(body_pos, body_quat, joint_pos, fps: float) -> dict:
    """Attach the velocity channels HDMI needs."""
    body_quat = enforce_quat_continuity(body_quat)
    return {
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": finite_diff(body_pos, fps),
        "body_ang_vel_w": angular_velocity_from_quat(body_quat, fps),
        "joint_pos": joint_pos,
        "joint_vel": finite_diff(joint_pos, fps),
    }


def append_object(motion: dict, name: str, obj_pos, obj_quat, fps: float,
                  body_names: list[str]) -> list[str]:
    """Concatenate one object's body state onto the robot's, in place."""
    obj_quat = enforce_quat_continuity(obj_quat[:, None])[:, 0]
    motion["body_pos_w"] = np.concatenate([motion["body_pos_w"], obj_pos[:, None]], axis=1)
    motion["body_quat_w"] = np.concatenate([motion["body_quat_w"], obj_quat[:, None]], axis=1)
    motion["body_lin_vel_w"] = np.concatenate(
        [motion["body_lin_vel_w"], finite_diff(obj_pos, fps)[:, None]], axis=1)
    motion["body_ang_vel_w"] = np.concatenate(
        [motion["body_ang_vel_w"], angular_velocity_from_quat(obj_quat[:, None], fps)], axis=1)
    return body_names + [name]


def append_object_joint(motion: dict, name: str, values, fps: float,
                        joint_names: list[str]) -> list[str]:
    """Append an articulated object's joint (e.g. a door hinge) to joint state."""
    values = np.asarray(values, dtype=float)[:, None]
    motion["joint_pos"] = np.concatenate([motion["joint_pos"], values], axis=1)
    motion["joint_vel"] = np.concatenate([motion["joint_vel"], finite_diff(values, fps)], axis=1)
    return joint_names + [name]


def write(out_dir: str | Path, motion: dict, body_names, joint_names, fps: float,
          object_contact=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    T = motion["body_pos_w"].shape[0]
    for k in MOTION_KEYS:
        if motion[k].shape[0] != T:
            raise ValueError(f"{k} has {motion[k].shape[0]} frames, expected {T}")
    if motion["body_pos_w"].shape[1] != len(body_names):
        raise ValueError(
            f"body_pos_w has {motion['body_pos_w'].shape[1]} bodies but "
            f"meta lists {len(body_names)}")
    if motion["joint_pos"].shape[1] != len(joint_names):
        raise ValueError(
            f"joint_pos has {motion['joint_pos'].shape[1]} joints but "
            f"meta lists {len(joint_names)}")

    payload = {k: np.asarray(motion[k], dtype=np.float64) for k in MOTION_KEYS}
    if object_contact is None:
        object_contact = np.zeros((T, 1), dtype=bool)
    payload["object_contact"] = np.asarray(object_contact, dtype=bool).reshape(T, 1)

    np.savez(out / "motion.npz", **payload)
    with open(out / "meta.json", "w") as f:
        json.dump({"body_names": list(body_names),
                   "joint_names": list(joint_names),
                   "fps": float(fps)}, f, indent=4)
    return out
