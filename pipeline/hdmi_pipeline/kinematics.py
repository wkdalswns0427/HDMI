"""Forward kinematics + finite-difference velocities.

GMR emits only (root_pos, root_rot, dof_pos) per frame. HDMI's `motion.npz`
wants the full per-link world state plus velocities, so we push the GMR qpos
through the robot's MuJoCo model and read `xpos`/`xquat` back out.
"""

from __future__ import annotations

import numpy as np
import mujoco


# HDMI/Isaac and MuJoCo both use scalar-first (w, x, y, z) quaternions.
# GMR's saved pkl is the odd one out: it stores xyzw.


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[..., [1, 2, 3, 0]]


def enforce_quat_continuity(quat: np.ndarray) -> np.ndarray:
    """Flip signs so the sequence never jumps across the double cover.

    q and -q are the same rotation, but a sign flip between consecutive frames
    turns finite differencing into garbage, so fix it before differentiating.
    """
    quat = quat.copy()
    for t in range(1, quat.shape[0]):
        dot = np.sum(quat[t] * quat[t - 1], axis=-1)
        quat[t][dot < 0] *= -1.0
    return quat


class RobotFK:
    """MuJoCo-backed forward kinematics for a floating-base humanoid."""

    def __init__(self, xml_path: str, joint_names: list[str], body_names: list[str]):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.joint_names = list(joint_names)
        self.body_names = list(body_names)

        # Map requested joints onto their qpos addresses so we are never at the
        # mercy of the XML's declaration order.
        self.qpos_adr = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"joint {name!r} not in {xml_path}")
            self.qpos_adr.append(self.model.jnt_qposadr[jid])
        self.qpos_adr = np.asarray(self.qpos_adr, dtype=int)

        self.body_ids = []
        for name in self.body_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise KeyError(f"body {name!r} not in {xml_path}")
            self.body_ids.append(bid)
        self.body_ids = np.asarray(self.body_ids, dtype=int)

        # The free joint driving the floating base.
        free = [j for j in range(self.model.njnt)
                if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise ValueError(f"expected exactly one free joint, found {len(free)}")
        self.root_qpos_adr = self.model.jnt_qposadr[free[0]]

    def run(self, root_pos, root_quat_wxyz, dof_pos):
        """[T,3], [T,4], [T,J] -> body_pos [T,B,3], body_quat [T,B,4] (wxyz)."""
        T = root_pos.shape[0]
        B = len(self.body_ids)
        body_pos = np.zeros((T, B, 3), dtype=np.float64)
        body_quat = np.zeros((T, B, 4), dtype=np.float64)

        a = self.root_qpos_adr
        for t in range(T):
            self.data.qpos[:] = 0.0
            self.data.qpos[a:a + 3] = root_pos[t]
            self.data.qpos[a + 3:a + 7] = root_quat_wxyz[t]
            self.data.qpos[self.qpos_adr] = dof_pos[t]
            # Pose-only: we never need contact/inertial results here.
            mujoco.mj_kinematics(self.model, self.data)
            body_pos[t] = self.data.xpos[self.body_ids]
            body_quat[t] = self.data.xquat[self.body_ids]

        return body_pos, enforce_quat_continuity(body_quat)


def finite_diff(x: np.ndarray, fps: float) -> np.ndarray:
    """d/dt via forward differences, last frame repeated to preserve length."""
    v = (x[1:] - x[:-1]) * fps
    return np.concatenate([v, v[-1:]], axis=0)


def angular_velocity_from_quat(quat: np.ndarray, fps: float) -> np.ndarray:
    """World-frame angular velocity from a wxyz quaternion sequence.

    Mirrors `quat_to_angular_velocity` in HDMI's active_adaptation/utils/motion.py
    so the values we bake in match what the trainer would compute itself.
    """
    q1, q2 = quat[:-1], quat[1:]
    w = 2.0 * fps * np.stack([
        q1[..., 0] * q2[..., 1] - q1[..., 1] * q2[..., 0] - q1[..., 2] * q2[..., 3] + q1[..., 3] * q2[..., 2],
        q1[..., 0] * q2[..., 2] + q1[..., 1] * q2[..., 3] - q1[..., 2] * q2[..., 0] - q1[..., 3] * q2[..., 1],
        q1[..., 0] * q2[..., 3] - q1[..., 1] * q2[..., 2] + q1[..., 2] * q2[..., 1] - q1[..., 3] * q2[..., 0],
    ], axis=-1)
    return np.concatenate([w, w[-1:]], axis=0)
