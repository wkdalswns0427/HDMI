"""Resolve a hand-written object annotation into a per-frame object trajectory.

This is the stage the HDMI paper/FAQ describes as manual: you mark the frames
where the object is picked up and put down, say which link carries it while
it is held, and where it rests otherwise. Everything between keyframes is
interpolated and smoothed here.

Annotation schema (JSON) -- see annotations/ for worked examples:

{
  "object":   {"name": "suitcase", "asset": "suitcase"},
  "articulation": {"joint": "door_joint", "keyframes": [[0,0.0],[400,-1.4]]},
  "extra_objects": [{"name": "support0", "pos": [...], "quat": [...]}],
  "segments": [
    {"start":   0, "end": 119, "mode": "static", "pos": [0.6,0.1,0.12], "quat": [1,0,0,0]},
    {"start": 120, "end": 400, "mode": "attach", "parent": "left_rubber_hand"},
    {"start": 401, "end": 599, "mode": "static", "pos": [1.8,0.1,0.12], "quat": [1,0,0,0]}
  ],
  "contact": [[120, 400]],
  "smoothing": {"window": 9, "blend": 8}
}

Modes:
  static  - object holds a fixed pose.
  lerp    - object moves between this segment's pose and the next segment's
            start pose (lerp on position, slerp on rotation).
  attach  - object rides a robot link rigidly. The relative transform is taken
            from `offset_pos`/`offset_quat` if given, otherwise it is derived
            at the segment's first frame from the pose the object already had,
            which keeps the trajectory continuous at pickup.
  two_hand- object is a long tool spanning BOTH hands (shovel, broom, bar).
            Its axis is the line between the two grip links, so it follows the
            hands even as their relative pose drifts -- which a rigid `attach`
            to a single link cannot do. Fields:
              parent_a / parent_b : the two grip links; the shaft axis runs a->b
              grip_offset         : where the shaft sits in EACH grip link's
                                    own frame, in metres. Defaults to [0,0,0],
                                    which puts the shaft through the wrist joint
                                    origin -- behind the hand geometry, where no
                                    end-effector can reach it. Set this to the
                                    pocket of the actual collision shape.
              axis_offset         : slide the origin along that axis, in metres.
                                    Negative extends past parent_a (e.g. toward
                                    a shovel blade below the lower hand).
              local_axis          : which object-local axis is the shaft
                                    ("x", "y" or "z"; default "z")
"""

from __future__ import annotations

import json
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def _quat_wxyz_to_R(q: np.ndarray) -> R:
    return R.from_quat(np.asarray(q, dtype=float)[..., [1, 2, 3, 0]])


def _R_to_quat_wxyz(r: R) -> np.ndarray:
    return np.atleast_2d(r.as_quat())[..., [3, 0, 1, 2]]


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding; window is forced odd."""
    if window is None or window <= 1:
        return x
    window = int(window) | 1
    pad = window // 2
    xp = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    kern = np.ones(window) / window
    out = np.empty_like(x, dtype=float)
    flat = xp.reshape(xp.shape[0], -1)
    res = np.empty((x.shape[0], flat.shape[1]))
    for c in range(flat.shape[1]):
        res[:, c] = np.convolve(flat[:, c], kern, mode="valid")
    return res.reshape(out.shape)


def _smooth_quats(quat: np.ndarray, window: int) -> np.ndarray:
    """Smooth a rotation sequence by averaging in quaternion space.

    Fine for the small frame-to-frame deltas we get here; signs are aligned
    first so the average does not collapse toward zero.
    """
    if window is None or window <= 1:
        return quat
    q = quat.copy()
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] *= -1.0
    q = _moving_average(q, window)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def resolve_object_trajectory(
    ann: dict,
    num_frames: int,
    body_names: list[str],
    body_pos: np.ndarray,
    body_quat: np.ndarray,
):
    """Return (object_pos [T,3], object_quat [T,4] wxyz, contact [T,1] bool).

    `body_pos`/`body_quat` are the retargeted robot link states, used to make
    `attach` segments follow the carrying link.
    """
    T = num_frames
    obj_pos = np.zeros((T, 3), dtype=float)
    obj_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))

    segs = sorted(ann["segments"], key=lambda s: s["start"])
    if not segs:
        raise ValueError("annotation has no segments")

    for i, seg in enumerate(segs):
        s = int(seg["start"])
        e = min(int(seg["end"]), T - 1)
        if e < s:
            continue
        idx = np.arange(s, e + 1)
        mode = seg.get("mode", "static")

        if mode == "static":
            obj_pos[idx] = np.asarray(seg["pos"], dtype=float)
            obj_quat[idx] = np.asarray(seg.get("quat", [1, 0, 0, 0]), dtype=float)

        elif mode == "lerp":
            nxt = segs[i + 1] if i + 1 < len(segs) else None
            p0 = np.asarray(seg["pos"], dtype=float)
            q0 = np.asarray(seg.get("quat", [1, 0, 0, 0]), dtype=float)
            p1 = np.asarray(seg.get("pos_end", nxt["pos"] if nxt else seg["pos"]), dtype=float)
            q1 = np.asarray(seg.get("quat_end", (nxt or seg).get("quat", [1, 0, 0, 0])), dtype=float)
            a = np.linspace(0.0, 1.0, len(idx))[:, None]
            obj_pos[idx] = (1 - a) * p0 + a * p1
            slerp = Slerp([0.0, 1.0], _quat_wxyz_to_R(np.stack([q0, q1])))
            obj_quat[idx] = _R_to_quat_wxyz(slerp(a.ravel()))

        elif mode == "attach":
            parent = seg["parent"]
            if parent not in body_names:
                raise KeyError(f"attach parent {parent!r} not among retargeted bodies")
            b = body_names.index(parent)
            R_par = _quat_wxyz_to_R(body_quat[idx, b])
            p_par = body_pos[idx, b]

            if "offset_pos" in seg:
                off_p = np.asarray(seg["offset_pos"], dtype=float)
                off_q = _quat_wxyz_to_R(np.asarray(seg.get("offset_quat", [1, 0, 0, 0]), dtype=float))
            else:
                # Derive the grasp transform so the object continues from the
                # pose it already held. The previous segment ends at s-1, so
                # that -- not s, which is still unwritten -- is the pose to
                # match. Pinning it against the parent at frame s makes
                # obj(s) == obj(s-1) exactly, then it rides the link.
                ref = s - 1 if s > 0 else s
                R_par0 = _quat_wxyz_to_R(body_quat[s, b])
                off_p = R_par0.inv().apply(obj_pos[ref] - body_pos[s, b])
                off_q = R_par0.inv() * _quat_wxyz_to_R(obj_quat[ref])

                grasp_dist = float(np.linalg.norm(obj_pos[ref] - body_pos[s, b]))
                if grasp_dist > 0.6:
                    print(f"  [warn] frame {s}: object is {grasp_dist:.2f} m from "
                          f"{parent!r} at pickup -- check the annotation, the hand "
                          f"should be at the object when it is grasped")

            obj_pos[idx] = p_par + R_par.apply(off_p)
            obj_quat[idx] = _R_to_quat_wxyz(R_par * off_q)

        elif mode == "two_hand":
            a_name, b_name = seg["parent_a"], seg["parent_b"]
            for nm in (a_name, b_name):
                if nm not in body_names:
                    raise KeyError(f"two_hand parent {nm!r} not among retargeted bodies")
            ia, ib = body_names.index(a_name), body_names.index(b_name)
            pa, pb = body_pos[idx, ia], body_pos[idx, ib]

            # Move the grip points into each hand's own frame. Without this the
            # shaft runs through the wrist joint origins, which sit behind the
            # collision geometry -- a pose no end-effector can actually touch.
            grip = np.asarray(seg.get("grip_offset", [0.0, 0.0, 0.0]), dtype=float)
            if np.any(grip):
                pa = pa + _quat_wxyz_to_R(body_quat[idx, ia]).apply(grip)
                pb = pb + _quat_wxyz_to_R(body_quat[idx, ib]).apply(grip)

            axis = pb - pa
            span = np.linalg.norm(axis, axis=-1, keepdims=True)
            if (span < 1e-6).any():
                raise ValueError("two_hand: grip links coincide on some frame")
            u = axis / span                                   # shaft direction

            off = float(seg.get("axis_offset", 0.0))
            obj_pos[idx] = pa + u * off

            # Complete the frame: shaft along `local_axis`, remaining roll
            # resolved against world up (falling back when near-parallel).
            up = np.tile(np.array([0.0, 0.0, 1.0]), (len(idx), 1))
            degenerate = np.abs(np.sum(u * up, axis=-1)) > 0.99
            up[degenerate] = np.array([1.0, 0.0, 0.0])
            x = np.cross(up, u)
            x /= np.linalg.norm(x, axis=-1, keepdims=True)
            y = np.cross(u, x)

            local_axis = seg.get("local_axis", "z")
            if local_axis == "z":
                cols = (x, y, u)
            elif local_axis == "y":
                cols = (x, u, y)
            elif local_axis == "x":
                cols = (u, x, y)
            else:
                raise ValueError(f"local_axis must be x/y/z, got {local_axis!r}")
            mats = np.stack(cols, axis=-1)                     # [n,3,3]
            obj_quat[idx] = _R_to_quat_wxyz(R.from_matrix(mats))

        else:
            raise ValueError(f"unknown segment mode {mode!r}")

    # Any frames past the last annotated segment hold the final pose.
    last_end = min(int(segs[-1]["end"]), T - 1)
    if last_end < T - 1:
        obj_pos[last_end + 1:] = obj_pos[last_end]
        obj_quat[last_end + 1:] = obj_quat[last_end]

    sm = ann.get("smoothing", {}) or {}
    obj_pos = _moving_average(obj_pos, sm.get("window", 0))
    obj_quat = _smooth_quats(obj_quat, sm.get("window", 0))
    obj_quat /= np.linalg.norm(obj_quat, axis=-1, keepdims=True)

    contact = np.zeros((T, 1), dtype=bool)
    for a, b in ann.get("contact", []):
        contact[int(a):min(int(b), T - 1) + 1] = True

    return obj_pos, obj_quat, contact


def resolve_articulation(ann: dict, num_frames: int):
    """Return (joint_name, joint_pos [T]) for an articulated object, or None."""
    art = ann.get("articulation")
    if not art:
        return None
    kf = sorted(art["keyframes"], key=lambda k: k[0])
    frames = np.array([k[0] for k in kf], dtype=float)
    values = np.array([k[1] for k in kf], dtype=float)
    t = np.arange(num_frames, dtype=float)
    q = np.interp(t, frames, values)
    q = _moving_average(q[:, None], (ann.get("smoothing", {}) or {}).get("window", 0))[:, 0]
    return art["joint"], q


def load_annotation(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
