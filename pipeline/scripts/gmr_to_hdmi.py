#!/usr/bin/env python
"""Stage 3-5: GMR retargeted qpos -> HDMI motion.npz + meta.json.

GMR gives us root pose + joint angles only. HDMI wants every link's world pose
and velocity, plus any object bodies and the contact flag. This script closes
that gap:

  GMR .pkl --FK--> link states --resample--> +object/contact --> motion.npz

Example
-------
python scripts/gmr_to_hdmi.py \
    --gmr_pkl out/move_suitcase.pkl \
    --out_dir ../HDMI/data/motion/data_for_sim/move_suitcase \
    --annotation annotations/move_suitcase.json \
    --target_fps 50
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdmi_pipeline import robots, pack
from hdmi_pipeline.kinematics import RobotFK, quat_xyzw_to_wxyz, enforce_quat_continuity
from hdmi_pipeline.objects import (
    load_annotation, resolve_object_trajectory, resolve_articulation,
)


def load_gmr_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    root_pos = np.asarray(d["root_pos"], dtype=float)
    # GMR writes root_rot as xyzw; everything downstream of here is wxyz.
    root_quat = enforce_quat_continuity(quat_xyzw_to_wxyz(np.asarray(d["root_rot"], dtype=float)))
    dof_pos = np.asarray(d["dof_pos"], dtype=float)
    fps = float(d.get("fps", 30))
    return root_pos, root_quat, dof_pos, fps


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gmr_pkl", required=True, help="output of GMR's *_to_robot.py --save_path")
    p.add_argument("--out_dir", required=True, help="motion folder to create")
    p.add_argument("--robot", default="unitree_g1", choices=sorted(robots.ROBOTS))
    p.add_argument("--annotation", default=None,
                   help="object annotation JSON; omitted -> robot-only motion")
    p.add_argument("--target_fps", type=float, default=50.0,
                   help="HDMI trains at 50 fps")
    p.add_argument("--trim", default=None, metavar="START:END",
                   help="frame range to keep, applied before resampling")
    p.add_argument("--z_offset", type=float, default=0.0,
                   help="raise/lower the whole motion, e.g. to fix foot float")
    args = p.parse_args()

    cfg = robots.get(args.robot)
    root_pos, root_quat, dof_pos, src_fps = load_gmr_pkl(args.gmr_pkl)

    if dof_pos.shape[1] != len(cfg["joint_names"]):
        raise ValueError(
            f"{args.gmr_pkl} has {dof_pos.shape[1]} dofs but {args.robot} "
            f"expects {len(cfg['joint_names'])}")

    if args.trim:
        a, b = args.trim.split(":")
        s = int(a) if a else 0
        e = int(b) if b else len(root_pos)
        root_pos, root_quat, dof_pos = root_pos[s:e], root_quat[s:e], dof_pos[s:e]
        print(f"[trim] kept frames {s}:{e} -> {len(root_pos)}")

    root_pos = root_pos.copy()
    root_pos[:, 2] += args.z_offset

    print(f"[load] {len(root_pos)} frames @ {src_fps} fps from {args.gmr_pkl}")

    # --- FK over the superset so annotations can attach to hands -------------
    fk = RobotFK(cfg["xml"], cfg["joint_names"], cfg["fk_body_names"])
    body_pos, body_quat = fk.run(root_pos, root_quat, dof_pos)
    print(f"[fk] {body_pos.shape[1]} bodies")

    body_pos, body_quat, dof_pos, fps = pack.resample(
        body_pos, body_quat, dof_pos, src_fps, args.target_fps)
    print(f"[resample] {src_fps} -> {fps} fps, {body_pos.shape[0]} frames")
    T = body_pos.shape[0]

    # --- object trajectory + contact ----------------------------------------
    obj = None
    if args.annotation:
        ann = load_annotation(args.annotation)
        # Annotation frame indices refer to the resampled timeline.
        obj_pos, obj_quat, contact = resolve_object_trajectory(
            ann, T, cfg["fk_body_names"], body_pos, body_quat)
        obj = (ann, obj_pos, obj_quat, contact)
        print(f"[object] {ann['object']['name']}: contact on "
              f"{int(contact.sum())}/{T} frames")

    # --- keep only the bodies HDMI writes ------------------------------------
    keep = [cfg["fk_body_names"].index(n) for n in cfg["out_body_names"]]
    body_pos, body_quat = body_pos[:, keep], body_quat[:, keep]

    motion = pack.build_motion(body_pos, body_quat, dof_pos, fps)
    body_names = list(cfg["out_body_names"])
    joint_names = list(cfg["joint_names"])
    contact = None

    if obj is not None:
        ann, obj_pos, obj_quat, contact = obj
        body_names = pack.append_object(
            motion, ann["object"]["name"], obj_pos, obj_quat, fps, body_names)
        for extra in ann.get("extra_objects", []):
            ep = np.tile(np.asarray(extra["pos"], dtype=float), (T, 1))
            eq = np.tile(np.asarray(extra.get("quat", [1, 0, 0, 0]), dtype=float), (T, 1))
            body_names = pack.append_object(motion, extra["name"], ep, eq, fps, body_names)
            print(f"[object] static prop {extra['name']}")
        art = resolve_articulation(ann, T)
        if art is not None:
            jname, jvals = art
            joint_names = pack.append_object_joint(motion, jname, jvals, fps, joint_names)
            print(f"[object] articulation {jname}: "
                  f"[{jvals.min():.3f}, {jvals.max():.3f}] rad")

    out = pack.write(args.out_dir, motion, body_names, joint_names, fps, contact)
    print(f"\n[done] {out}")
    print(f"  bodies : {len(body_names)}  {body_names[-3:]}")
    print(f"  joints : {len(joint_names)}")
    print(f"  frames : {T} @ {fps} fps  ({T/fps:.2f}s)")


if __name__ == "__main__":
    main()
