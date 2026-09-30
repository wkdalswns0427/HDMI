#!/usr/bin/env python
"""Render an HDMI motion folder to an mp4 (headless, via MuJoCo + EGL).

Replays the reference motion so you can actually look at it before training.
Object bodies are drawn as translucent boxes and turn GREEN on frames where
`object_contact` is True, which makes a mis-annotated contact window obvious.

    python scripts/render_motion.py <motion_dir> --robot unitree_g1 -o out.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")   # must be set before mujoco import

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hdmi_pipeline import robots

CONTACT_RGBA = (0.20, 0.85, 0.35, 0.55)   # green while in contact
IDLE_RGBA = (0.85, 0.55, 0.15, 0.45)      # amber otherwise


def add_box(scene, pos, quat_wxyz, size, rgba):
    """Append one translucent box to the render scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat_wxyz, dtype=float))
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_BOX,
        np.asarray(size, dtype=float),
        np.asarray(pos, dtype=float), mat,
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("motion_dir")
    p.add_argument("-o", "--out", default=None, help="output mp4")
    p.add_argument("--robot", default="unitree_g1", choices=sorted(robots.ROBOTS))
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--distance", type=float, default=3.2)
    p.add_argument("--azimuth", type=float, default=135.0)
    p.add_argument("--elevation", type=float, default=-15.0)
    p.add_argument("--object_size", type=float, nargs=3, default=[0.15, 0.15, 0.15])
    p.add_argument("--stride", type=int, default=1, help="render every Nth frame")
    p.add_argument("--max_frames", type=int, default=None)
    args = p.parse_args()

    path = Path(args.motion_dir)
    d = np.load(path / "motion.npz")
    meta = json.load(open(path / "meta.json"))
    names, fps = meta["body_names"], float(meta["fps"])

    cfg = robots.get(args.robot)
    model = mujoco.MjModel.from_xml_path(str(cfg["xml"]))
    data = mujoco.MjData(model)

    # Reuse the FK address mapping so joints land in the right qpos slots.
    fk = robots.get(args.robot)
    joint_adr = []
    for n in cfg["joint_names"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        joint_adr.append(model.jnt_qposadr[jid])
    joint_adr = np.asarray(joint_adr, dtype=int)
    free = [j for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    root_adr = model.jnt_qposadr[free]

    root_i = names.index(cfg["root_body"])
    # Anything that is not a robot link is an appended object body.
    robot_set = set(cfg["fk_body_names"])
    obj_idx = [i for i, n in enumerate(names) if n not in robot_set]
    contact = d["object_contact"] if "object_contact" in d.files else None

    T = d["body_pos_w"].shape[0]
    frames = range(0, T, args.stride)
    if args.max_frames:
        frames = list(frames)[:args.max_frames]
    frames = list(frames)

    out = Path(args.out) if args.out else path / "preview.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_fps = max(1.0, fps / args.stride)

    print(f"rendering {len(frames)} frames -> {out} @ {out_fps:.1f} fps")
    if obj_idx:
        print(f"objects: {[names[i] for i in obj_idx]} (green = in contact)")

    ff = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{args.width}x{args.height}", "-r", f"{out_fps}",
        "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", str(out),
    ], stdin=subprocess.PIPE)

    renderer = mujoco.Renderer(model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = args.distance, args.azimuth, args.elevation

    try:
        for k, t in enumerate(frames):
            data.qpos[:] = 0.0
            data.qpos[root_adr:root_adr + 3] = d["body_pos_w"][t, root_i]
            data.qpos[root_adr + 3:root_adr + 7] = d["body_quat_w"][t, root_i]
            data.qpos[joint_adr] = d["joint_pos"][t, :len(joint_adr)]
            mujoco.mj_forward(model, data)

            cam.lookat[:] = d["body_pos_w"][t, root_i]
            renderer.update_scene(data, camera=cam)

            in_contact = bool(contact[t, 0]) if contact is not None else False
            for i in obj_idx:
                add_box(renderer.scene,
                        d["body_pos_w"][t, i], d["body_quat_w"][t, i],
                        args.object_size,
                        CONTACT_RGBA if in_contact else IDLE_RGBA)

            ff.stdin.write(renderer.render().tobytes())
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(frames)}")
    finally:
        ff.stdin.close()
        ff.wait()
        # Drop the renderer before interpreter teardown; EGL cleanup at exit is noisy.
        del renderer

    print(f"[done] {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
