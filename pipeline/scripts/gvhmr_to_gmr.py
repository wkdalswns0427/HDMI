#!/usr/bin/env python
"""Stage 2: GVHMR SMPL-X prediction -> retargeted robot qpos (headless).

Same retargeting as GMR's scripts/gvhmr_to_robot.py, minus the MuJoCo viewer,
so it runs over SSH / in batch. Output is the usual GMR pkl, which
scripts/gmr_to_hdmi.py then turns into HDMI's motion.npz.

    python scripts/gvhmr_to_gmr.py \
        --gvhmr_pred ../GVHMR/outputs/demo/<clip>/hmr4d_results.pt \
        --robot unitree_g1 --save_path out/<clip>.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from hdmi_pipeline import paths

GMR_ROOT = paths.GMR_ROOT


def detect_fps(video: str):
    """Read the true average frame rate with ffprobe (nb_frames/duration)."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate", "-of",
             "default=noprint_wrappers=1:nokey=1", video],
            capture_output=True, text=True, check=True).stdout.strip()
        num, den = out.split("/")
        return float(num) / float(den) if float(den) else None
    except Exception as e:
        print(f"[fps] could not detect fps from {video}: {e}")
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gvhmr_pred", required=True, help="hmr4d_results.pt from GVHMR")
    p.add_argument("--robot", default="unitree_g1")
    p.add_argument("--save_path", required=True)
    p.add_argument("--tgt_fps", type=int, default=30,
                   help="retargeting fps; gmr_to_hdmi resamples to 50 afterwards")
    p.add_argument("--src_fps", type=float, default=None,
                   help="frame rate of the SOURCE VIDEO. GMR's loader hardcodes 30, "
                        "so a 60fps clip would otherwise retarget at half speed. "
                        "Auto-detected from --video if given.")
    p.add_argument("--video", default=None,
                   help="source video, used only to auto-detect --src_fps")
    args = p.parse_args()

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import (
        load_gvhmr_pred_file, get_gvhmr_data_offline_fast,
    )

    smplx_folder = GMR_ROOT / "assets" / "body_models"
    smplx_data, body_model, smplx_output, human_height = load_gvhmr_pred_file(
        args.gvhmr_pred, smplx_folder)
    print(f"[gvhmr] loaded {args.gvhmr_pred}; estimated human height {human_height:.2f} m")

    # GVHMR emits one pose per video frame, but load_gvhmr_pred_file hardcodes
    # mocap_frame_rate=30. If the clip is not ~30fps the retarget comes out at
    # the wrong speed (60fps -> half speed) with wrong velocities. Override it.
    src_fps = args.src_fps
    if src_fps is None and args.video:
        src_fps = detect_fps(args.video)
        if src_fps:
            print(f"[fps] detected {src_fps:.2f} fps from {args.video}")
    if src_fps:
        import torch
        if abs(src_fps - 30.0) > 1.0:
            print(f"[fps] overriding GMR's hardcoded 30 -> {src_fps:.2f} "
                  f"(otherwise the motion would play at {src_fps/30:.2f}x wrong speed)")
        smplx_data["mocap_frame_rate"] = torch.tensor(float(src_fps))

    frames, fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.tgt_fps)
    print(f"[gvhmr] {len(frames)} frames @ {fps} fps")

    retarget = GMR(actual_human_height=human_height, src_human="smplx",
                   tgt_robot=args.robot)

    qpos_list = []
    for i, frame in enumerate(frames):
        qpos_list.append(retarget.retarget(frame))
        if (i + 1) % 100 == 0:
            print(f"[retarget] {i+1}/{len(frames)}")

    qpos = np.asarray(qpos_list)
    out = Path(args.save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({
            "fps": fps,
            "root_pos": qpos[:, :3],
            # GMR's own convention: store root rotation as xyzw.
            "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],
            "dof_pos": qpos[:, 7:],
            "local_body_pos": None,
            "link_body_list": None,
        }, f)
    print(f"[done] {out}  ({qpos.shape[0]} frames, {qpos.shape[1]-7} dof)")


if __name__ == "__main__":
    main()
