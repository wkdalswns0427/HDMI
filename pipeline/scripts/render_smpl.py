#!/usr/bin/env python
"""Render the SMPL-X human mesh from a GVHMR result to an mp4.

GVHMR's own `render_global` needs the licensed **SMPL** model -- not for its
weights, but only for its face topology, because it converts SMPL-X vertices
down to SMPL via `smplx2smpl_sparse.pt`. We have SMPL-X but not SMPL, which is
why `demo.py` dies at `make_smplx("smpl").faces`.

This skips the conversion and renders the SMPL-X mesh with SMPL-X's own faces,
so no extra download is needed. Output is the world-grounded ("global") view,
which is the same motion that gets retargeted to the robot -- so you can put it
side by side with pipeline/<task>.mp4.

    python scripts/render_smpl.py \
        ../GVHMR/outputs/demo/shovel_dirt/hmr4d_results.pt -o human.mp4

Run from the GVHMR directory, or pass --gvhmr_root: several GVHMR modules
resolve resources relative to cwd.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm


import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from hdmi_pipeline import paths


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pred", help="hmr4d_results.pt from GVHMR")
    p.add_argument("-o", "--out", default=None, help="output mp4")
    p.add_argument("--gvhmr_root", default=str(paths.GVHMR_ROOT))
    p.add_argument("--fps", type=int, default=30, help="GVHMR predicts at video fps")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--bin_size", type=int, default=0,
                   help="pytorch3d rasterizer bin size. 0 = naive rasterization "
                        "(slower but correct). The default binned path overflows "
                        "on the 20908-face SMPL-X mesh and drops geometry.")
    p.add_argument("--incam", action="store_true",
                   help="render in camera frame instead of world-grounded")
    args = p.parse_args()

    root = Path(args.gvhmr_root).resolve()
    sys.path.insert(0, str(root))
    import os
    os.chdir(root)   # GVHMR resolves several resources relative to cwd

    from hmr4d.utils.smplx_utils import make_smplx
    from hmr4d.utils.net_utils import to_cuda
    from hmr4d.utils.vis.renderer import (
        Renderer, get_global_cameras_static, get_ground_params_from_points,
    )
    from hmr4d.utils.geo.hmr_cam import create_camera_sensor
    from hmr4d.utils.video_io_utils import get_writer

    pred_path = Path(args.pred).resolve()
    pred = torch.load(pred_path)
    key = "smpl_params_incam" if args.incam else "smpl_params_global"
    print(f"[load] {pred_path}  ({key})")

    smplx = make_smplx("supermotion").cuda()
    faces = smplx.faces          # SMPL-X topology -- no SMPL download required
    out = smplx(**to_cuda(pred[key]))
    verts = out.vertices         # (L, 10475, 3)
    L = verts.shape[0]
    print(f"[smplx] {L} frames, {verts.shape[1]} verts, {len(faces)} faces")

    # Put the mesh on the ground with its start position at the origin. The
    # J_regressor GVHMR uses here is SMPL-topology, so derive the root from the
    # vertex centroid instead.
    verts = verts.clone()
    offset = verts[0].mean(0)
    offset[1] = verts[:, :, 1].min()
    verts = verts - offset
    root_points = verts.mean(1)

    global_R, global_T, global_lights = get_global_cameras_static(
        verts.cpu(), beta=2.0, cam_height_degree=20, target_center_height=1.0)

    _, _, K = create_camera_sensor(args.width, args.height, 24)  # 24mm lens
    renderer = Renderer(args.width, args.height, device="cuda", faces=faces, K=K,
                        bin_size=args.bin_size)

    scale, cx, cz = get_ground_params_from_points(root_points, verts)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    out_path = Path(args.out).resolve() if args.out else pred_path.parent / "human_global.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = get_writer(str(out_path), fps=args.fps, crf=23)
    for i in tqdm(range(L), desc="rendering"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(verts[[i]], color[None], cameras, global_lights)
        writer.write_frame(img)
    writer.close()

    print(f"[done] {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
