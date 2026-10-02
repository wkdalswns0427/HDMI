"""Author the wall the painting tasks paint on.

A wall standing on the ground, not a floating rectangle: the prim ORIGIN is at
the bottom edge, so the command grounds it by placing the origin at z = 0 and
never needs to know its height. The painted target is a region ON this wall and
moves independently of it -- a real wall does not slide up and down when the
job does.

By default a visualization prop: kinematic, collision DISABLED, so the roller
passes through it and the physics is identical to a run without it. That keeps
results comparable with every run so far.

With --collision it is a real wall: the plate gets a collider and a physics
material, so the roller (and the robot) can press on it and cannot pass
through. The contact-gated painting tasks use that variant:

    python scripts/make_canvas_usd.py --name canvas_collide --collision

Friction defaults low (0.1). The sim roller head is part of one rigid body and
cannot spin, so it slides along the wall where a real roller rolls; full
sliding friction would fight every stroke.

Local frame: the plate lies in the XY plane, so local +X is the wall's axis_u,
local +Y is axis_v, and local +Z is the wall normal. The command positions and
orients it from the target rectangle at reset.

Same nested-same-name layout as the roller and shovel, because HDMI builds prim
paths and body indices from the same name.

    python scripts/make_canvas_usd.py     (launches IsaacSim headless)
"""

import argparse, sys
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from hdmi_pipeline import paths

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--name", default="canvas")
_ap.add_argument(
    "--target",
    default=str(paths.TASK_INFO / "wall_painting2" / "paint_target_rectangle.npz"),
    help="paint_target_rectangle.npz to take width/height from",
)
_ap.add_argument("--thickness", type=float, default=0.04)
_ap.add_argument("--width", type=float, default=2.0,
                 help="wall width in metres; wider than the target so the "
                      "painted region reads as part of a wall")
_ap.add_argument("--max-z-offset", type=float, default=0.5,
                 help="largest target_region_pos_range.z the wall must still "
                      "cover; the wall is built tall enough for it")
_ap.add_argument("--headroom", type=float, default=0.2,
                 help="extra wall above the highest reachable target")
_ap.add_argument("--collision", action="store_true",
                 help="give the plate a collider and a physics material")
_ap.add_argument("--friction", type=float, default=0.1,
                 help="static and dynamic friction of the wall with --collision")
ARGS, _rest = _ap.parse_known_args()
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import os
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf

NAME = ARGS.name
OUT = str(paths.ASSETS / "objects" / NAME / f"{NAME}.usd")

data = np.load(ARGS.target, allow_pickle=True)
target_w = float(np.asarray(data["width"]).squeeze())
target_h = float(np.asarray(data["height"]).squeeze())
target_top = float(np.asarray(data["corners_world"])[:, 2].max())

# Ground to above the highest the target can be randomized to.
width = ARGS.width
height = target_top + ARGS.max_z_offset + ARGS.headroom

os.makedirs(os.path.dirname(OUT), exist_ok=True)

stage = Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

root = UsdGeom.Xform.Define(stage, f"/{NAME}")
stage.SetDefaultPrim(root.GetPrim())

body = UsdGeom.Xform.Define(stage, f"/{NAME}/{NAME}")
UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
mass.CreateMassAttr(1.0)

# The plate sits ABOVE the origin in local +Y (which maps to world z), so the
# origin is the wall's bottom edge and grounding it is just "put the origin on
# the floor". Half the thickness is pushed to -Z so the painted face lies
# exactly on the local XY plane, the plane the rasterizer works in.
plate = UsdGeom.Cube.Define(stage, f"/{NAME}/{NAME}/plate")
plate.CreateSizeAttr(1.0)
plate.AddTranslateOp().Set(Gf.Vec3d(0.0, height / 2.0, -ARGS.thickness / 2.0))
plate.AddScaleOp().Set(Gf.Vec3f(width, height, ARGS.thickness))
plate.CreateDisplayColorAttr([Gf.Vec3f(0.92, 0.90, 0.86)])

if ARGS.collision:
    UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
    material = UsdShade.Material.Define(stage, f"/{NAME}/{NAME}/wall_material")
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(ARGS.friction)
    material_api.CreateDynamicFrictionAttr(ARGS.friction)
    material_api.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(plate.GetPrim()).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )

stage.GetRootLayer().Save()

with open(f"/tmp/{NAME}_usd_report.txt", "w") as f:
    f.write(f"wrote {OUT}\n")
    f.write(f"target rectangle {target_w:.4f} x {target_h:.4f} m, top at "
            f"{target_top:.4f} m\n")
    f.write(f"wall             {width:.4f} w x {height:.4f} h m, origin at the "
            f"bottom edge (ground)\n")
    f.write(f"                 covers target z offsets up to "
            f"+-{ARGS.max_z_offset} m with {ARGS.headroom} m headroom\n")
    st = Usd.Stage.Open(OUT)
    f.write(f"defaultPrim: {st.GetDefaultPrim().GetPath()}\n")
    for p in st.Traverse():
        apis = [s for s in ("RigidBodyAPI", "CollisionAPI", "MassAPI")
                if p.HasAPI(getattr(UsdPhysics, s))]
        f.write(f"  {p.GetPath()}  [{p.GetTypeName()}]"
                + (f"  APIs={apis}" if apis else "") + "\n")

app.close()
