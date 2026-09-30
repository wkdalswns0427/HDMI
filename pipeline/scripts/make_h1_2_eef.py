"""Author an H1-2 robot variant with wrist colliders — a dedicated end-effector.

`h1_2_handless.usd` ships with **zero colliders on its wrist links** (its wrist
geom is `contype=0 conaffinity=0`, visual only). HDMI's contact termination
needs contact *force* (`frc_thres: 1.0`), so a wrist that cannot collide can
never hold anything: every episode dies at `min_steps: 25` and the object task
learns nothing.

G1 solves this with three boxes per wrist spanning x[-0.03, 0.175]. H1-2 has no
hand at all, so the shape is a design choice rather than an inheritance.

Measured from the retargeted shovel motion, the shaft passes within 0.0011 m of
the wrist ORIGIN, but its direction in the wrist frame varies and differs per
hand (right [-0.24, 0.79, 0.52], left [-0.54, 0.39, -0.67]). The collider
therefore has to capture from any angle in the y-z plane, not point one way —
hence a forearm-aligned capsule centred on the origin rather than a flat palm.

Written as a NEW usd that references the original, so `h1_2_handless.usd` is
left untouched.

    python scripts/make_h1_2_eef.py            # default: capsule r=0.045
"""

import argparse, sys

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--name", default="h1_2_handless-eef_capsule")
_ap.add_argument("--radius", type=float, default=0.045)   # > shaft r=0.025
_ap.add_argument("--height", type=float, default=0.060)   # cylindrical section
_ap.add_argument("--offset", type=float, default=0.025)   # along +X, forearm out
ARGS, _ = _ap.parse_known_args()
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app

from pxr import Usd, UsdGeom, UsdPhysics, Gf

ASSETS = "/home/mchang344/mj_ws/simbench/HDMI/active_adaptation/assets/h1_2"
SRC = f"{ASSETS}/h1_2_handless.usd"
OUT = f"{ASSETS}/{ARGS.name}.usd"

stage = Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

root = stage.DefinePrim("/h1_2", "Xform")
root.GetReferences().AddReference("./h1_2_handless.usd")
stage.SetDefaultPrim(root)

for side in ("left", "right"):
    link = f"/h1_2/{side}_wrist_yaw_link"
    grp = UsdGeom.Xform.Define(stage, f"{link}/collisions")
    cap = UsdGeom.Capsule.Define(stage, f"{link}/collisions/palm")
    cap.CreateRadiusAttr(ARGS.radius)
    cap.CreateHeightAttr(ARGS.height)
    cap.CreateAxisAttr(UsdGeom.Tokens.x)       # along the forearm
    half = ARGS.height / 2.0 + ARGS.radius
    cap.CreateExtentAttr([Gf.Vec3f(-half, -ARGS.radius, -ARGS.radius),
                          Gf.Vec3f(half, ARGS.radius, ARGS.radius)])
    cap.CreateDisplayColorAttr([Gf.Vec3f(0.20, 0.20, 0.22)])
    UsdGeom.Xformable(cap).AddTranslateOp().Set(Gf.Vec3d(ARGS.offset, 0.0, 0.0))
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())

stage.GetRootLayer().Save()

with open("/tmp/h1_2_eef_report.txt", "w") as f:
    f.write(f"wrote {OUT}\n")
    st = Usd.Stage.Open(OUT)
    n = 0
    for prim in st.Traverse():
        p = str(prim.GetPath())
        if "wrist" in p and prim.HasAPI(UsdPhysics.CollisionAPI):
            n += 1
            f.write(f"  {p}  [{prim.GetTypeName()}]\n")
    f.write(f"  -> {n} colliders on wrist links\n")
    lo = ARGS.offset - (ARGS.height/2 + ARGS.radius)
    hi = ARGS.offset + (ARGS.height/2 + ARGS.radius)
    f.write(f"  spans x [{lo:+.3f}, {hi:+.3f}], radius {ARGS.radius} "
            f"-> encloses wrist origin: {lo < 0 < hi}\n")

app.close()
