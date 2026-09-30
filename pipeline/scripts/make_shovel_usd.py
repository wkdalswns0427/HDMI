"""Author a minimal shovel asset that matches HDMI's object convention.

HDMI's scene setup assumes every object is a USD file and builds the contact
sensor path as `{ENV_REGEX_NS}/<asset>/<object_body_name>`, while the command
separately does `object.body_names.index(object_body_name)`. Both only line up
with the nested-same-name layout the shipped assets use:

    /shovel                        Xform                     <- defaultPrim
    /shovel/shovel                 Xform  RigidBody + Mass   <- the body
    /shovel/shovel/collisions      Capsule  Collision        <- collider

A shovel's blade geometry is irrelevant here -- HDMI tracks only the object's
root pose and the two grip points along the shaft -- so a capsule is enough.
Local +Z is the shaft, origin at the shaft midpoint.

    python scripts/make_shovel_usd.py     (launches IsaacSim headless)
"""

import argparse, sys

# AppLauncher eats unknown argv, so parse ours out first.
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--name", default="shovel")
_ap.add_argument("--radius", type=float, default=0.025)
_ap.add_argument("--height", type=float, default=1.0)   # cylindrical section
_ap.add_argument("--mass", type=float, default=2.5)
ARGS, _rest = _ap.parse_known_args()
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

NAME = ARGS.name
OUT = ("/home/mchang344/mj_ws/simbench/HDMI/active_adaptation/assets/"
       f"objects/{NAME}/{NAME}.usd")
RADIUS, HEIGHT, MASS = ARGS.radius, ARGS.height, ARGS.mass

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)

stage = Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

root = UsdGeom.Xform.Define(stage, f"/{NAME}")
stage.SetDefaultPrim(root.GetPrim())

body = UsdGeom.Xform.Define(stage, f"/{NAME}/{NAME}")
UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
mass.CreateMassAttr(MASS)

cap = UsdGeom.Capsule.Define(stage, f"/{NAME}/{NAME}/collisions")
cap.CreateRadiusAttr(RADIUS)
cap.CreateHeightAttr(HEIGHT)
cap.CreateAxisAttr(UsdGeom.Tokens.z)      # shaft along local +Z
cap.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.35, 0.15)])
# Extent keeps the bbox honest for the viewport and for scaling.
half = HEIGHT / 2.0 + RADIUS
cap.CreateExtentAttr([Gf.Vec3f(-RADIUS, -RADIUS, -half),
                      Gf.Vec3f(RADIUS, RADIUS, half)])
UsdPhysics.CollisionAPI.Apply(cap.GetPrim())

stage.GetRootLayer().Save()

with open(f"/tmp/{NAME}_usd_report.txt", "w") as f:
    f.write(f"wrote {OUT}\n")
    st = Usd.Stage.Open(OUT)
    f.write(f"defaultPrim: {st.GetDefaultPrim().GetPath()}\n")
    for p in st.Traverse():
        apis = [s for s in ("RigidBodyAPI", "CollisionAPI", "MassAPI")
                if p.HasAPI(getattr(UsdPhysics, s))]
        f.write(f"  {p.GetPath()}  [{p.GetTypeName()}]"
                + (f"  APIs={apis}" if apis else "") + "\n")

app.close()
