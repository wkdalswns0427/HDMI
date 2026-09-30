"""Author a roller asset that matches HDMI's object convention.

HDMI's scene setup assumes every object is a USD file and builds the contact
sensor path as `{ENV_REGEX_NS}/<asset>/<object_body_name>`, while the command
separately does `object.body_names.index(object_body_name)`. Both only line up
with the nested-same-name layout the shipped assets use:

    /roller                        Xform                     <- defaultPrim
    /roller/roller                 Xform  RigidBody + Mass   <- the body
    /roller/roller/pole            Capsule  Collision
    /roller/roller/connector_1     Cylinder Collision
    /roller/roller/connector_2     Cylinder Collision
    /roller/roller/connector_3     Cylinder Collision
    /roller/roller/connector_4     Cylinder Collision
    /roller/roller/joint_1         Sphere   Collision
    /roller/roller/joint_2         Sphere   Collision
    /roller/roller/joint_3         Sphere   Collision
    /roller/roller/head            Cylinder Collision

    python scripts/make_roller_usd.py     (launches IsaacSim headless)
"""

import argparse, sys

# AppLauncher eats unknown argv, so parse ours out first.
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--name", default="roller")
_ap.add_argument("--mass", type=float, default=2.5)
ARGS, _rest = _ap.parse_known_args()
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Usd, UsdGeom, UsdPhysics, Gf

NAME = ARGS.name
OUT = ("/home/mchang344/mj_ws/simbench/HDMI/active_adaptation/assets/"
       f"objects/{NAME}/{NAME}.usd")
MASS = ARGS.mass

import os
import math
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
mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.05))

pole_radius = 0.020865
pole_total_length = 1.0
pole_height = pole_total_length - 2.0 * pole_radius
pole_center_z = (-0.82553955 + 0.17446045) / 2.0

pole = UsdGeom.Capsule.Define(stage, f"/{NAME}/{NAME}/pole")
pole.CreateRadiusAttr(pole_radius)
pole.CreateHeightAttr(pole_height)
pole.CreateAxisAttr(UsdGeom.Tokens.z)
pole.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, pole_center_z))
pole.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.55)])
UsdPhysics.CollisionAPI.Apply(pole.GetPrim())

connector_radius = 0.00615

connector_1 = UsdGeom.Cylinder.Define(stage, f"/{NAME}/{NAME}/connector_1")
connector_1.CreateRadiusAttr(connector_radius)
connector_1.CreateHeightAttr(0.040)
connector_1.CreateAxisAttr(UsdGeom.Tokens.z)
connector_1.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.191))
connector_1.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.55)])
UsdPhysics.CollisionAPI.Apply(connector_1.GetPrim())

connector_2 = UsdGeom.Cylinder.Define(stage, f"/{NAME}/{NAME}/connector_2")
connector_2.CreateRadiusAttr(connector_radius)
connector_2.CreateHeightAttr(0.120)
connector_2.CreateAxisAttr(UsdGeom.Tokens.x)
connector_2.AddTranslateOp().Set(Gf.Vec3d(0.060, 0.0, 0.211))
connector_2.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.55)])
UsdPhysics.CollisionAPI.Apply(connector_2.GetPrim())

connector_3 = UsdGeom.Cylinder.Define(stage, f"/{NAME}/{NAME}/connector_3")
connector_3.CreateRadiusAttr(connector_radius)
connector_3.CreateHeightAttr(0.060)
connector_3.CreateAxisAttr(UsdGeom.Tokens.z)
connector_3.AddTranslateOp().Set(Gf.Vec3d(0.120, 0.0, 0.241))
connector_3.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.55)])
UsdPhysics.CollisionAPI.Apply(connector_3.GetPrim())

connector_4 = UsdGeom.Cylinder.Define(stage, f"/{NAME}/{NAME}/connector_4")
connector_4.CreateRadiusAttr(connector_radius)
connector_4.CreateHeightAttr(0.120)
connector_4.CreateAxisAttr(UsdGeom.Tokens.x)
connector_4.AddTranslateOp().Set(Gf.Vec3d(0.060, 0.0, 0.270))
connector_4.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.55)])
UsdPhysics.CollisionAPI.Apply(connector_4.GetPrim())

joint_1 = UsdGeom.Sphere.Define(stage, f"/{NAME}/{NAME}/joint_1")
joint_1.CreateRadiusAttr(connector_radius)
joint_1.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.211))
joint_1.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.55)])
UsdPhysics.CollisionAPI.Apply(joint_1.GetPrim())

joint_2 = UsdGeom.Sphere.Define(stage, f"/{NAME}/{NAME}/joint_2")
joint_2.CreateRadiusAttr(connector_radius)
joint_2.AddTranslateOp().Set(Gf.Vec3d(0.120, 0.0, 0.211))
joint_2.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.55)])
UsdPhysics.CollisionAPI.Apply(joint_2.GetPrim())

joint_3 = UsdGeom.Sphere.Define(stage, f"/{NAME}/{NAME}/joint_3")
joint_3.CreateRadiusAttr(connector_radius)
joint_3.AddTranslateOp().Set(Gf.Vec3d(0.120, 0.0, 0.270))
joint_3.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.55)])
UsdPhysics.CollisionAPI.Apply(joint_3.GetPrim())

head_radius = (0.31272733 - 0.22781034) / 2.0
head_length = 0.11411140 - (-0.11502573)
head_center_x = (-0.11502573 + 0.11411140) / 2.0
head_center_y = -(-0.04266473 + 0.04225311) / 2.0
head_center_z = (0.22781034 + 0.31272733) / 2.0

head = UsdGeom.Cylinder.Define(stage, f"/{NAME}/{NAME}/head")
head.CreateRadiusAttr(head_radius)
head.CreateHeightAttr(head_length)
head.CreateAxisAttr(UsdGeom.Tokens.x)
head.AddTranslateOp().Set(
    Gf.Vec3d(head_center_x, head_center_y, head_center_z)
)
head.CreateDisplayColorAttr([Gf.Vec3f(0.75, 0.75, 0.75)])
UsdPhysics.CollisionAPI.Apply(head.GetPrim())

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