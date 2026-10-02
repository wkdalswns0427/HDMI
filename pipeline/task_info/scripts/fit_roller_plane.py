#!/usr/bin/env python3

"""
Fit a vertical wall plane from the 3D roller-head trajectory.

Pipeline:
    1. Load roller-head endpoints and wrist positions.
    2. Sample points along each roller-head centerline.
    3. Fit a plane using PCA / SVD.
    4. Enforce vertical-wall constraint: normal_z = 0.
    5. Determine which side of the fitted plane the human is on.
    6. Move the center-axis plane one roller radius AWAY from the human.
    7. The resulting offset plane is treated as the wall surface.
    8. Project the roller-head centerlines onto the wall.
    9. Convert projected roller trajectories to wall-local (u, v).
   10. Save results and visualize interactively.
"""

from pathlib import Path

import numpy as np


# ============================================================
# Paths
# ============================================================

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from hdmi_pipeline import paths

import argparse as _argparse
_ap = _argparse.ArgumentParser()
_ap.add_argument("--task", default="wall_painting2",
                 help="motion under data/motion/data_for_sim/ and output dir under task_info/")
_ap.add_argument("--no_show", action="store_true",
                 help="save the outputs without opening a window")
ARGS = _ap.parse_args()

INPUT_PATH = paths.TASK_INFO / ARGS.task / "roller_head_trajectory.npz"
OUTPUT_PATH = paths.TASK_INFO / ARGS.task / "roller_fitted_plane.npz"


# ============================================================
# Settings
# ============================================================

N_SAMPLES_PER_ROLLER = 25

# Roller-head radius [m]
ROLLER_RADIUS = (
    0.31272733 - 0.22781034
) / 2.0


# ============================================================
# Load data
# ============================================================

data = np.load(
    INPUT_PATH,
    allow_pickle=True,
)

print("====================================================")
print("INPUT")
print("====================================================")

print("path:")
print(INPUT_PATH)

print()
print("keys:")
print(data.files)


A = data["endpoint_a_world"]          # [T, 3]
B = data["endpoint_b_world"]          # [T, 3]

left_wrist = data["left_wrist_world"]
right_wrist = data["right_wrist_world"]


if A.shape != B.shape:
    raise ValueError(
        f"A and B shapes differ: {A.shape} vs {B.shape}"
    )

if A.ndim != 2 or A.shape[1] != 3:
    raise ValueError(
        f"Expected [T,3], got {A.shape}"
    )


T = len(A)

print()
print("frames :", T)
print("A shape:", A.shape)
print("B shape:", B.shape)

print()
print(
    f"roller radius: "
    f"{ROLLER_RADIUS:.6f} m"
)


# ============================================================
# 1. Sample roller-head centerlines
# ============================================================

alpha = np.linspace(
    0.0,
    1.0,
    N_SAMPLES_PER_ROLLER,
)[None, :, None]


sampled_points = (
    A[:, None, :] * (1.0 - alpha)
    + B[:, None, :] * alpha
)


points = sampled_points.reshape(
    -1,
    3,
)


print()
print("====================================================")
print("SWEPT REGION")
print("====================================================")

print(
    "samples per roller:",
    N_SAMPLES_PER_ROLLER,
)

print(
    "total fitting points:",
    len(points),
)


# ============================================================
# 2. Compute centroid
# ============================================================

centroid = points.mean(
    axis=0
)

centered = (
    points
    - centroid
)


# ============================================================
# 3. Raw PCA / SVD plane
# ============================================================

_, singular_values, Vt = np.linalg.svd(
    centered,
    full_matrices=False,
)


raw_normal = Vt[2].copy()

raw_normal /= np.linalg.norm(
    raw_normal
)


# Deterministic sign
if raw_normal[1] < 0:
    raw_normal = -raw_normal


raw_d = -np.dot(
    raw_normal,
    centroid,
)


raw_signed_distance = (
    centered
    @ raw_normal
)


raw_rmse = np.sqrt(
    np.mean(
        raw_signed_distance ** 2
    )
)


print()
print("====================================================")
print("RAW PCA PLANE")
print("====================================================")

print("singular values:")
print(singular_values)

print()
print("raw normal:")
print(raw_normal)

print()
print("raw plane equation:")

print(
    f"{raw_normal[0]:+.6f} x "
    f"{raw_normal[1]:+.6f} y "
    f"{raw_normal[2]:+.6f} z "
    f"{raw_d:+.6f} = 0"
)

print()
print(
    f"raw RMSE: "
    f"{raw_rmse:.6f} m"
)


# ============================================================
# 4. Enforce vertical-wall constraint
# ============================================================

# Vertical wall:
#
#     normal_z = 0

normal = raw_normal.copy()

normal[2] = 0.0


horizontal_norm = np.linalg.norm(
    normal
)

if horizontal_norm < 1e-8:
    raise ValueError(
        "PCA normal is almost vertical. "
        "Cannot construct vertical wall."
    )


normal /= horizontal_norm


# Deterministic sign
if normal[1] < 0:
    normal = -normal


# ============================================================
# 5. Define wall coordinate system
# ============================================================

# Vertical axis of wall
axis_v = np.array([
    0.0,
    0.0,
    1.0,
])


# Horizontal axis along wall
axis_u = np.cross(
    axis_v,
    normal,
)

axis_u /= np.linalg.norm(
    axis_u
)


# ============================================================
# 6. Center-axis plane equation
# ============================================================

center_plane_d = -np.dot(
    normal,
    centroid,
)


# Fitting errors after vertical constraint

signed_distance = (
    centered
    @ normal
)

distance = np.abs(
    signed_distance
)


mean_error = np.mean(
    distance
)

rmse = np.sqrt(
    np.mean(
        signed_distance ** 2
    )
)

p95_error = np.percentile(
    distance,
    95,
)

max_error = np.max(
    distance
)


print()
print("====================================================")
print("CONSTRAINED CENTER-AXIS PLANE")
print("====================================================")

print("centroid:")
print(centroid)

print()
print("normal:")
print(normal)

print()
print("axis_u:")
print(axis_u)

print()
print("axis_v:")
print(axis_v)

print()
print("center-axis plane:")

print(
    f"{normal[0]:+.6f} x "
    f"{normal[1]:+.6f} y "
    f"{normal[2]:+.6f} z "
    f"{center_plane_d:+.6f} = 0"
)

print()
print("fit error:")

print(
    f"mean : {mean_error:.6f} m"
)

print(
    f"RMSE : {rmse:.6f} m"
)

print(
    f"p95  : {p95_error:.6f} m"
)

print(
    f"max  : {max_error:.6f} m"
)


# ============================================================
# 7. Estimate which side contains the human
# ============================================================

# Use the midpoint between the two wrists as a human-side
# reference for every frame.

human_reference_per_frame = (
    left_wrist
    + right_wrist
) / 2.0


# Average across the whole demonstration.

human_reference = (
    human_reference_per_frame.mean(
        axis=0
    )
)


# Vector:
#
# center plane -> human

center_to_human = (
    human_reference
    - centroid
)


# Only the normal component matters.

human_side_value = np.dot(
    center_to_human,
    normal,
)


# If positive:
#     +normal points toward human
#     therefore wall is -normal.
#
# If negative:
#     -normal points toward human
#     therefore wall is +normal.

if human_side_value > 0.0:

    human_direction = normal.copy()

    wall_direction = -normal.copy()

else:

    human_direction = -normal.copy()

    wall_direction = normal.copy()


print()
print("====================================================")
print("HUMAN / WALL SIDE")
print("====================================================")

print("human reference:")
print(human_reference)

print()
print(
    "human side dot product:",
    human_side_value,
)

print()
print("direction toward human:")
print(human_direction)

print()
print("direction toward wall:")
print(wall_direction)


# ============================================================
# 8. Offset center plane by roller radius
# ============================================================

# The actual wall is farther away from the human than the
# roller-head center axis.
#
# Therefore:
#
# wall_origin
#     = center plane
#       + radius * direction away from human

wall_origin = (
    centroid
    + ROLLER_RADIUS
    * wall_direction
)


# Wall plane equation:
#
# normal . x + wall_d = 0

wall_d = -np.dot(
    normal,
    wall_origin,
)


print()
print("====================================================")
print("FINAL WALL SURFACE")
print("====================================================")

print(
    f"roller radius: "
    f"{ROLLER_RADIUS:.6f} m"
)

print()
print("wall origin:")
print(wall_origin)

print()
print("wall plane:")

print(
    f"{normal[0]:+.6f} x "
    f"{normal[1]:+.6f} y "
    f"{normal[2]:+.6f} z "
    f"{wall_d:+.6f} = 0"
)


# ============================================================
# 9. Project original roller endpoints onto wall surface
# ============================================================

def project_to_plane(
    points,
    plane_origin,
    plane_normal,
):

    signed_dist = (
        points - plane_origin
    ) @ plane_normal

    projected = (
        points
        - signed_dist[:, None]
        * plane_normal[None, :]
    )

    return (
        projected,
        signed_dist,
    )


A_wall, A_wall_distance = project_to_plane(
    A,
    wall_origin,
    normal,
)

B_wall, B_wall_distance = project_to_plane(
    B,
    wall_origin,
    normal,
)


# ============================================================
# 10. Convert wall projection to wall-local UV
# ============================================================

def world_to_wall_uv(
    points,
    origin,
    axis_u,
    axis_v,
):

    relative = (
        points
        - origin
    )

    return np.column_stack([
        relative @ axis_u,
        relative @ axis_v,
    ])


A_wall_uv = world_to_wall_uv(
    A_wall,
    wall_origin,
    axis_u,
    axis_v,
)

B_wall_uv = world_to_wall_uv(
    B_wall,
    wall_origin,
    axis_u,
    axis_v,
)


# ============================================================
# 11. Wall-local trajectory range
# ============================================================

all_wall_uv = np.vstack([
    A_wall_uv,
    B_wall_uv,
])


u_min = all_wall_uv[:, 0].min()
u_max = all_wall_uv[:, 0].max()

v_min = all_wall_uv[:, 1].min()
v_max = all_wall_uv[:, 1].max()


print()
print("====================================================")
print("WALL-LOCAL ROLLER TRAJECTORY")
print("====================================================")

print(
    f"u range: "
    f"{u_min:.6f} "
    f"to "
    f"{u_max:.6f} m"
)

print(
    f"v range: "
    f"{v_min:.6f} "
    f"to "
    f"{v_max:.6f} m"
)

print()

print(
    f"horizontal span: "
    f"{u_max - u_min:.6f} m"
)

print(
    f"vertical span: "
    f"{v_max - v_min:.6f} m"
)


# ============================================================
# 12. Save
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


np.savez(
    OUTPUT_PATH,

    # Raw PCA
    raw_normal=raw_normal,
    raw_d=raw_d,
    raw_rmse=raw_rmse,
    singular_values=singular_values,

    # Center-axis fitted plane
    centroid=centroid,
    normal=normal,
    center_plane_d=center_plane_d,

    # Wall coordinate system
    axis_u=axis_u,
    axis_v=axis_v,

    # Fitting errors
    signed_distance=signed_distance,
    mean_error=mean_error,
    rmse=rmse,
    p95_error=p95_error,
    max_error=max_error,

    # Human / wall direction
    human_reference=human_reference,
    human_side_value=human_side_value,
    human_direction=human_direction,
    wall_direction=wall_direction,

    # Roller geometry
    roller_radius=ROLLER_RADIUS,

    # Final wall
    wall_origin=wall_origin,
    wall_d=wall_d,

    # Original roller endpoints
    endpoint_a_world=A,
    endpoint_b_world=B,

    # Roller projected onto wall
    endpoint_a_wall=A_wall,
    endpoint_b_wall=B_wall,

    # Wall-local roller coordinates
    endpoint_a_wall_uv=A_wall_uv,
    endpoint_b_wall_uv=B_wall_uv,

    # Metadata
    n_frames=T,
    n_samples_per_roller=N_SAMPLES_PER_ROLLER,
)


print()
print("====================================================")
print("SAVED")
print("====================================================")

print(OUTPUT_PATH)


# ============================================================
# 13. Interactive 3D visualization
# ============================================================

# Interactive 3D view only from here on; everything above is saved already.
# plotly is optional -- none of mj's envs has it -- so --no_show (or a
# missing plotly) stops here.
if ARGS.no_show:
    raise SystemExit(0)
import plotly.graph_objects as go

fig = go.Figure()


# ------------------------------------------------------------
# Original roller centerlines
# ------------------------------------------------------------

x_orig = []
y_orig = []
z_orig = []

for a, b in zip(A, B):

    x_orig += [
        a[0],
        b[0],
        None,
    ]

    y_orig += [
        a[1],
        b[1],
        None,
    ]

    z_orig += [
        a[2],
        b[2],
        None,
    ]


fig.add_trace(
    go.Scatter3d(
        x=x_orig,
        y=y_orig,
        z=z_orig,
        mode="lines",
        name="Original roller sweep",
        line=dict(
            width=2,
        ),
        opacity=0.35,
    )
)


# ------------------------------------------------------------
# Roller projected onto final wall
# ------------------------------------------------------------

x_wall = []
y_wall = []
z_wall = []

for a, b in zip(
    A_wall,
    B_wall,
):

    x_wall += [
        a[0],
        b[0],
        None,
    ]

    y_wall += [
        a[1],
        b[1],
        None,
    ]

    z_wall += [
        a[2],
        b[2],
        None,
    ]


fig.add_trace(
    go.Scatter3d(
        x=x_wall,
        y=y_wall,
        z=z_wall,
        mode="lines",
        name="Roller projected on wall",
        line=dict(
            width=3,
        ),
        opacity=0.9,
    )
)


# ============================================================
# Center plane and wall surface
# ============================================================

U_MARGIN = 0.10
V_MARGIN = 0.10


u_grid = np.linspace(
    u_min - U_MARGIN,
    u_max + U_MARGIN,
    20,
)

v_grid = np.linspace(
    v_min - V_MARGIN,
    v_max + V_MARGIN,
    20,
)


U, V = np.meshgrid(
    u_grid,
    v_grid,
)


# ------------------------------------------------------------
# Center-axis fitted plane
# ------------------------------------------------------------

center_plane_points = (
    centroid[None, None, :]
    + U[:, :, None]
    * axis_u[None, None, :]
    + V[:, :, None]
    * axis_v[None, None, :]
)


fig.add_trace(
    go.Surface(
        x=center_plane_points[:, :, 0],
        y=center_plane_points[:, :, 1],
        z=center_plane_points[:, :, 2],
        name="Roller center plane",
        opacity=0.20,
        showscale=False,
    )
)


# ------------------------------------------------------------
# Final wall surface
# ------------------------------------------------------------

wall_plane_points = (
    wall_origin[None, None, :]
    + U[:, :, None]
    * axis_u[None, None, :]
    + V[:, :, None]
    * axis_v[None, None, :]
)


fig.add_trace(
    go.Surface(
        x=wall_plane_points[:, :, 0],
        y=wall_plane_points[:, :, 1],
        z=wall_plane_points[:, :, 2],
        name="Estimated wall surface",
        opacity=0.40,
        showscale=False,
    )
)


# ------------------------------------------------------------
# Human reference point
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter3d(
        x=[human_reference[0]],
        y=[human_reference[1]],
        z=[human_reference[2]],
        mode="markers",
        name="Human reference",
        marker=dict(
            size=7,
        ),
    )
)


# ------------------------------------------------------------
# Center centroid
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter3d(
        x=[centroid[0]],
        y=[centroid[1]],
        z=[centroid[2]],
        mode="markers",
        name="Center-plane centroid",
        marker=dict(
            size=6,
        ),
    )
)


# ------------------------------------------------------------
# Wall direction arrow
# ------------------------------------------------------------

ARROW_LENGTH = 0.20

wall_direction_end = (
    centroid
    + wall_direction
    * ARROW_LENGTH
)


fig.add_trace(
    go.Scatter3d(
        x=[
            centroid[0],
            wall_direction_end[0],
        ],
        y=[
            centroid[1],
            wall_direction_end[1],
        ],
        z=[
            centroid[2],
            wall_direction_end[2],
        ],
        mode="lines+markers",
        name="Direction toward wall",
        line=dict(
            width=6,
        ),
        marker=dict(
            size=4,
        ),
    )
)


# ============================================================
# Layout
# ============================================================

fig.update_layout(

    title=(
        "Roller Sweep + Estimated Wall Surface"
    ),

    scene=dict(

        xaxis_title="World X [m]",
        yaxis_title="World Y [m]",
        zaxis_title="World Z [m]",

        aspectmode="data",
    ),

    legend=dict(
        x=0.01,
        y=0.99,
    ),

    margin=dict(
        l=0,
        r=0,
        b=0,
        t=40,
    ),
)


# ============================================================
# Show interactively
# ============================================================

if not ARGS.no_show:
    fig.show()