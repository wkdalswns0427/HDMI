#!/usr/bin/env python3
"""
Find the smallest wall-axis-aligned rectangle that contains
the entire area swept by the roller head on the fitted wall.

Assumption:
    The roller remains in contact with the wall throughout the clip.

Outputs:
    paint_target_rectangle.npz:
        center_world, corners_world, width, height,
        center_uv, corners_uv, wall axes, etc.

The swept region is rasterized only for visualization.
The rectangle bounds are calculated directly from the
roller endpoint trajectories.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


# ============================================================
# Paths
# ============================================================

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from hdmi_pipeline import paths

INPUT_PATH = paths.TASK_INFO / "wall_painting2" / "roller_fitted_plane.npz"
OUTPUT_PATH = paths.TASK_INFO / "wall_painting2" / "paint_target_rectangle.npz"


# ============================================================
# Settings
# ============================================================

# Used only to draw the swept area in the visualization.
# It does not affect the calculated rectangle.
VIS_RESOLUTION = 0.005  # [m/pixel]

# Extra distance around the swept area, if desired.
# Keep at 0.0 for the smallest enclosing rectangle.
RECTANGLE_MARGIN = 0.0  # [m]


# ============================================================
# 1. Load fitted wall and roller trajectory
# ============================================================

data = np.load(INPUT_PATH)

a_uv = np.asarray(
    data["endpoint_a_wall_uv"],
    dtype=float,
)

b_uv = np.asarray(
    data["endpoint_b_wall_uv"],
    dtype=float,
)

wall_origin = np.asarray(
    data["wall_origin"],
    dtype=float,
)

axis_u = np.asarray(
    data["axis_u"],
    dtype=float,
)

axis_v = np.asarray(
    data["axis_v"],
    dtype=float,
)


if a_uv.shape != b_uv.shape:
    raise ValueError(
        f"Endpoint shapes differ: {a_uv.shape} vs {b_uv.shape}"
    )

if a_uv.ndim != 2 or a_uv.shape[1] != 2:
    raise ValueError(
        f"Expected endpoint arrays with shape [T, 2], got {a_uv.shape}"
    )

if len(a_uv) < 2:
    raise ValueError(
        "At least two frames are required."
    )

if not np.isfinite(a_uv).all() or not np.isfinite(b_uv).all():
    raise ValueError(
        "Roller endpoints contain NaN or infinity."
    )


# ============================================================
# 2. Find the rectangle enclosing the entire swept area
# ============================================================
#
# At each instant, the roller head is the segment A--B.
#
# Between two frames, A and B are linearly interpolated.
# An interpolated point on the roller segment is a weighted
# combination of the four endpoint positions.
#
# Therefore, its u and v coordinates cannot exceed the
# minimum or maximum coordinates of all recorded endpoints.
# Taking those extrema encloses the entire interpolated sweep.
# ============================================================

all_uv = np.concatenate(
    [a_uv, b_uv],
    axis=0,
)

sweep_min_uv = all_uv.min(axis=0)
sweep_max_uv = all_uv.max(axis=0)

u_left = sweep_min_uv[0] - RECTANGLE_MARGIN
u_right = sweep_max_uv[0] + RECTANGLE_MARGIN

v_bottom = sweep_min_uv[1] - RECTANGLE_MARGIN
v_top = sweep_max_uv[1] + RECTANGLE_MARGIN

width = u_right - u_left
height = v_top - v_bottom

center_uv = np.array([
    (u_left + u_right) / 2.0,
    (v_bottom + v_top) / 2.0,
])

# Order: bottom-left, bottom-right, top-right, top-left
corners_uv = np.array([
    [u_left, v_bottom],
    [u_right, v_bottom],
    [u_right, v_top],
    [u_left, v_top],
])


# ============================================================
# 3. Convert rectangle to 3D world coordinates
# ============================================================

def wall_uv_to_world(points_uv):
    """
    Convert wall-local UV points to 3D world coordinates.

    points_uv: [..., 2]
    returns:   [..., 3]
    """
    points_uv = np.asarray(
        points_uv,
        dtype=float,
    )

    return (
        wall_origin
        + points_uv[..., 0, None] * axis_u
        + points_uv[..., 1, None] * axis_v
    )


center_world = wall_uv_to_world(center_uv)
corners_world = wall_uv_to_world(corners_uv)


# ============================================================
# 4. Save
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

np.savez(
    OUTPUT_PATH,

    # Main result in 3D world coordinates
    center_world=center_world,
    corners_world=corners_world,
    width=width,
    height=height,

    # Same rectangle in wall-local coordinates
    center_uv=center_uv,
    corners_uv=corners_uv,

    # Bounds of the projected roller sweep
    sweep_min_uv=sweep_min_uv,
    sweep_max_uv=sweep_max_uv,

    # Wall coordinate system
    wall_origin=wall_origin,
    axis_u=axis_u,
    axis_v=axis_v,

    # Settings
    rectangle_margin=RECTANGLE_MARGIN,
    vis_resolution=VIS_RESOLUTION,
)

print("======================================")
print("FULL-SWEEP TARGET RECTANGLE")
print("======================================")

print("input:", INPUT_PATH)
print("frames:", len(a_uv))

print()
print("sweep minimum [u, v] [m]:", sweep_min_uv)
print("sweep maximum [u, v] [m]:", sweep_max_uv)

print()
print("center_uv [m]:", center_uv)
print("center_world [m]:", center_world)

print()
print(f"width  [m]: {width:.6f}")
print(f"height [m]: {height:.6f}")
print(f"area  [m²]: {width * height:.6f}")

print()
print("corners_world [m] (BL, BR, TR, TL):")
print(corners_world)

print()
print("saved:", OUTPUT_PATH)


# ============================================================
# 5. Rasterize the swept area for visualization only
# ============================================================

plot_padding = 3 * VIS_RESOLUTION

grid_min_uv = (
    sweep_min_uv
    - RECTANGLE_MARGIN
    - plot_padding
)

grid_max_uv = (
    sweep_max_uv
    + RECTANGLE_MARGIN
    + plot_padding
)

grid_width = (
    int(np.ceil(
        (grid_max_uv[0] - grid_min_uv[0])
        / VIS_RESOLUTION
    ))
    + 1
)

grid_height = (
    int(np.ceil(
        (grid_max_uv[1] - grid_min_uv[1])
        / VIS_RESOLUTION
    ))
    + 1
)

mask = np.zeros(
    (grid_height, grid_width),
    dtype=np.uint8,
)


def uv_to_pixel(point_uv):
    """
    Convert one wall-local [u, v] point to
    integer [column, row] pixel coordinates.
    """
    return np.rint(
        (point_uv - grid_min_uv)
        / VIS_RESOLUTION
    ).astype(np.int32)


for t in range(len(a_uv) - 1):

    displacement = max(
        np.linalg.norm(
            a_uv[t + 1] - a_uv[t]
        ),
        np.linalg.norm(
            b_uv[t + 1] - b_uv[t]
        ),
    )

    steps = max(
        1,
        int(np.ceil(
            displacement
            / (VIS_RESOLUTION / 2.0)
        )),
    )

    for k in range(steps + 1):

        alpha = k / steps

        a = (
            (1.0 - alpha) * a_uv[t]
            + alpha * a_uv[t + 1]
        )

        b = (
            (1.0 - alpha) * b_uv[t]
            + alpha * b_uv[t + 1]
        )

        pa = uv_to_pixel(a)
        pb = uv_to_pixel(b)

        cv2.line(
            mask,
            tuple(pa),
            tuple(pb),
            color=1,
            thickness=1,
            lineType=cv2.LINE_8,
        )


# ============================================================
# 6. Interactive visualization
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 9),
)

ax.imshow(
    np.ma.masked_where(mask == 0, mask),
    origin="lower",
    interpolation="nearest",
    extent=(
        grid_min_uv[0] - VIS_RESOLUTION / 2.0,
        grid_min_uv[0]
        + (grid_width - 0.5) * VIS_RESOLUTION,

        grid_min_uv[1] - VIS_RESOLUTION / 2.0,
        grid_min_uv[1]
        + (grid_height - 0.5) * VIS_RESOLUTION,
    ),
    cmap="Blues",
    vmin=0,
    vmax=1,
    alpha=0.85,
)

ax.add_patch(
    Rectangle(
        (u_left, v_bottom),
        width,
        height,
        fill=False,
        edgecolor="red",
        linewidth=2.5,
        label="Rectangle enclosing entire sweep",
    )
)

ax.scatter(
    center_uv[0],
    center_uv[1],
    color="red",
    marker="x",
    s=80,
    label="Rectangle center",
)

ax.set_xlabel("Wall u [m]")
ax.set_ylabel("Wall v [m]")

ax.set_title(
    "Roller swept area and enclosing target rectangle"
)

ax.set_aspect("equal")
ax.grid(alpha=0.2)
ax.legend()

plt.tight_layout()
plt.show()