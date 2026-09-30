#!/usr/bin/env python3
"""
Visualize the 3D surface swept by the CENTERLINE of the roller head.

- Loads HDMI FK results from motion.npz
- Reproduces the `two_hand` calculation from hdmi_pipeline/objects.py
- Computes the roller-head centerline endpoints
- Visualizes the 3D swept surface

NOTE:
- Roller radius is NOT considered here.
- No wall fitting is performed here.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from hdmi_pipeline import paths

MOTION_PATH = paths.MOTION_DATA / "wall_painting2" / "motion.npz"
OUTPUT_PATH = paths.TASK_INFO / "wall_painting2" / "roller_head_swept_3d.png"
TRAJECTORY_PATH = paths.TASK_INFO / "wall_painting2" / "roller_head_trajectory.npz"

UNITREE_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]


AXIS_OFFSET = 0.7

HEAD_X_MIN = -0.11502573
HEAD_X_MAX = 0.11411140

HEAD_Y_MIN = -0.04266473
HEAD_Y_MAX = 0.04225311

HEAD_Z_MIN = 0.22781034
HEAD_Z_MAX = 0.31272733

HEAD_LENGTH = HEAD_X_MAX - HEAD_X_MIN

HEAD_CENTER_LOCAL = np.array([
    (HEAD_X_MIN + HEAD_X_MAX) / 2.0,
    -((HEAD_Y_MIN + HEAD_Y_MAX) / 2.0),
    (HEAD_Z_MIN + HEAD_Z_MAX) / 2.0,
])

# Roller cylinder axis = local X
HEAD_ENDPOINT_A_LOCAL = (
    HEAD_CENTER_LOCAL
    + np.array([-HEAD_LENGTH / 2.0, 0.0, 0.0])
)

HEAD_ENDPOINT_B_LOCAL = (
    HEAD_CENTER_LOCAL
    + np.array([+HEAD_LENGTH / 2.0, 0.0, 0.0])
)

def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)

    if np.any(norm < 1e-12):
        raise ValueError("Near-zero vector encountered.")

    return v / norm


def compute_two_hand_pose(right_wrist, left_wrist):
    """
    Reproduce the exact two_hand construction used in
    pipeline/hdmi_pipeline/objects.py.

    parent_a = right_wrist_yaw_link
    parent_b = left_wrist_yaw_link
    axis_offset = 0.7
    local_axis = z
    """

    axis = left_wrist - right_wrist

    span = np.linalg.norm(
        axis,
        axis=-1,
        keepdims=True,
    )

    if np.any(span < 1e-6):
        raise ValueError(
            "Left/right wrist positions coincide."
        )

    u = axis / span

    roller_pos = (
        right_wrist
        + u * AXIS_OFFSET
    )

    up = np.tile(
        np.array([0.0, 0.0, 1.0]),
        (len(u), 1),
    )

    degenerate = (
        np.abs(np.sum(u * up, axis=-1))
        > 0.99
    )

    up[degenerate] = np.array(
        [1.0, 0.0, 0.0]
    )

    x = np.cross(up, u)
    x = normalize(x)

    y = np.cross(u, x)
    y = normalize(y)

    # local_axis == "z"
    #
    # local X -> x
    # local Y -> y
    # local Z -> u

    roller_rot = np.stack(
        (x, y, u),
        axis=-1,
    )

    return roller_pos, roller_rot


def transform_point(
    roller_pos,
    roller_rot,
    point_local,
):
    """
    Transform one local point into world coordinates
    for every frame.
    """

    rotated = np.einsum(
        "tij,j->ti",
        roller_rot,
        point_local,
    )

    return roller_pos + rotated


def set_axes_equal(ax, points):
    """
    Make X/Y/Z use the same physical scale.
    """

    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    center = (mins + maxs) / 2.0

    radius = np.max(
        maxs - mins
    ) / 2.0

    if radius < 1e-9:
        radius = 0.5

    ax.set_xlim(
        center[0] - radius,
        center[0] + radius,
    )

    ax.set_ylim(
        center[1] - radius,
        center[1] + radius,
    )

    ax.set_zlim(
        center[2] - radius,
        center[2] + radius,
    )

    ax.set_box_aspect((1, 1, 1))

motion = np.load(
    MOTION_PATH,
    allow_pickle=True,
)

body_pos = motion["body_pos_w"]

print("======================================")
print("Motion")
print("======================================")

print("path :", MOTION_PATH)
print("body_pos_w shape :", body_pos.shape)

T = body_pos.shape[0]

print("frames :", T)

LEFT_WRIST_INDEX = UNITREE_BODY_NAMES.index(
    "left_wrist_yaw_link"
)

RIGHT_WRIST_INDEX = UNITREE_BODY_NAMES.index(
    "right_wrist_yaw_link"
)

print()
print("left wrist index  :", LEFT_WRIST_INDEX)
print("right wrist index :", RIGHT_WRIST_INDEX)

left_wrist = body_pos[
    :,
    LEFT_WRIST_INDEX,
    :
]

right_wrist = body_pos[
    :,
    RIGHT_WRIST_INDEX,
    :
]

roller_pos, roller_rot = compute_two_hand_pose(
    right_wrist,
    left_wrist,
)

endpoint_a = transform_point(
    roller_pos,
    roller_rot,
    HEAD_ENDPOINT_A_LOCAL,
)

endpoint_b = transform_point(
    roller_pos,
    roller_rot,
    HEAD_ENDPOINT_B_LOCAL,
)

head_center = transform_point(
    roller_pos,
    roller_rot,
    HEAD_CENTER_LOCAL,
)

head_lengths = np.linalg.norm(
    endpoint_b - endpoint_a,
    axis=1,
)

wrist_spans = np.linalg.norm(
    left_wrist - right_wrist,
    axis=1,
)

print()
print("======================================")
print("Roller geometry")
print("======================================")

print(
    f"head length        : "
    f"{HEAD_LENGTH:.6f} m"
)

print(
    "head center local  :",
    HEAD_CENTER_LOCAL,
)

print(
    f"measured head length: "
    f"{head_lengths.mean():.6f} m"
)

print()
print("======================================")
print("Wrist span")
print("======================================")

print(
    f"mean : {wrist_spans.mean():.6f} m"
)

print(
    f"min  : {wrist_spans.min():.6f} m"
)

print(
    f"max  : {wrist_spans.max():.6f} m"
)

np.savez(
    TRAJECTORY_PATH,

    endpoint_a_world=endpoint_a,
    endpoint_b_world=endpoint_b,

    head_center_world=head_center,

    roller_pos_world=roller_pos,
    roller_rot_world=roller_rot,

    left_wrist_world=left_wrist,
    right_wrist_world=right_wrist,

    head_length=HEAD_LENGTH,
)

print()
print(
    "trajectory saved :",
    TRAJECTORY_PATH,
)

quads = []

for t in range(T - 1):

    quad = [
        endpoint_a[t],
        endpoint_b[t],
        endpoint_b[t + 1],
        endpoint_a[t + 1],
    ]

    quads.append(quad)

fig = plt.figure(
    figsize=(11, 9)
)

ax = fig.add_subplot(
    111,
    projection="3d",
)

surface = Poly3DCollection(
    quads,
    alpha=0.25,
    linewidths=0.15,
)

ax.add_collection3d(surface)

ax.plot(
    head_center[:, 0],
    head_center[:, 1],
    head_center[:, 2],
    linewidth=2.0,
    label="Roller head center",
)

LINE_STEP = 10

for t in range(
    0,
    T,
    LINE_STEP,
):

    ax.plot(
        [
            endpoint_a[t, 0],
            endpoint_b[t, 0],
        ],
        [
            endpoint_a[t, 1],
            endpoint_b[t, 1],
        ],
        [
            endpoint_a[t, 2],
            endpoint_b[t, 2],
        ],
        linewidth=1.0,
    )

ax.plot(
    [
        endpoint_a[0, 0],
        endpoint_b[0, 0],
    ],
    [
        endpoint_a[0, 1],
        endpoint_b[0, 1],
    ],
    [
        endpoint_a[0, 2],
        endpoint_b[0, 2],
    ],
    linewidth=3.0,
    label="Start",
)

ax.plot(
    [
        endpoint_a[-1, 0],
        endpoint_b[-1, 0],
    ],
    [
        endpoint_a[-1, 1],
        endpoint_b[-1, 1],
    ],
    [
        endpoint_a[-1, 2],
        endpoint_b[-1, 2],
    ],
    linewidth=3.0,
    label="End",
)

ax.plot(
    roller_pos[:, 0],
    roller_pos[:, 1],
    roller_pos[:, 2],
    linewidth=1.0,
    alpha=0.7,
    label="Roller origin",
)

all_points = np.concatenate(
    [
        endpoint_a,
        endpoint_b,
        roller_pos,
    ],
    axis=0,
)

set_axes_equal(
    ax,
    all_points,
)

ax.set_xlabel("World X [m]")
ax.set_ylabel("World Y [m]")
ax.set_zlabel("World Z [m]")

ax.set_title(
    "Roller Head Centerline Swept Surface\n"
    "(roller radius not included)"
)

ax.legend()
ax.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

print(
    "figure saved     :",
    OUTPUT_PATH,
)

plt.show()