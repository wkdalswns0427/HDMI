"""Robot presets: MuJoCo model + the joint/body ordering HDMI expects.

`joint_names` and `out_body_names` are copied from
HDMI/active_adaptation/utils/motion.py so the emitted meta.json lines up with
what the trainer resolves against the Isaac asset.

`fk_body_names` is a superset used only during FK, so annotations can attach an
object to a link (a rubber hand, say) that is not itself written to the file.
"""

from __future__ import annotations
from pathlib import Path

GMR_ASSETS = Path("/home/mchang344/mj_ws/simbench/GMR/assets")

G1_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_BODIES = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]


# H1-2 "handless": 27 actuated joints. Note the leg order is yaw/pitch/roll
# (G1 is pitch/roll/yaw), and the three G1 waist joints collapse to one
# `torso_joint` -- matching the note in HDMI/active_adaptation/assets/h1_2.py.
# Order below is the MuJoCo qpos order, which is what GMR's dof_pos follows.
H1_2_JOINTS = [
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

H1_2_BODIES = [
    "pelvis",
    "left_hip_yaw_link", "left_hip_pitch_link", "left_hip_roll_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_yaw_link", "right_hip_pitch_link", "right_hip_roll_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

ROBOTS = {
    "unitree_g1": {
        "xml": GMR_ASSETS / "unitree_g1" / "g1_mocap_29dof.xml",
        "joint_names": G1_JOINTS,
        "out_body_names": G1_BODIES,
        # rubber hands exist in the MuJoCo model and are the natural grasp
        # frames, so keep them available for `attach` segments.
        "fk_body_names": G1_BODIES + ["left_rubber_hand", "right_rubber_hand"],
        "root_body": "pelvis",
    },
    "unitree_h1_2": {
        "xml": GMR_ASSETS / "unitree_h1_2" / "h1_2_handless.xml",
        "joint_names": H1_2_JOINTS,
        "out_body_names": H1_2_BODIES,
        # No rubber-hand bodies on the handless model; wrists are the grasp frames.
        "fk_body_names": H1_2_BODIES,
        "root_body": "pelvis",
    },
}


def get(name: str) -> dict:
    if name not in ROBOTS:
        raise KeyError(f"unknown robot {name!r}; known: {sorted(ROBOTS)}")
    cfg = dict(ROBOTS[name])
    if not Path(cfg["xml"]).exists():
        raise FileNotFoundError(f"MuJoCo model missing: {cfg['xml']}")
    return cfg
