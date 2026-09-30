# Unitree H1-2 (27 DoF, handless).
#
# Motor groupings, armatures and effort limits follow Unitree's own mjlab
# constants (unitree_rl_mjlab/src/assets/robots/unitree_h1_2/h1_2_constants.py).
# Stiffness/damping are regenerated here from the same natural-frequency rule
# g1.py uses, which reproduces the mjlab numbers to within rounding:
#   M107_24_2 (armature 0.025) -> K 98.7,  D 6.3
#   M107_24_1 (armature 0.04)  -> K 157.9, D 10.1
#   GO2HV_1   (armature 0.005) -> K 19.7,  D 1.3
#   GO2HV_2   (armature 0.002) -> K 7.9,   D 0.5
# Per-joint effort/velocity limits are taken from h1_2_handless.urdf.
#
# Kinematic delta vs G1: identical 28 link names, but the 3-DoF waist
# (waist_yaw/roll/pitch_joint) is replaced by a single torso_joint.
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os

ASSET_PATH = os.path.dirname(__file__)

ARMATURE_M107_24_2 = 0.025   # hip yaw/pitch/roll, torso
ARMATURE_M107_24_1 = 0.04    # knee
ARMATURE_GO2HV_1 = 0.005     # ankles, shoulder pitch/roll
ARMATURE_GO2HV_2 = 0.002     # shoulder yaw, elbow, wrists

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_M107_24_2 = ARMATURE_M107_24_2 * NATURAL_FREQ**2
STIFFNESS_M107_24_1 = ARMATURE_M107_24_1 * NATURAL_FREQ**2
STIFFNESS_GO2HV_1 = ARMATURE_GO2HV_1 * NATURAL_FREQ**2
STIFFNESS_GO2HV_2 = ARMATURE_GO2HV_2 * NATURAL_FREQ**2

DAMPING_M107_24_2 = 2.0 * DAMPING_RATIO * ARMATURE_M107_24_2 * NATURAL_FREQ
DAMPING_M107_24_1 = 2.0 * DAMPING_RATIO * ARMATURE_M107_24_1 * NATURAL_FREQ
DAMPING_GO2HV_1 = 2.0 * DAMPING_RATIO * ARMATURE_GO2HV_1 * NATURAL_FREQ
DAMPING_GO2HV_2 = 2.0 * DAMPING_RATIO * ARMATURE_GO2HV_2 * NATURAL_FREQ

H1_2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}" + "/h1_2/{ROBOT_TYPE}.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.02),
        joint_pos={
            ".*_hip_pitch_joint": -0.2,
            ".*_knee_joint": 0.5,
            ".*_ankle_pitch_joint": -0.3,
            ".*_elbow_joint": 0.52,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.28,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.28,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 300.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 23.0,
                ".*_hip_roll_joint": 23.0,
                ".*_hip_pitch_joint": 23.0,
                ".*_knee_joint": 14.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_M107_24_2,
                ".*_hip_roll_joint": STIFFNESS_M107_24_2,
                ".*_hip_yaw_joint": STIFFNESS_M107_24_2,
                ".*_knee_joint": STIFFNESS_M107_24_1,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_M107_24_2,
                ".*_hip_roll_joint": DAMPING_M107_24_2,
                ".*_hip_yaw_joint": DAMPING_M107_24_2,
                ".*_knee_joint": DAMPING_M107_24_1,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_M107_24_2,
                ".*_hip_roll_joint": ARMATURE_M107_24_2,
                ".*_hip_yaw_joint": ARMATURE_M107_24_2,
                ".*_knee_joint": ARMATURE_M107_24_1,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 60.0,
                ".*_ankle_roll_joint": 40.0,
            },
            velocity_limit_sim=9.0,
            stiffness=STIFFNESS_GO2HV_1,
            damping=DAMPING_GO2HV_1,
            armature=ARMATURE_GO2HV_1,
        ),
        # H1-2 has a single 1-DoF torso yaw where G1 has a 3-DoF waist.
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=23.0,
            stiffness=STIFFNESS_M107_24_2,
            damping=DAMPING_M107_24_2,
            armature=ARMATURE_M107_24_2,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 18.0,
                ".*_elbow_joint": 18.0,
                ".*_wrist_roll_joint": 19.0,
                ".*_wrist_pitch_joint": 19.0,
                ".*_wrist_yaw_joint": 19.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 9.0,
                ".*_shoulder_roll_joint": 9.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*_elbow_joint": 20.0,
                ".*_wrist_roll_joint": 31.4,
                ".*_wrist_pitch_joint": 31.4,
                ".*_wrist_yaw_joint": 31.4,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_GO2HV_1,
                ".*_shoulder_roll_joint": STIFFNESS_GO2HV_1,
                ".*_shoulder_yaw_joint": STIFFNESS_GO2HV_2,
                ".*_elbow_joint": STIFFNESS_GO2HV_2,
                ".*_wrist_roll_joint": STIFFNESS_GO2HV_2,
                ".*_wrist_pitch_joint": STIFFNESS_GO2HV_2,
                ".*_wrist_yaw_joint": STIFFNESS_GO2HV_2,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_GO2HV_1,
                ".*_shoulder_roll_joint": DAMPING_GO2HV_1,
                ".*_shoulder_yaw_joint": DAMPING_GO2HV_2,
                ".*_elbow_joint": DAMPING_GO2HV_2,
                ".*_wrist_roll_joint": DAMPING_GO2HV_2,
                ".*_wrist_pitch_joint": DAMPING_GO2HV_2,
                ".*_wrist_yaw_joint": DAMPING_GO2HV_2,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_GO2HV_1,
                ".*_shoulder_roll_joint": ARMATURE_GO2HV_1,
                ".*_shoulder_yaw_joint": ARMATURE_GO2HV_2,
                ".*_elbow_joint": ARMATURE_GO2HV_2,
                ".*_wrist_roll_joint": ARMATURE_GO2HV_2,
                ".*_wrist_pitch_joint": ARMATURE_GO2HV_2,
                ".*_wrist_yaw_joint": ARMATURE_GO2HV_2,
            },
        ),
    },
)

H1_2_ACTION_SCALE = {}
for a in H1_2_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            H1_2_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
