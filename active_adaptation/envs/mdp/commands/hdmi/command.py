from active_adaptation.envs.mdp.base import Command
from active_adaptation.utils.motion import MotionDataset, MotionData

from typing import List, Dict, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor
    from isaaclab.assets import Articulation, RigidObject

import torch
import numpy as np
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz, quat_mul, quat_apply, quat_apply_inverse, quat_from_matrix
from tensordict import TensorDict
from omegaconf import ListConfig
from active_adaptation.utils.math import batchify
quat_apply = batchify(quat_apply)
quat_apply_inverse = batchify(quat_apply_inverse)
torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)

class RobotTracking(Command):
    def __init__(
        self, env, data_path: List[str] | str,
        tracking_keypoint_names: List[str],
        tracking_joint_names: List[str],
        # reset parameters
        root_body_name: str = "pelvis",
        debug_vis: bool = True,   # False -> no keypoint/contact markers (clean renders)
        reset_range: Tuple[float, float] | None = None,
        pose_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0., 0.),
            "pitch": (-0., 0.),
            "yaw": (-0., 0.)},
        velocity_range: Dict[str, Tuple[float, float]] = {
            "x": (-0., 0.),
            "y": (-0., 0.),
            "z": (-0., 0.),
            "roll": (-0., 0.),
            "pitch": (-0., 0.),
            "yaw": (-0., 0.)},
        init_joint_pos_noise: float = 0.0,
        init_joint_vel_noise: float = 0.0,
        # observation parameters
        future_steps: List[int] = [1, 2, 8, 16],
        call_update: bool = True,
        sample_motion: bool = False,
        replay_motion: bool = False,
        record_motion: bool = False,
    ):
        from . import observations
        from . import rewards
        from . import randomizations
        from . import terminations
        super().__init__(env)
        self.dataset = MotionDataset.create_from_path(
            data_path,
            isaac_joint_names=self.asset.joint_names,
            target_fps=int(1/self.env.step_dt)
        ).to(self.device)

        # Set tracking body and joint names for observation and termination
        self.tracking_keypoint_names = self.asset.find_bodies(tracking_keypoint_names)[1]
        self.tracking_body_indices_motion = [self.dataset.body_names.index(name) for name in self.tracking_keypoint_names]
        self.tracking_body_indices_asset = [self.asset.body_names.index(name) for name in self.tracking_keypoint_names]

        self.tracking_joint_names = self.asset.find_joints(tracking_joint_names)[1]
        self.tracking_joint_indices_motion = [self.dataset.joint_names.index(name) for name in self.tracking_joint_names]
        self.tracking_joint_indices_asset = [self.asset.joint_names.index(name) for name in self.tracking_joint_names]

        self.num_tracking_bodies = len(self.tracking_body_indices_asset)
        self.num_tracking_joints = len(self.tracking_joint_indices_asset)
        self.num_future_steps = len(future_steps)

        # get root body and joint indices in motion for reset
        self.root_body_name = root_body_name
        self.root_body_idx_motion = self.dataset.body_names.index(root_body_name)
        
        asset_joint_names = self.asset.joint_names
        self.asset_joint_idx_motion = [self.dataset.joint_names.index(joint_name) for joint_name in asset_joint_names]

        with torch.device(self.device):
            self.is_standing_env = torch.zeros(self.num_envs, 1, dtype=bool)
            self.future_steps = torch.tensor(future_steps)

            self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_len = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_starts = torch.zeros(self.num_envs, dtype=torch.long)
            self.motion_ends = torch.zeros(self.num_envs, dtype=torch.long)
            self.t = torch.zeros(self.num_envs, dtype=torch.long)
            self.replay_motion_t = torch.zeros(self.num_envs, dtype=torch.long)

            self.eval_t = torch.randint(0, self.dataset.lengths[0], (self.num_envs,), device=self.device)

        self.reset_range = reset_range

        pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.pose_range = torch.tensor(pose_range_list, device=self.device)
        velocity_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.velocity_range = torch.tensor(velocity_range_list, device=self.device)

        self.init_joint_pos_noise = init_joint_pos_noise
        self.init_joint_vel_noise = init_joint_vel_noise

        self.first_sample_motion = True
        self.sample_motion = sample_motion
        self.replay_motion = replay_motion
        self.record_motion = record_motion

        if self.replay_motion:
            self.pose_range.fill_(0.0)
            self.init_joint_pos_noise = 0.0
            self.init_joint_vel_noise = 0.0
        
        if self.record_motion:
            assert self.num_envs == 1, "record_motion only supports num_envs=1"
            self.pose_range.fill_(0.0)
            self.init_joint_pos_noise = 0.0
            self.init_joint_vel_noise = 0.0

        self.debug_vis = debug_vis
        if call_update:
            self._init_debug_draw()
            self.update()
            if self.record_motion:
                self.motion_frames = []
        
    def _sample_motions(self, env_ids: torch.Tensor) -> None:
        if self.sample_motion or self.first_sample_motion:
            # sample motion id and start time for each env
            motion_ids = torch.randint(0, self.dataset.num_motions, size=(len(env_ids),), device=self.device)
            self.motion_ids[env_ids] = motion_ids
            self.motion_len[env_ids] = motion_len = self.dataset.lengths[motion_ids]
            self.motion_starts[env_ids] = self.dataset.starts[motion_ids]
            self.motion_ends[env_ids] = self.dataset.ends[motion_ids]
            self.first_sample_motion = False
        else:
            motion_len = self.motion_len[env_ids]

        if self.reset_range is None:
            max_len = motion_len - self.future_steps[-1]
            start_phase = torch.rand(len(env_ids), device=self.device)
            start_t = (start_phase * max_len).long()
        else:
            start_t = torch.randint(*self.reset_range, (len(env_ids),), device=self.device)
            
        if not self.env.training or self.record_motion:
            start_t.fill_(0)

        if self.replay_motion:
            self.replay_motion_t[env_ids] = (self.replay_motion_t[env_ids] + 1) % motion_len
            start_t = self.replay_motion_t[env_ids]

        self.t[env_ids] = start_t


    def sample_init(self, env_ids: torch.Tensor) -> None:
        self._sample_motions(env_ids)

        # reset root state and joint position/velocity from motion
        self._motion_reset: MotionData = self.dataset.get_slice(self.motion_ids[env_ids], self.t[env_ids], 1).squeeze(1)
        # shape: [len(env_ids), num_bodies/num_joints, 3/4/...]
        
        motion = self._motion_reset
        init_root_pos = motion.body_pos_w[:, self.root_body_idx_motion]
        init_root_quat = motion.body_quat_w[:, self.root_body_idx_motion]
        init_root_lin_vel = motion.body_lin_vel_w[:, self.root_body_idx_motion]
        init_root_ang_vel = motion.body_ang_vel_w[:, self.root_body_idx_motion]

        # poses
        rand_samples = sample_uniform(self.pose_range[:, 0], self.pose_range[:, 1], (len(env_ids), 6), device=self.device)
        if not self.env.training:
            rand_samples.fill_(0.0)
        positions = init_root_pos + self.env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = quat_mul(init_root_quat, orientations_delta)

        # velocities
        rand_samples = sample_uniform(self.velocity_range[:, 0], self.velocity_range[:, 1], (len(env_ids), 6), device=self.device)
        if not self.env.training:
            rand_samples.fill_(0.0)
        velocities = torch.cat([init_root_lin_vel, init_root_ang_vel], dim=-1) + rand_samples

        self.asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        self.asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)

        init_joint_pos = motion.joint_pos[:, self.asset_joint_idx_motion]
        init_joint_vel = motion.joint_vel[:, self.asset_joint_idx_motion]

        joint_pos_noise = sample_uniform(-1, 1, (init_joint_pos.shape[0], init_joint_pos.shape[1]), device=self.device) * self.init_joint_pos_noise
        joint_vel_noise = sample_uniform(-1, 1, (init_joint_vel.shape[0], init_joint_vel.shape[1]), device=self.device) * self.init_joint_vel_noise

        init_joint_pos += joint_pos_noise
        init_joint_vel += joint_vel_noise

        joint_pos_limits = self.asset.data.soft_joint_pos_limits[env_ids]
        joint_vel_limits = self.asset.data.soft_joint_vel_limits[env_ids]
        init_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        init_joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

        self.asset.write_joint_state_to_sim(init_joint_pos, init_joint_vel, env_ids=env_ids)

        if self.record_motion:
            if len(self.motion_frames) > 0:
                self._save_motion()
                self.motion_frames = []
    
    def _save_motion(self):
        motion_data: TensorDict = torch.cat(self.motion_frames, dim=0)
        motion_data = motion_data[25:].numpy()
        moton_meta = {
            "joint_names": self.asset.joint_names,
            "body_names": self.asset.body_names,
            "fps": int(1/self.env.step_dt),
        }
        save_dir = "record_motion"
        motion_data_path = f"{save_dir}/motion.npz"
        motion_meta_path = f"{save_dir}/meta.json"
        import os, json
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(motion_data_path, **motion_data)
        with open(motion_meta_path, "w") as f:
            json.dump(moton_meta, f, indent=4)
        print(f"Saved recorded motion to {motion_data_path} and {motion_meta_path}")
        # upstream left a bare breakpoint() here, which drops every recording run
        # into pdb after the save completes.
            

    @property
    def success(self):
        return (self.t >= self.motion_len - 1).unsqueeze(1)
    
    @property
    def finished(self):
        if self.replay_motion:
            return torch.ones(self.num_envs, 1, dtype=bool, device=self.device)
        return (self.t >= self.motion_len).unsqueeze(1)

    def update(self):
        if hasattr(self, "motion_frames"):
            motion_frame = {}
            motion_frame["body_pos_w"] = self.asset.data.body_link_pos_w.cpu()
            motion_frame["body_quat_w"] = self.asset.data.body_link_quat_w.cpu()
            motion_frame["body_lin_vel_w"] = self.asset.data.body_com_lin_vel_w.cpu()
            motion_frame["body_ang_vel_w"] = self.asset.data.body_com_ang_vel_w.cpu()
            motion_frame["joint_pos"] = self.asset.data.joint_pos.cpu()
            motion_frame["joint_vel"] = self.asset.data.joint_vel.cpu()
            self.motion_frames.append(TensorDict(motion_frame, batch_size=[1]))
            
        # future ref motion for actor observation
        self.future_ref_motion = self.dataset.get_slice(self.motion_ids, self.t, steps=self.future_steps)
        # shape: [num_envs, len(future_steps), num_bodies/num_joints, 3/4/...]

        # Observations: future ref body and joint states
        self.ref_body_pos_future_w = self.future_ref_motion.body_pos_w[..., self.tracking_body_indices_motion, :] + self.env.scene.env_origins[:, None, None, :]
        self.ref_body_lin_vel_future_w = self.future_ref_motion.body_lin_vel_w[..., self.tracking_body_indices_motion, :]
        self.ref_body_quat_future_w = self.future_ref_motion.body_quat_w[..., self.tracking_body_indices_motion, :]
        self.ref_body_ang_vel_future_w = self.future_ref_motion.body_ang_vel_w[..., self.tracking_body_indices_motion, :]
        self.ref_joint_pos_future_ = self.future_ref_motion.joint_pos[..., self.tracking_joint_indices_motion]
        self.ref_joint_vel_future_ = self.future_ref_motion.joint_vel[..., self.tracking_joint_indices_motion]
        self.ref_root_pos_future_w = self.future_ref_motion.body_pos_w[..., self.root_body_idx_motion, :] + self.env.scene.env_origins[:, None, :]
        self.ref_root_quat_future_w = self.future_ref_motion.body_quat_w[..., self.root_body_idx_motion, :]
        self.ref_root_lin_vel_future_w = self.future_ref_motion.body_lin_vel_w[..., self.root_body_idx_motion, :]
        self.ref_root_ang_vel_future_w = self.future_ref_motion.body_ang_vel_w[..., self.root_body_idx_motion, :]

        # Reward: current robot body and joint states
        self.robot_body_pos_w = self.asset.data.body_link_pos_w[:, self.tracking_body_indices_asset]
        self.robot_body_lin_vel_w = self.asset.data.body_com_lin_vel_w[:, self.tracking_body_indices_asset]
        self.robot_body_quat_w = self.asset.data.body_link_quat_w[:, self.tracking_body_indices_asset]
        self.robot_body_ang_vel_w = self.asset.data.body_com_ang_vel_w[:, self.tracking_body_indices_asset]
        self.robot_joint_pos = self.asset.data.joint_pos[:, self.tracking_joint_indices_asset]
        self.robot_joint_vel = self.asset.data.joint_vel[:, self.tracking_joint_indices_asset]
        self.robot_root_pos_w = self.asset.data.root_link_pos_w
        self.robot_root_quat_w = self.asset.data.root_link_quat_w

        # Reward: current ref body and joint states
        self.current_ref_motion: MotionData = self.future_ref_motion[:, 0]
        self.ref_body_pos_w = self.ref_body_pos_future_w[:, 0]
        self.ref_body_lin_vel_w = self.ref_body_lin_vel_future_w[:, 0]
        self.ref_body_quat_w = self.ref_body_quat_future_w[:, 0]
        self.ref_body_ang_vel_w = self.ref_body_ang_vel_future_w[:, 0]
        self.ref_joint_pos = self.ref_joint_pos_future_[:, 0]
        self.ref_joint_vel = self.ref_joint_vel_future_[:, 0]
        self.ref_root_pos_w = self.ref_root_pos_future_w[:, 0]
        self.ref_root_quat_w = self.ref_root_quat_future_w[:, 0]
        # shape: [num_envs, num_future_steps, num_tracking_bodies, xxx]

        if self.env.backend == "isaac":
            self.all_marker_pos_w[0] = self.robot_body_pos_w
            self.all_marker_pos_w[1] = self.ref_body_pos_w
            # self.all_marker_pos_w[0] = self.ref_body_pos_future_w[:, 0]
            # self.all_marker_pos_w[1] = self.ref_body_pos_future_w[:, -1]

        self.t += 1
    
    def _init_debug_draw(self):
        if self.env.backend != "isaac":
            return
        
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils
        vis_markers_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Keypoints",
            markers={
                "robot": sim_utils.SphereCfg(
                    radius=0.04,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0)
                    ),
                ),
                "reference": sim_utils.SphereCfg(
                    radius=0.04,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0)
                    ),
                ),
            },
        )
        self.vis_markers = VisualizationMarkers(vis_markers_cfg)
        num_ref_markers = self.num_envs * self.num_tracking_bodies
        self.marker_indices = [0] * num_ref_markers + [1] * num_ref_markers
        self.all_marker_pos_w = torch.zeros(2, self.num_envs, self.num_tracking_bodies, 3, device=self.device)
        if not getattr(self, "debug_vis", True):
            self.vis_markers.set_visibility(False)

    def debug_draw(self):
        if self.env.backend != "isaac" or not getattr(self, "debug_vis", True):
            return

        if self.replay_motion:
            self.all_marker_pos_w.fill_(-1000)
        
        # shape: [2, num_envs, num_tracking_bodies, 3]
        self.vis_markers.visualize(
            translations=self.all_marker_pos_w.reshape(-1, 3),
            marker_indices=self.marker_indices,
        )

        # robot_keypoints_w = self.all_marker_pos_w[0].reshape(-1, 3)
        # target_keypoints_w = self.all_marker_pos_w[1].reshape(-1, 3)
        # self.env.debug_draw.vector(
        #     robot_keypoints_w,
        #     target_keypoints_w - robot_keypoints_w,
        #     color=(0, 0, 1, 1)
        # )

class RobotObjectTracking(RobotTracking):
    def __init__(
        self,
        extra_object_names: List[str],
        object_asset_name: str, # for finding the object in the scene
        object_body_name: str, # for the body that defines the contact target position
        object_joint_name: str | None = None, # object joint to track
        # for reset
        object_pose_range: Dict[str, Tuple[float, float]] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0., 0.),
            "pitch": (-0., 0.),
            "yaw": (-0., 0.)},
        object_init_joint_pos_noise: float = 0.1, 
        object_init_joint_vel_noise: float = 0.1,
        # for contact rewards
        contact_eef_body_name: List[str] = ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        contact_frc_eef_body_name: List[str | List[str]] = ["left_wrist_(roll|pitch|yaw)_link", "right_wrist_(roll|pitch|yaw)_link"],
        ## offset from object to contact target position
        contact_target_pos_offset: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        ## offset from end effector
        contact_eef_pos_offset: List[Tuple[float, float, float]] = [(0.1, 0.0, 0.0), (0.1, 0.0, 0.0)],
        # wall-painting target region
        target_region_path: str | None = None,
        target_region_pos_range: Dict[str, Tuple[float, float]] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
        },
        target_region_scale_range: Dict[str, Tuple[float, float]] = {
            "width": (1.0, 1.0),
            "height": (1.0, 1.0),
        },
        # Slide the target ALONG the wall. target_region_pos_range is a world
        # xyz offset, so randomizing x or y there pushes the region off the
        # wall plane; u/v move within it -- u along the in-plane horizontal,
        # v vertically. Use this for "anywhere on the wall", and pos_range
        # only when you really mean to move the plane itself.
        target_region_uv_range: Dict[str, Tuple[float, float]] = {
            "u": (0.0, 0.0),
            "v": (0.0, 0.0),
        },
        coverage_grid_size: int = 4,
        show_canvas: bool = True,
        paint_resolution: float = 0.005,
        roller_head_length: float = 0.22913713,
        # roller_head_radius: float = 0.042458495,
        **kwargs
    ):
        super().__init__(**kwargs, call_update=False)

        self.extra_objects: List[Articulation | RigidObject] = [self.env.scene[name] for name in extra_object_names]
        self.extra_object_body_id_motion = [self.dataset.body_names.index(name) for name in extra_object_names]

        self.object_asset_name = object_asset_name
        if object_joint_name is None:
            self.object = self.env.scene.rigid_objects[object_asset_name]
            self.object_joint_idx_motion = None
            self.object_joint_idx_asset = None
        else:
            self.object = self.env.scene.articulations[object_asset_name]
            self.object_joint_idx_motion = self.dataset.joint_names.index(object_joint_name)
            self.object_joint_idx_asset = self.object.joint_names.index(object_joint_name)
        
        self.object_body_id_asset = self.object.body_names.index(object_body_name)
        self.object_body_id_motion = self.dataset.body_names.index(object_asset_name)

        pose_range_list = [object_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.object_pose_range = torch.tensor(pose_range_list, device=self.device)
        self.object_init_joint_pos_noise = object_init_joint_pos_noise
        self.object_init_joint_vel_noise = object_init_joint_vel_noise

        if self.replay_motion or self.record_motion:
            self.object_pose_range.fill_(0.0)
            self.object_init_joint_pos_noise = 0.0
            self.object_init_joint_vel_noise = 0.0

        # setup contact body indices
        assert len(contact_eef_body_name) == len(contact_target_pos_offset) == len(contact_eef_pos_offset), \
            "contact_eef_body_name, contact_target_pos_offset, and contact_eef_pos_offset must have the same length"
        self.num_eefs = len(contact_eef_body_name)
        self.contact_eef_body_indices_asset = [self.asset.body_names.index(name) for name in contact_eef_body_name]

        self.eef_filtered_sensor: List[List[ContactSensor]] = []
        # [self.env.scene.sensors[f"{eef_name}_{object_asset_name}_contact_forces"] for eef_name in contact_eef_body_name] for object_asset_name in self.asset.data.object_names]
        self.eef_filtered_sensor_indices: List[List[int]] = []
        # = [eef_sensor.body_names.index(eef_name) for (eef_name, eef_sensor) in zip(contact_eef_body_name, self.eef_object_contact_forces)]
        for eef_name in contact_eef_body_name:
            eef_names = self.asset.find_bodies(eef_name)[1]
            sensors_for_this_eef = []
            sensor_indices_for_this_eef = []
            for eef_name in eef_names:
                eef_sensor_name = f"{eef_name}_{object_asset_name}_contact_forces"
                eef_sensor_filtered = self.env.scene.sensors[eef_sensor_name]
                sensors_for_this_eef.append(eef_sensor_filtered)
                sensor_indices_for_this_eef.append(eef_sensor_filtered.body_names.index(eef_name))
            self.eef_filtered_sensor.append(sensors_for_this_eef)
            self.eef_filtered_sensor_indices.append(sensor_indices_for_this_eef)

        with torch.device(self.device):
            self.contact_target_pos_offset = torch.tensor(contact_target_pos_offset, device=self.device).repeat(self.num_envs, 1, 1)
            self.contact_eef_pos_offset = torch.tensor(contact_eef_pos_offset, device=self.device).repeat(self.num_envs, 1, 1)

            self.contact_target_pos_w = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)
            self.contact_eef_pos_w = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)

            self.eef_contact_forces_w = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)
            self.eef_contact_forces_b = torch.zeros(self.num_envs, len(contact_eef_body_name), 3, device=self.device)
        
        scale = getattr(self.object.cfg.spawn, "scale", None)
        if not isinstance(scale, torch.Tensor):
            scale_tensor = torch.ones(self.num_envs, 3)
            if scale is None:
                pass
            elif isinstance(scale, float):
                scale_tensor[:] = scale
            elif isinstance(scale, tuple):
                scale_tensor[:] = torch.tensor(scale)
            else:
                raise ValueError(f"Invalid scale type: {type(scale)}")
            scale = scale_tensor
        self.contact_target_pos_offset *= scale.unsqueeze(1).to(self.device)

        self.target_region_path = target_region_path
        self.coverage_grid_size = coverage_grid_size
        self.show_canvas = show_canvas
        # Present only when locomotion.py spawned it (painting tasks, canvas on).
        self.canvas = self.env.scene.rigid_objects.get("canvas", None)

        if self.target_region_path is not None:
            # One rectangle per motion, in the same order as data_path. A single
            # string stays a one-element list, so the single-clip case is
            # unchanged: motion_ids are all zero and every lookup returns row 0.
            region_paths = (
                list(self.target_region_path)
                if isinstance(self.target_region_path, (list, ListConfig))
                else [self.target_region_path]
            )
            num_motions = self.dataset.num_motions
            assert len(region_paths) == num_motions, (
                f"target_region_path has {len(region_paths)} entries but data_path "
                f"matched {num_motions} motions. Give one rectangle per motion, in "
                f"the same order, or one rectangle total for a single motion."
            )
            # Relative paths resolve against the repo root, the same way
            # MotionDataset resolves data_path. A plain relative np.load would
            # break: hydra runs with job.chdir=true, so cwd is the output dir.
            import active_adaptation
            from pathlib import Path
            repo_root = Path(active_adaptation.__file__).parent.parent
            region_paths = [
                p if Path(p).is_absolute() else str(repo_root / p)
                for p in region_paths
            ]
            region_data = [np.load(p, allow_pickle=True) for p in region_paths]

            # Nominal target region per motion, extracted from each demonstration.
            self.nominal_target_center_w = torch.tensor(
                np.stack([d["center_world"] for d in region_data]),
                dtype=torch.float32,
                device=self.device,
            )                                       # [num_motions, 3]

            self.nominal_target_width = torch.tensor(
                [float(np.asarray(d["width"]).squeeze()) for d in region_data],
                dtype=torch.float32,
                device=self.device,
            )                                       # [num_motions]
            self.nominal_target_height = torch.tensor(
                [float(np.asarray(d["height"]).squeeze()) for d in region_data],
                dtype=torch.float32,
                device=self.device,
            )                                       # [num_motions]

            # Wall coordinate axes, used to reconstruct the rectangle after
            # scaling and to project the roller into (u, v). Shared across
            # motions: axis_v is [0,0,1] by construction and axis_u is the
            # in-plane horizontal, so clips filmed against the same wall from the
            # same standoff agree. Keeping them shared avoids per-env axes in the
            # rasterizer; the check below fails loudly if the walls disagree.
            for i, d in enumerate(region_data[1:], start=1):
                for key in ("axis_u", "axis_v"):
                    if not np.allclose(d[key], region_data[0][key], atol=1e-3):
                        raise ValueError(
                            f"{key} of {region_paths[i]} differs from "
                            f"{region_paths[0]} by more than 1e-3. The clips were "
                            f"filmed against different wall planes, so their "
                            f"rectangles are not comparable. Re-shoot from the same "
                            f"standoff, or extend the rasterizer to per-motion axes."
                        )

            self.wall_axis_u_w = torch.tensor(
                region_data[0]["axis_u"],
                dtype=torch.float32,
                device=self.device,
            )
            self.wall_axis_v_w = torch.tensor(
                region_data[0]["axis_v"],
                dtype=torch.float32,
                device=self.device,
            )

            # Position randomization range in world coordinates.
            pos_range_list = [
                target_region_pos_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z"]
            ]
            self.target_region_pos_range = torch.tensor(
                pos_range_list,
                dtype=torch.float32,
                device=self.device,
            )

            # In-plane displacement range, in wall coordinates.
            uv_range_list = [
                target_region_uv_range.get(key, (0.0, 0.0))
                for key in ["u", "v"]
            ]
            self.target_region_uv_range = torch.tensor(
                uv_range_list,
                dtype=torch.float32,
                device=self.device,
            )

            # Width/height scaling range.
            scale_range_list = [
                target_region_scale_range.get(key, (1.0, 1.0))
                for key in ["width", "height"]
            ]
            self.target_region_scale_range = torch.tensor(
                scale_range_list,
                dtype=torch.float32,
                device=self.device,
            )

            # Per-environment target region.
            self.target_center_w = torch.zeros(
                self.num_envs, 3,
                dtype=torch.float32,
                device=self.device,
            )
            self.target_corners_w = torch.zeros(
                self.num_envs, 4, 3,
                dtype=torch.float32,
                device=self.device,
            )
            self.target_width = torch.zeros(
                self.num_envs, 1,
                dtype=torch.float32,
                device=self.device,
            )
            self.target_height = torch.zeros(
                self.num_envs, 1,
                dtype=torch.float32,
                device=self.device,
            )

            self.paint_resolution = paint_resolution

            # Roller-head geometry in the roller object's local frame.
            self.roller_head_length = roller_head_length
            # self.roller_head_radius = roller_head_radius

            self.roller_head_center_b = torch.tensor(
                [-0.000457165, 0.00020581, 0.270268835],
                dtype=torch.float32,
                device=self.device,
            )

            self.roller_head_axis_b = torch.tensor(
                [1.0, 0.0, 0.0],
                dtype=torch.float32,
                device=self.device,
            )

            # Previous roller-head segment in wall coordinates.
            # Used to rasterize the swept area between simulation steps.
            self.prev_head_a_uv = torch.zeros(
                self.num_envs, 2,
                dtype=torch.float32,
                device=self.device,
            )

            self.prev_head_b_uv = torch.zeros(
                self.num_envs, 2,
                dtype=torch.float32,
                device=self.device,
            )

            self.prev_head_valid = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )

            # Use a fixed raster large enough for the maximum randomized
            # target-region size.
            # Sized for the largest rectangle any motion can produce, so the
            # fixed raster fits every env regardless of which clip it drew.
            max_width = (
                float(self.nominal_target_width.max())
                * float(self.target_region_scale_range[0, 1])
            )
            max_height = (
                float(self.nominal_target_height.max())
                * float(self.target_region_scale_range[1, 1])
            )

            self.paint_grid_width = int(
                np.ceil(max_width / self.paint_resolution)
            )
            self.paint_grid_height = int(
                np.ceil(max_height / self.paint_resolution)
            )

            # Cell indices of the fixed raster, reused every step so the
            # sweep rasterizer never rebuilds a meshgrid.
            self._row_idx = torch.arange(
                self.paint_grid_height, device=self.device, dtype=torch.float32
            )
            self._col_idx = torch.arange(
                self.paint_grid_width, device=self.device, dtype=torch.float32
            )

            # Active sub-raster per env; recomputed on reset in
            # _sample_target_region, constant in between.
            self._target_rows = torch.full(
                (self.num_envs,), self.paint_grid_height,
                dtype=torch.long, device=self.device,
            )
            self._target_cols = torch.full(
                (self.num_envs,), self.paint_grid_width,
                dtype=torch.long, device=self.device,
            )

            # True = this cell has already been painted.
            self.paint_mask = torch.zeros(
                self.num_envs,
                self.paint_grid_height,
                self.paint_grid_width,
                dtype=torch.bool,
                device=self.device,
            )

            # Coverage ratio after the current step.
            self.coverage_ratio = torch.zeros(
                self.num_envs, 1,
                dtype=torch.float32,
                device=self.device,
            )

            # Increase in coverage produced by the current step.
            self.coverage_delta = torch.zeros(
                self.num_envs, 1,
                dtype=torch.float32,
                device=self.device,
            )

        # Load object contact data. One array per motion, concatenated in the same
        # order MotionDataset used, because it is addressed below by the dataset's
        # global index (motion_starts + t), not by a per-motion index.
        motion_paths = self.dataset.motion_paths
        object_contact = np.concatenate(
            [np.load(p, allow_pickle=True)["object_contact"] for p in motion_paths],
            axis=0,
        )
        expected = int(self.dataset.ends[-1])
        assert object_contact.shape[0] == expected, (
            f"object_contact has {object_contact.shape[0]} frames but the concatenated "
            f"motion buffer has {expected}. The per-motion lengths must match "
            f"dataset.ends[i] - dataset.starts[i] AFTER the 50 fps resample; a clip "
            f"recorded at another fps needs its contact array resampled the same way "
            f"(nearest-neighbour on the bool array, not linear)."
        )
        self._object_contact = torch.tensor(object_contact, device=self.device, dtype=torch.bool)
        # if self._object_contact.shape[1] == 1:
        #     # expand to num_eefs
        #     self._object_contact = self._object_contact.repeat(1, self.num_eefs)
        # # shape: [num_steps, num_eefs/1]

        self._init_debug_draw()
        self.update()
        if self.record_motion:
            self.motion_frames = []

    def _sample_target_region(self, env_ids: torch.Tensor) -> None:
        if self.target_region_path is None:
            return

        num_envs = len(env_ids)

        # Random translation in world coordinates.
        pos_offset = sample_uniform(
            self.target_region_pos_range[:, 0],
            self.target_region_pos_range[:, 1],
            (num_envs, 3),
            device=self.device,
        )

        # Random width/height scaling.
        size_scale = sample_uniform(
            self.target_region_scale_range[:, 0],
            self.target_region_scale_range[:, 1],
            (num_envs, 2),
            device=self.device,
        )

        # Each env paints the rectangle belonging to the clip it drew.
        # motion_ids was set by super().sample_init() earlier in this reset.
        mids = self.motion_ids[env_ids]

        width = self.nominal_target_width[mids] * size_scale[:, 0]
        height = self.nominal_target_height[mids] * size_scale[:, 1]

        # NPZ coordinates correspond to the local world of one environment.
        # Add each parallel environment's origin.
        # In-plane slide, so the region stays ON the wall.
        uv_offset = sample_uniform(
            self.target_region_uv_range[:, 0],
            self.target_region_uv_range[:, 1],
            (num_envs, 2),
            device=self.device,
        )
        inplane = (
            uv_offset[:, 0:1] * self.wall_axis_u_w.unsqueeze(0)
            + uv_offset[:, 1:2] * self.wall_axis_v_w.unsqueeze(0)
        )

        center_w = (
            self.nominal_target_center_w[mids]
            + self.env.scene.env_origins[env_ids]
            + pos_offset
            + inplane
        )

        self.target_center_w[env_ids] = center_w
        self.target_width[env_ids, 0] = width
        self.target_height[env_ids, 0] = height

        # Reconstruct the four rectangle corners.
        half_u = (
            0.5
            * width.unsqueeze(1)
            * self.wall_axis_u_w.unsqueeze(0)
        )
        half_v = (
            0.5
            * height.unsqueeze(1)
            * self.wall_axis_v_w.unsqueeze(0)
        )

        self.target_corners_w[env_ids, 0] = center_w - half_u - half_v
        self.target_corners_w[env_ids, 1] = center_w + half_u - half_v
        self.target_corners_w[env_ids, 2] = center_w + half_u + half_v
        self.target_corners_w[env_ids, 3] = center_w - half_u + half_v

        # Active raster size for these envs, matching the offline rasterizer:
        # ceil(size / resolution), at least one cell, capped by the fixed raster.
        self._target_cols[env_ids] = torch.ceil(
            width / self.paint_resolution
        ).long().clamp(1, self.paint_grid_width)
        self._target_rows[env_ids] = torch.ceil(
            height / self.paint_resolution
        ).long().clamp(1, self.paint_grid_height)

        self._place_canvas(env_ids)

        # Reset accumulated paint coverage for these environments.
        # During replay_motion, each demonstration frame is loaded through a reset,
        # so preserve coverage state across those frame-by-frame resets.
        if not self.replay_motion:
            self.paint_mask[env_ids] = False
            self.coverage_ratio[env_ids] = 0.0
            self.coverage_delta[env_ids] = 0.0
            self.prev_head_valid[env_ids] = False

    def _place_canvas(self, env_ids: torch.Tensor) -> None:
        """Stand the wall on the ground, in the plane the target lives in.

        The wall is a fixed structure: the painted target is a region ON it and
        moves with target randomization, but the wall itself does not follow.
        A real wall does not slide up and down when the job does.

        The prim origin is authored at the bottom edge, so grounding it is just
        putting the origin at z = 0. Horizontally it sits under the nominal
        target of motion 0 -- with clips shot in a fixed environment they all
        share one wall plane, which is also what the axis_u/axis_v check at load
        enforces.

        The plate lies in its local XY plane, so the rotation whose columns are
        [axis_u, axis_v, normal] maps local X/Y/Z onto the wall's horizontal,
        vertical and outward directions. normal = axis_u x axis_v points back
        toward the human, which is the face that should be visible.
        """
        if self.canvas is None:
            return

        normal = torch.linalg.cross(self.wall_axis_u_w, self.wall_axis_v_w)
        normal = normal / normal.norm().clamp_min(1e-8)

        rot = torch.stack([self.wall_axis_u_w, self.wall_axis_v_w, normal], dim=-1)
        quat = quat_from_matrix(rot).unsqueeze(0).expand(len(env_ids), -1)

        base = self.nominal_target_center_w[0].clone()
        base[2] = 0.0                                   # origin = bottom edge
        pos = base.unsqueeze(0) + self.env.scene.env_origins[env_ids]

        pose = torch.cat([pos, quat], dim=-1)
        self.canvas.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        self.canvas.write_root_com_velocity_to_sim(
            torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
        )

    def _update_paint_coverage(self) -> None:
        """Rasterize the roller-head sweep into the per-env paint mask.

        Fully batched over envs: the raster size is fixed at construction, so the
        (u, v) cell centres and the four half-plane tests are evaluated for all
        envs at once. The half-plane test is separable -- cross = ex*(v-p0y) -
        ey*(u-p0x) -- so only one [N, H, W] tensor is materialized per edge.
        """
        if self.target_region_path is None:
            return

        num_envs = self.num_envs

        # 1. Roller-head centerline endpoints in world coordinates.
        head_center_offset_b = self.roller_head_center_b.unsqueeze(0).expand(num_envs, -1)
        head_axis_b = self.roller_head_axis_b.unsqueeze(0).expand(num_envs, -1)

        head_center_w = self.object_pos_w + quat_apply(self.object_quat_w, head_center_offset_b)

        head_axis_w = quat_apply(self.object_quat_w, head_axis_b)
        head_axis_w = head_axis_w / torch.linalg.norm(head_axis_w, dim=-1, keepdim=True).clamp_min(1e-8)

        half_length = 0.5 * self.roller_head_length
        endpoint_a_w = head_center_w - half_length * head_axis_w
        endpoint_b_w = head_center_w + half_length * head_axis_w

        # 2. Project both endpoints into the wall's (u, v) frame.
        axis_u = self.wall_axis_u_w.unsqueeze(0)
        axis_v = self.wall_axis_v_w.unsqueeze(0)

        rel_a = endpoint_a_w - self.target_center_w
        rel_b = endpoint_b_w - self.target_center_w

        curr_a = torch.stack([(rel_a * axis_u).sum(-1), (rel_a * axis_v).sum(-1)], dim=-1)
        curr_b = torch.stack([(rel_b * axis_u).sum(-1), (rel_b * axis_v).sum(-1)], dim=-1)

        old_coverage = self.coverage_ratio[:, 0].clone()

        # 3. Swept quad between the previous roller line and this one, ordered
        #    prev_a -> prev_b -> curr_b -> curr_a to match the offline script.
        quad = torch.stack([self.prev_head_a_uv, self.prev_head_b_uv, curr_b, curr_a], dim=1)
        nxt = torch.roll(quad, shifts=-1, dims=1)

        signed_area_twice = (quad[..., 0] * nxt[..., 1] - nxt[..., 0] * quad[..., 1]).sum(-1)
        orientation = torch.where(
            signed_area_twice >= 0.0,
            torch.ones_like(signed_area_twice),
            -torch.ones_like(signed_area_twice),
        ).view(num_envs, 1, 1)

        # 4. Cell centres of each env's active sub-raster, in wall coordinates.
        rows = self._target_rows
        cols = self._target_cols

        u = (self._col_idx.view(1, 1, -1) + 0.5) * self.paint_resolution \
            - 0.5 * self.target_width[:, 0].view(num_envs, 1, 1)
        v = (self._row_idx.view(1, -1, 1) + 0.5) * self.paint_resolution \
            - 0.5 * self.target_height[:, 0].view(num_envs, 1, 1)

        # 5. Four half-plane tests, one [N, H, W] temporary each.
        swept = (self._col_idx.view(1, 1, -1) < cols.view(num_envs, 1, 1)) & \
                (self._row_idx.view(1, -1, 1) < rows.view(num_envs, 1, 1))

        for edge_idx in range(4):
            p0 = quad[:, edge_idx]
            p1 = quad[:, (edge_idx + 1) % 4]

            # Fold the orientation sign into the edge so no extra [N, H, W]
            # multiply is needed after the broadcast.
            ex = (orientation * (p1[:, 0] - p0[:, 0]).view(num_envs, 1, 1))
            ey = (orientation * (p1[:, 1] - p0[:, 1]).view(num_envs, 1, 1))

            term_v = ex * (v - p0[:, 1].view(num_envs, 1, 1))   # [N, H, 1]
            term_u = ey * (u - p0[:, 0].view(num_envs, 1, 1))   # [N, 1, W]
            swept &= (term_v - term_u) >= 0.0                   # [N, H, W]

        # 6. An env whose previous line is not yet valid seeds prev and paints
        #    nothing this step, exactly as the per-env version did.
        valid = self.prev_head_valid
        swept &= valid.view(num_envs, 1, 1)

        self.paint_mask |= swept

        # 7. Coverage inside each env's own target rectangle.
        active_count = (rows * cols).float().clamp_min(1.0)
        new_ratio = self.paint_mask.sum(dim=(1, 2)).float() / active_count
        self.coverage_ratio[:, 0] = torch.where(valid, new_ratio, self.coverage_ratio[:, 0])

        # Current line becomes the previous line for the next sweep.
        self.prev_head_a_uv = curr_a
        self.prev_head_b_uv = curr_b
        self.prev_head_valid[:] = True

        # 8. Newly covered fraction during this timestep.
        self.coverage_delta[:, 0] = (self.coverage_ratio[:, 0] - old_coverage).clamp_min(0.0)

    
    def sample_init(self, env_ids):
        super().sample_init(env_ids)
         
        init_object_pos = self._motion_reset.body_pos_w[:, self.object_body_id_motion]
        init_object_quat = self._motion_reset.body_quat_w[:, self.object_body_id_motion]

        rand_samples = sample_uniform(self.object_pose_range[:, 0], self.object_pose_range[:, 1], (len(env_ids), 6), device=self.device)

        init_object_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        init_object_quat = quat_mul(init_object_quat, orientations_delta)
        
        init_object_state_w = self.object.data.default_root_state[env_ids]
        init_object_state_w[:, 0:3] = init_object_pos + self.env.scene.env_origins[env_ids]
        init_object_state_w[:, 3:7] = init_object_quat
        init_object_state_w[:, 7:]  = 0.0  # zero velocity

        self.object.write_root_link_pose_to_sim(init_object_state_w[:, 0:7], env_ids=env_ids)
        self.object.write_root_com_velocity_to_sim(init_object_state_w[:, 7:], env_ids=env_ids)

        for object_, object_body_id_motion in zip(self.extra_objects, self.extra_object_body_id_motion):
            init_object_pos = self._motion_reset.body_pos_w[:, object_body_id_motion]
            init_object_quat = self._motion_reset.body_quat_w[:, object_body_id_motion]
            
            init_object_pos += rand_samples[:, 0:3]
            init_object_quat = quat_mul(init_object_quat, orientations_delta)

            init_object_state_w = object_.data.default_root_state[env_ids]
            init_object_state_w[:, 0:3] = init_object_pos + self.env.scene.env_origins[env_ids]
            init_object_state_w[:, 3:7] = init_object_quat
            init_object_state_w[:, 7:]  = 0.0  # zero velocity

            object_.write_root_link_pose_to_sim(init_object_state_w[:, 0:7], env_ids=env_ids)
            object_.write_root_com_velocity_to_sim(init_object_state_w[:, 7:], env_ids=env_ids)

        # robot_pos_w = self.asset.data.root_link_pos_w[env_ids]
        # robot_quat_w = self.asset.data.root_link_quat_w[env_ids]
        # object_pos_b = quat_apply_inverse(robot_quat_w, (init_object_pos + self.env.scene.env_origins[env_ids]) - robot_pos_w)
        # from isaaclab.utils.math import quat_conjugate
        # object_quat_b = quat_mul(quat_conjugate(robot_quat_w), init_object_quat)
        # print(f"Object initial position in robot frame: {object_pos_b}, orientation: {object_quat_b}")

        if self.object_joint_idx_asset is not None:
            init_joint_pos = self._motion_reset.joint_pos[:, self.object_joint_idx_motion].unsqueeze(1)
            init_joint_vel = self._motion_reset.joint_vel[:, self.object_joint_idx_motion].unsqueeze(1)

            joint_pos_noise = sample_uniform(-1, 1, (init_joint_pos.shape[0], init_joint_pos.shape[1]), device=self.device) * self.object_init_joint_pos_noise
            joint_vel_noise = sample_uniform(-1, 1, (init_joint_vel.shape[0], init_joint_vel.shape[1]), device=self.device) * self.object_init_joint_vel_noise
            
            init_joint_pos += joint_pos_noise
            init_joint_vel += joint_vel_noise
            
            joint_pos_limits = self.object.data.soft_joint_pos_limits[env_ids]
            joint_vel_limits = self.object.data.soft_joint_vel_limits[env_ids]
            init_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
            init_joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

            self.object.write_joint_state_to_sim(init_joint_pos, init_joint_vel, env_ids=env_ids, joint_ids=[self.object_joint_idx_asset])
        
        # Sample/reset wall-painting target region.
        self._sample_target_region(env_ids)

    def _save_motion(self):
        motion_data: TensorDict = torch.cat(self.motion_frames, dim=0)
        motion_data = motion_data[25:].numpy()
        motion_data["object_contact"] = self._object_contact[25:].cpu().numpy()
        moton_meta = {
            "joint_names": self.asset.joint_names,
            "body_names": self.asset.body_names + [self.object_asset_name],
            "fps": int(1/self.env.step_dt),
        }
        save_dir = "record_motion"
        motion_data_path = f"{save_dir}/motion.npz"
        motion_meta_path = f"{save_dir}/meta.json"
        import os, json
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(motion_data_path, **motion_data)
        with open(motion_meta_path, "w") as f:
            json.dump(moton_meta, f, indent=4)
        print(f"Saved recorded motion to {motion_data_path} and {motion_meta_path}")
        # upstream left a bare breakpoint() here, which drops every recording run
        # into pdb after the save completes.

    def update(self):
        super().update()
        if hasattr(self, "motion_frames"):
            motion_frame = self.motion_frames[-1]
            # add object data to the motion frame
            object_pos_w = self.object.data.body_link_pos_w[:, self.object_body_id_asset].cpu()
            object_quat_w = self.object.data.body_link_quat_w[:, self.object_body_id_asset].cpu()
            object_lin_vel_w = self.object.data.body_com_lin_vel_w[:, self.object_body_id_asset].cpu()
            object_ang_vel_w = self.object.data.body_com_ang_vel_w[:, self.object_body_id_asset].cpu()
            motion_frame["body_pos_w"] = torch.cat([motion_frame["body_pos_w"], object_pos_w.unsqueeze(1)], dim=1)
            motion_frame["body_quat_w"] = torch.cat([motion_frame["body_quat_w"], object_quat_w.unsqueeze(1)], dim=1)
            motion_frame["body_lin_vel_w"] = torch.cat([motion_frame["body_lin_vel_w"], object_lin_vel_w.unsqueeze(1)], dim=1)
            motion_frame["body_ang_vel_w"] = torch.cat([motion_frame["body_ang_vel_w"], object_ang_vel_w.unsqueeze(1)], dim=1)

        self.ref_object_pos_future_w = self.future_ref_motion.body_pos_w[..., self.object_body_id_motion, :] + self.env.scene.env_origins[:, None, :]
        self.ref_object_quat_future_w = self.future_ref_motion.body_quat_w[..., self.object_body_id_motion, :]
        self.ref_object_pos_w = self.ref_object_pos_future_w[:, 0]
        self.ref_object_quat_w = self.ref_object_quat_future_w[:, 0]
        self.object_pos_w = self.object.data.root_link_pos_w
        self.object_quat_w = self.object.data.root_link_quat_w
        # Update wall-painting coverage from the actual roller motion.
        self._update_paint_coverage()

        if self.object_joint_idx_asset is not None:
            self.ref_object_joint_pos_future = self.future_ref_motion.joint_pos[..., self.object_joint_idx_motion]
            self.ref_object_joint_vel_future = self.future_ref_motion.joint_vel[..., self.object_joint_idx_motion]
            self.ref_object_joint_pos = self.ref_object_joint_pos_future[:, 0]
            self.ref_object_joint_vel = self.ref_object_joint_vel_future[:, 0]
            self.object_joint_pos = self.object.data.joint_pos[:, self.object_joint_idx_asset]
            self.object_joint_vel = self.object.data.joint_vel[:, self.object_joint_idx_asset]
            
        idx = (self.motion_starts + self.t).unsqueeze(1) + self.future_steps.unsqueeze(0)
        idx.clamp_max_(self.motion_ends.unsqueeze(1) - 1)
        self.ref_object_contact_future = self._object_contact[idx]
        self.ref_object_contact = self.ref_object_contact_future[:, 0]
        
        # contact target and eef pos
        object_pos_w = self.object.data.body_link_pos_w[:, self.object_body_id_asset]
        object_quat_w = self.object.data.body_link_quat_w[:, self.object_body_id_asset]
        self.contact_target_pos_w[:] = object_pos_w.unsqueeze(1) + quat_apply(object_quat_w.unsqueeze(1), self.contact_target_pos_offset)
        
        eef_pos_w = self.asset.data.body_link_pos_w[:, self.contact_eef_body_indices_asset]
        eef_quat_w = self.asset.data.body_link_quat_w[:, self.contact_eef_body_indices_asset]
        self.contact_eef_pos_w[:] = eef_pos_w + quat_apply(eef_quat_w, self.contact_eef_pos_offset)
        
        self.eef_contact_forces_w.zero_()
        for eef_idx, (eef_sensors, eef_sensor_indices) in enumerate(zip(self.eef_filtered_sensor, self.eef_filtered_sensor_indices)):
            for eef_sensor, eef_sensor_id in zip(eef_sensors, eef_sensor_indices):
                self.eef_contact_forces_w[:, eef_idx] += eef_sensor.data.force_matrix_w[:, eef_sensor_id, 0]

        self.eef_contact_forces_b[:] = quat_apply_inverse(object_quat_w.unsqueeze(1), self.eef_contact_forces_w)

    def _init_debug_draw(self):
        super()._init_debug_draw()

        if self.env.backend != "isaac":
            return
        
        from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
        import isaaclab.sim as sim_utils
        vis_markers_cfg = VisualizationMarkersCfg(
            prim_path=f"/World/EefContact",
            markers={
                "left": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.3),
                        metallic=1.0,
                    )
                ),
                "right": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.3, 1.0),
                        metallic=1.0,
                    )
                ),
            }
        )
        self.eef_contact_markers = VisualizationMarkers(vis_markers_cfg)
        if not getattr(self, "debug_vis", True):
            self.eef_contact_markers.set_visibility(False)
        self.eef_contact_markers_indices = [0, 1] * (self.num_envs * self.num_eefs)
        self.eef_contact_markers_pos_w = torch.zeros(self.num_envs, 2, self.num_eefs, 3)

    def debug_draw(self):
        super().debug_draw()

        if self.env.backend != "isaac" or not getattr(self, "debug_vis", True):
            return
        
        self.eef_contact_markers_pos_w[:, 0, :, :] = self.contact_eef_pos_w
        self.eef_contact_markers_pos_w[:, 1, :, :] = self.contact_target_pos_w
        out_of_range_mask = ~self.ref_object_contact[:, None, :, None].expand_as(self.eef_contact_markers_pos_w)
        self.eef_contact_markers_pos_w[out_of_range_mask] = -1000.0
        
        self.eef_contact_markers.visualize(
            translations=self.eef_contact_markers_pos_w.view(-1, 3),
            marker_indices=self.eef_contact_markers_indices,
        )

        # visualize contact forces
        self.env.debug_draw.vector(
            self.contact_eef_pos_w.reshape(-1, 3),
            self.eef_contact_forces_w.reshape(-1, 3) / 80,
            color=(1.0, 1.0, 1.0, 1.0),
            size=4.0,
        )

        # draw vector from robot root to contact target

        self.env.debug_draw.vector(
            self.contact_eef_pos_w.view(-1, 3),
            (self.contact_target_pos_w - self.contact_eef_pos_w).view(-1, 3),
            color=(0, 1, 0, 1),
            size=4.0,
        )

        self._debug_draw_paint()

    def _debug_draw_paint(self, num_envs_shown: int = 4, stride: int = 4):
        """Outline the paint target and dot the cells the roller has covered.

        Purely visual: no prim is spawned and the physics is untouched, so a run
        with this on is comparable to one without. Drawn only for the first
        `num_envs_shown` envs and every `stride`-th raster cell, because the
        raster is ~151x214 and one point per painted cell per env would be tens
        of thousands of draw calls a frame.
        """
        if self.target_region_path is None:
            return

        n = min(num_envs_shown, self.num_envs)

        # Target rectangle outline: corners are ordered BL, BR, TR, TL.
        corners = self.target_corners_w[:n]                      # [n, 4, 3]
        starts = corners.reshape(-1, 3)
        ends = corners.roll(-1, dims=1).reshape(-1, 3)
        self.env.debug_draw.vector(
            starts, ends - starts, color=(1.0, 0.85, 0.2, 1.0), size=3.0,
        )

        # Painted cells, subsampled, as points on the wall plane.
        mask = self.paint_mask[:n, ::stride, ::stride]            # [n, h, w]
        if not bool(mask.any()):
            return

        env_i, row_i, col_i = torch.nonzero(mask, as_tuple=True)

        u = ((col_i * stride).float() + 0.5) * self.paint_resolution \
            - 0.5 * self.target_width[env_i, 0]
        v = ((row_i * stride).float() + 0.5) * self.paint_resolution \
            - 0.5 * self.target_height[env_i, 0]

        pts = (
            self.target_center_w[env_i]
            + u.unsqueeze(-1) * self.wall_axis_u_w.unsqueeze(0)
            + v.unsqueeze(-1) * self.wall_axis_v_w.unsqueeze(0)
        )
        self.env.debug_draw.point(pts, color=(0.15, 0.55, 1.0, 1.0), size=6.0)
