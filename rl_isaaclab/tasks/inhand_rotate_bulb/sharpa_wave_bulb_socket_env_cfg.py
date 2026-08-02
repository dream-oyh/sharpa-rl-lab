# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from rl_isaaclab.tasks.inhand_rotate.sharpa_wave_env_cfg import EventCfg
from rl_isaaclab.utils.modified_events import randomize_rigid_body_scale

from .sharpa_wave_bulb_env_cfg import SharpaWaveBulbEnvCfg


@configclass
class SharpaWaveBulbSocketEnvCfg(SharpaWaveBulbEnvCfg):
    """Bulb task with a fixed invisible socket anchor and fixed wrist pose."""

    socket_anchor_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/socket_anchor",
        spawn=sim_utils.CuboidCfg(
            size=(0.001, 0.001, 0.001),
            visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.09559, -0.00517, 0.63906),
            rot=(0.0, 1.0, 0.0, 0.0),
        ),
    )
    socket_joint_offset_z = -0.046
    socket_joint_damping = 0.0

    # Read all five buckets from the original cache without conversion.
    socket_grasp_cache_file = "cache/sharpa_bulb_grasp_high_1p0_angle25_0.7-1.1-5.npy"
    socket_grasp_cache_scales = (0.7, 0.8, 0.9, 1.0, 1.1)
    # If enabled, wrist-pose DR comes from stable cache grasps with different
    # object poses.  Keep it disabled for the fixed-wrist baseline.
    socket_wrist_pose_dr = False
    socket_wrist_pose_dr_max_angle_deg = 25.0
    grasp_cache_path = None
    scale_range = [0.7, 1.1, 5]
    events: EventCfg = EventCfg()
    events.rand_params(scale_range)
    events.randomize_socket_scale = EventTermCfg(
        func=randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "scale_range": scale_range,
            "asset_cfg": SceneEntityCfg("socket_anchor"),
        },
    )

    # Minimal-demo dynamics: only the cached grasp and socket constraint vary.
    randomize_pd_gains = False
    randomize_friction = True
    randomize_com = False
    randomize_mass = False
    randomize_joint_pos_offset = False
    gravity_curriculum = False
    force_scale = 0.0
    joint_noise_scale = 0.0
    # Keep the authored wrist, bulb, socket, and rotation axis poses fixed.
    reset_random_quat = False

    # These terms become constant once the socket fixes translation and tilt.
    object_linvel_penalty_scale = 0.0
    object_pos_reward_scale = 0.0
    object_z_penalty_scale = 0.0
    object_tip_z_penalty_scale = 0.0
    object_up_alignment_reward_scale = 0.0

    def __post_init__(self):
        self.sim.gravity = (0.0, 0.0, -9.81)
