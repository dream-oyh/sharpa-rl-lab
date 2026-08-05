# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from .sharpa_wave_bulb_socket_env_cfg import SharpaWaveBulbSocketEnvCfg


@configclass
class SharpaWaveBulbRingEnvCfg(SharpaWaveBulbSocketEnvCfg):
    """Bulb task guided by a physical socket ring with no joint constraint."""

    # Restore the original in-hand task's gravity curriculum. Training starts
    # with weak gravity and the base environment ramps it toward 10 m/s^2 once
    # the bulb remains inside the task's height bounds reliably.
    gravity_curriculum = True

    socket_anchor_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/socket_anchor",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../../assets/Bulb/E27_socket_guide.usda",
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.00075,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.09559, -0.00517, 0.63906),
            rot=(0.0, 1.0, 0.0, 0.0),
        ),
    )

    # Match the original in-hand bulb reward configuration exactly. The ring
    # changes only the contact geometry and reset placement.
    angvel_clip_min = -0.5
    angvel_clip_max = 0.5
    rotate_reward_scale = 2.5
    object_linvel_penalty_scale = -0.3
    pos_diff_penalty_scale = -0.4
    torque_penalty_scale = -0.1
    work_penalty_scale = -0.5
    object_pos_reward_scale = 0.003
    object_z_penalty_scale = 0.0
    object_tip_z_penalty_scale = -1.0
    object_up_alignment_reward_scale = 0.25

    def __post_init__(self):
        super().__post_init__()
        self.sim.gravity = (0.0, 0.0, -0.05)
        self.object_cfg.spawn.usd_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../assets/Bulb/E27_Bulb_ring_collision.usda",
        )
        # The original 2 mm contact envelope is large compared with the
        # guide's 0.75--1.0 mm radial clearance.
        self.object_cfg.spawn.collision_props.contact_offset = 0.00075
        self.object_cfg.spawn.collision_props.rest_offset = 0.0
