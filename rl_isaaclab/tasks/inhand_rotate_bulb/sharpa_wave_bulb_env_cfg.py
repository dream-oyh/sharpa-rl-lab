# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from rl_isaaclab.tasks.inhand_rotate.sharpa_wave_env_cfg import EventCfg
from rl_isaaclab.tasks.inhand_rotate.sharpa_wave_env_cfg import SharpaWaveEnvCfg


@configclass
class SharpaWaveBulbEnvCfg(SharpaWaveEnvCfg):
    bulb_asset_scale = (1.0, 1.0, 1.0)

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../../assets/Bulb/E27_Bulb_rigid.usda",
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
            scale=bulb_asset_scale,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.09559, -0.00517, 0.63906),
            rot=(0.0, 1.0, 0.0, 0.0),
        ),
    )

    events: EventCfg = EventCfg()

    scale_range = [1.0, 1.0, 1]
    events.rand_params(scale_range)

    grasp_cache_path = "cache/sharpa_bulb_grasp_high_1p0"
    object_base_friction = 0.4
    randomize_mass_lower = 0.02
    randomize_mass_upper = 0.08
    randomize_com_lower = -0.003
    randomize_com_upper = 0.003
    rot_axis = (0, 0, 1)
    object_axis_align_axis = (0, 0, 1)
    object_axis_align_penalty_scale = 0.0
    reset_height_lower = 0.61906
    reset_height_upper = 0.65906
    reset_angle_diff = 20 / 180 * math.pi
    debug_show_object_pos = True
