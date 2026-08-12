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
    priv_info_dim = 11
    include_rup_in_priv_info = True
    enable_contact_pos = False

    bulb_asset_scale = (1.0, 1.0, 1.0)
    bulb_model = "original"
    original_bulb_grasp_cache_path = "cache/sharpa_bulb_grasp_high_1p0_angle25"
    new_bulb_grasp_cache_path = "cache/sharpa_bulb_new_grasp_high_1p0_angle25"

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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.065),
            scale=bulb_asset_scale,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.09559, -0.00517, 0.63906),
            rot=(0.0, 1.0, 0.0, 0.0),
        ),
    )

    events: EventCfg = EventCfg()

    scale_range = [0.7, 1.1, 5]
    events.rand_params(scale_range)

    grasp_cache_path = original_bulb_grasp_cache_path
    object_base_friction = 0.5
    randomize_mass_lower = 0.065 / 1.2
    randomize_mass_upper = 0.065 * 1.2
    randomize_com_lower = -0.003
    randomize_com_upper = 0.003
    rot_axis = (0, 0, 1)
    object_z_penalty_scale = 0.0
    object_tip_local_pos = (0, 0, 0.05)
    object_tip_z_penalty_scale = -1.0
    object_up_axis = (0, 0, 1)
    object_heading_axis = (1, 0, 0)
    object_up_alignment_reward_scale = 0.25
    reset_height_lower = 0.59906
    reset_height_upper = 0.67906
    reset_angle_diff = 25 / 180 * math.pi
    debug_show_object_pos = True

    def set_bulb_model(self, bulb_model: str) -> None:
        """Switch between the original and physical new bulb assets."""
        normalized_model = bulb_model.lower()
        if normalized_model in ("original", "old"):
            normalized_model = "original"
            usd_path = "../../../assets/Bulb/E27_Bulb_rigid.usda"
            mass = 0.065
            grasp_cache_path = self.original_bulb_grasp_cache_path
        elif normalized_model == "new":
            usd_path = "../../../assets/Bulb/e27_bulb_new/E27_bulb_rigid.usda"
            mass = 0.0336
            grasp_cache_path = self.new_bulb_grasp_cache_path
        else:
            raise ValueError(
                "bulb_model must be one of {'original', 'old', 'new'}, "
                f"got {bulb_model!r}."
            )

        self.bulb_model = normalized_model
        self.object_cfg.spawn.usd_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            usd_path,
        )
        self.object_cfg.spawn.mass_props.mass = mass
        self.randomize_mass_lower = mass / 1.2
        self.randomize_mass_upper = mass * 1.2
        self.grasp_cache_path = grasp_cache_path
