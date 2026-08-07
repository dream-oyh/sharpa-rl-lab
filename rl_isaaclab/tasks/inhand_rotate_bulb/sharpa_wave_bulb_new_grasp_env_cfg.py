# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.utils import configclass

from .sharpa_wave_bulb_grasp_env_cfg import SharpaWaveBulbGraspEnvCfg


@configclass
class SharpaWaveBulbNewGraspEnvCfg(SharpaWaveBulbGraspEnvCfg):
    """Grasp-cache generator for the physical 60 x 109 mm E27 bulb."""

    # The cache is used by the guided Ring task, which experiences downward
    # gravity rather than the generic generator's six-direction stress test.
    episode_length_s = 4.0
    seed_grasp_cache_file = (
        "cache/sharpa_bulb_grasp_high_1p0_angle25_0.7-1.1-5.npy"
    )
    # The sequential grasp launcher sets these fields so a single-scale worker
    # reads only the matching scale bucket from a multi-scale seed cache.
    seed_grasp_cache_scale_count = None
    seed_grasp_cache_scale_id = None
    seed_flexion_offset_range = (0.0, 0.08)
    save_grasp_cache_path = "cache/sharpa_bulb_new_grasp_high_1p0_angle25"
    randomize_mass = False

    def __post_init__(self):
        self.object_cfg.spawn.usd_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../assets/Bulb/e27_bulb_new/E27_bulb_ring_collision.usda",
        )
        self.object_cfg.spawn.mass_props.mass = 0.0336
