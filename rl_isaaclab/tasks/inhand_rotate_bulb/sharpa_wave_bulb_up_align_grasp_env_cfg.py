# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Grasp-cache configuration for full-orientation bulb up alignment."""

from isaaclab.utils import configclass

from rl_isaaclab.tasks.inhand_rotate.sharpa_wave_grasp_env_cfg import EventCfg

from .sharpa_wave_bulb_new_grasp_env_cfg import SharpaWaveBulbNewGraspEnvCfg


@configclass
class SharpaWaveBulbUpAlignGraspEnvCfg(SharpaWaveBulbNewGraspEnvCfg):
    """Collect stable bulb grasps evenly across the complete angle range."""

    # Validate this collector at the physical-size scale first.  Define a new
    # event instance so changing this task cannot mutate another task's config.
    events: EventCfg = EventCfg()
    scale_range = [1.0, 1.0, 1]
    events.rand_params(scale_range)

    # Do not reject a grasp simply because the bulb has tilted. Contact loss
    # and root-height failures still reset the environment.
    grasp_terminate_on_orientation_deviation = False

    # One scale x eighteen 10-degree bins x 278 stable poses per bucket.  5004
    # is the nearest total at or above 5000 that divides evenly across 18 bins.
    grasp_angle_bins = 18
    grasp_cache_size = 5004
    grasp_angle_target_axis_w = (0.0, 0.0, -1.0)
    seed_grasp_cache_file = (
        "cache/sharpa_bulb_new_grasp_high_1p0_angle25_0.7-1.1-5.npy"
    )
    # Scale 1.0 is bucket 3 in the five-scale [0.7, ..., 1.1] seed cache.
    seed_grasp_cache_scale_count = 5
    seed_grasp_cache_scale_id = 3
    save_grasp_cache_path = "cache/sharpa_bulb_up_align_grasp_uniform_18bins"

    # Small translation diversity helps arbitrary orientations settle into
    # nearby valid grasps instead of testing only one object-root position.
    grasp_position_jitter = 0.003
