# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for aligning an inverted bulb with world negative Z."""

import math
import os

from isaaclab.utils import configclass

from rl_isaaclab.tasks.inhand_rotate.sharpa_wave_env_cfg import EventCfg

from .sharpa_wave_bulb_env_cfg import SharpaWaveBulbEnvCfg


@configclass
class SharpaWaveBulbUpAlignEnvCfg(SharpaWaveBulbEnvCfg):
    """Turn the bulb's local +Z axis onto the world-frame negative Z axis."""

    # Match the task-specific grasp cache collected at physical scale 1.0.
    events: EventCfg = EventCfg()
    scale_range = [1.0, 1.0, 1]
    events.rand_params(scale_range)

    episode_length_s = 10.0
    observation_space = 201
    priv_info_dim = 8
    include_rup_in_priv_info = False
    include_rup_in_policy_obs = True
    object_target_up_axis_w = (0.0, 0.0, -1.0)

    # The parent reset logic uses half of this interval width around each
    # cache pose's initial object height: 0.08 m total gives +/-4 cm.
    reset_height_lower = 0.59906
    reset_height_upper = 0.67906

    grasp_cache_path = "cache/sharpa_bulb_up_align_grasp_uniform_18bins"

    # Avoid spending most rollouts on cache poses which already satisfy the
    # task. Filtering preserves an equal number of rows in every scale bucket.
    min_initial_alignment_error_deg = 5.0

    # Reset-angle curriculum. Training begins with the first 10-degree bin.
    # Gravity is trained first; angle bins start expanding only after gravity
    # reaches its target magnitude. Each stage must run for at least one full
    # 10-second episode before the shared height-failure condition is checked.
    up_angle_curriculum = True
    up_angle_curriculum_total_bins = 18
    up_angle_curriculum_initial_bins = 1
    up_angle_curriculum_bins_per_step = 1
    up_angle_curriculum_min_stage_steps = 200
    up_angle_curriculum_gravity_target = 10.0
    up_angle_curriculum_frontier_fraction = 0.5
    up_angle_curriculum_success_threshold = 0.7
    up_angle_curriculum_min_frontier_episodes = 1024

    # Unclipped potential difference: decreasing the alignment angle is
    # rewarded and increasing it is penalized.  Since it telescopes across a
    # trajectory, oscillating back and forth cannot create net progress reward.
    alignment_progress_reward_scale = 10.0
    object_up_alignment_reward_scale = 1.0
    # Paid once when the success hold first completes. The episode continues
    # afterward so the policy is not incentivized to avoid a successful reset.
    alignment_completion_bonus = 10.0
    alignment_success_tolerance = 5.0 / 180.0 * math.pi
    alignment_success_hold_steps = 10
    # Root-position distance is measured in XYZ from the sampled reset pose.
    alignment_position_tolerance = 0.02
    # The normalized dense alignment reward reaches zero at this error.
    alignment_reward_zero_angle = 25.0 / 180.0 * math.pi

    rotate_reward_scale = 0.0
    object_linvel_penalty_scale = 0.0
    # Penalize squared joint-position distance from the configured default
    # hand shape. The cache starts around -0.072 reward/step at this scale.
    pos_diff_penalty_scale = -0.01
    pos_diff_reference_reset_pose = False
    torque_penalty_scale = 0.0
    work_penalty_scale = 0.0
    # Position reward stays disabled by default; base height is handled below.
    object_pos_reward_scale = 0.0
    # Keep the bulb base high enough for the downstream in-hand rotation task.
    # This is a base/root-z hinge penalty, not a top/tip-height target.
    object_base_z_min = 0.61906
    object_base_z_low_penalty_scale = 20.0
    object_z_penalty_scale = 0.0
    object_tip_z_penalty_scale = 0.0

    debug_show_object_vectors = False

    # Show the reward's current alignment angle above every bulb in GUI play.
    # No viewport labels are created during headless training.
    debug_show_alignment_angle_text = True
    alignment_angle_text_height = 0.14
    alignment_angle_text_size = 20
    alignment_angle_text_box_width = 150
    alignment_angle_text_box_height = 32

    def __post_init__(self):
        # Close-up view of the hand and bulb in environment zero. Using the env
        # frame keeps it centered when play creates a grid of environments.
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (0.35, -0.35, 0.85)
        self.viewer.lookat = (-0.09559, -0.00517, 0.63906)
        self.viewer.resolution = (1280, 720)
        self.object_cfg.spawn.usd_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../assets/Bulb/e27_bulb_new/E27_bulb_ring_collision.usda",
        )
        self.object_cfg.spawn.mass_props.mass = 0.0336
        self.randomize_mass_lower = 0.0336 / 1.2
        self.randomize_mass_upper = 0.0336 * 1.2
