# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import time
import os

import numpy as np
import torch
from collections.abc import Sequence

import carb
from isaaclab.utils.math import quat_conjugate, quat_mul, saturate

from .sharpa_wave_grasp_env_cfg import SharpaWaveEnvCfg
from .sharpa_wave_env import SharpaWaveInhandRotateEnv, quat_rotate


class SharpaWaveInhandRotateGraspEnv(SharpaWaveInhandRotateEnv):
    def __init__(self, cfg: SharpaWaveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._grasp_angle_bins = int(self.cfg.grasp_angle_bins)
        if self._grasp_angle_bins <= 0:
            raise ValueError(
                f"grasp_angle_bins must be positive, got {self._grasp_angle_bins}."
            )
        self._grasp_angle_max = float(getattr(self.cfg, "grasp_angle_max", torch.pi))
        if self._grasp_angle_max <= 0.0:
            raise ValueError(
                f"grasp_angle_max must be positive, got {self._grasp_angle_max}."
            )
        num_cache_buckets = int(self.cfg.scale_range[2]) * self._grasp_angle_bins
        self.saved_grasping_states = [
            torch.zeros((0, 29), dtype=torch.float32, device=self.device)
            for _ in range(num_cache_buckets)
        ]
        self._grasp_progress_start_time = time.monotonic()
        self._grasp_last_progress_time = -float("inf")
        self.gravity_id = 0
        self.gravity_all_directions = [
            carb.Float3(0.0, 0.0, 9.81),
            carb.Float3(0.0, 0.0, -9.81),
            carb.Float3(0.0, 9.81, 0.0),
            carb.Float3(0.0, -9.81, 0.0),
            carb.Float3(9.81, 0.0, 0.0),
            carb.Float3(-9.81, 0.0, 0.0),
        ]

    def _get_rewards(self) -> torch.Tensor:
        cond1 = (torch.norm(self.fingertip_pos - self.object_pos.unsqueeze(1), dim=-1, p=2) < 0.1).all(-1)
        filtered_force_matrix = torch.cat([self._contact_sensor[id].data.force_matrix_w[:, 0, 0, :].unsqueeze(1) for id in range(10)], dim=1)
        cond2 = (torch.norm(filtered_force_matrix, dim=-1, p=2) > 0.5).sum(-1) >= 3
        cond = cond1 & cond2
        if self.cfg.grasp_terminate_on_orientation_deviation:
            orientation_deviation = quat_to_rot(
                quat_mul(
                    self.object_rot,
                    quat_conjugate(
                        self.object.data.default_root_state.clone()[:, 3:7]
                    ),
                )
            )
            cond &= orientation_deviation < self.cfg.reset_angle_diff
        self.reset_buf[~cond] = 1
        if self.common_step_counter % 40 == 0:
            self.physics_sim_view.set_gravity(self.gravity_all_directions[self.gravity_id])
            self.gravity_id += 1
            self.gravity_id %= len(self.gravity_all_directions)
        return 0

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES

        self._refresh_lab()
        success = self.episode_length_buf == self.max_episode_length - 1
        all_states = torch.cat(
            [self.hand_dof_pos, self.object_pos, self.object_rot], dim=1
        )[success]
        saved_scale_ids = self.scale_ids[success].squeeze(-1).long()
        if self._grasp_angle_bins > 1 and len(all_states) > 0:
            object_up_axis = torch.tensor(
                self.cfg.object_up_axis, device=self.device, dtype=torch.float32
            ).expand(len(all_states), -1)
            object_up_axis = object_up_axis / torch.clamp(
                torch.linalg.vector_norm(object_up_axis, dim=-1, keepdim=True),
                min=1.0e-6,
            )
            object_up_w = quat_rotate(all_states[:, 25:29], object_up_axis)
            target_axis_w = torch.tensor(
                self.cfg.grasp_angle_target_axis_w,
                device=self.device,
                dtype=torch.float32,
            )
            target_axis_w = target_axis_w / torch.clamp(
                torch.linalg.vector_norm(target_axis_w), min=1.0e-6
            )
            grasp_angles = torch.acos(
                torch.clamp((object_up_w * target_axis_w).sum(dim=-1), -1.0, 1.0)
            )
            angle_valid = grasp_angles <= self._grasp_angle_max + 1.0e-6
            all_states = all_states[angle_valid]
            saved_scale_ids = saved_scale_ids[angle_valid]
            grasp_angles = grasp_angles[angle_valid]
            saved_angle_ids = torch.floor(
                grasp_angles / self._grasp_angle_max * self._grasp_angle_bins
            ).long().clamp_max(self._grasp_angle_bins - 1)
        else:
            saved_angle_ids = torch.zeros_like(saved_scale_ids)
        saved_bucket_ids = (
            saved_scale_ids * self._grasp_angle_bins + saved_angle_ids
        )
        max_cache_size = int(getattr(self.cfg, "grasp_cache_size", 50000))
        num_cache_buckets = int(self.cfg.scale_range[2]) * self._grasp_angle_bins
        max_cache_per_bucket = max_cache_size // num_cache_buckets
        if max_cache_per_bucket <= 0:
            raise ValueError(
                f"grasp_cache_size={max_cache_size} is smaller than the "
                f"{num_cache_buckets} scale/angle buckets."
            )
        for bucket_id in torch.unique(saved_bucket_ids).detach().cpu().tolist():
            saved_grasping_states = self.saved_grasping_states[bucket_id]
            remaining = max_cache_per_bucket - len(saved_grasping_states)
            if remaining > 0:
                candidates = all_states[saved_bucket_ids == bucket_id]
                if len(candidates) > 0:
                    self.saved_grasping_states[bucket_id] = torch.cat(
                        [saved_grasping_states, candidates[:remaining]], dim=0
                    )
        bucket_sizes = [len(bucket) for bucket in self.saved_grasping_states]
        finished_buckets = sum(
            size >= max_cache_per_bucket for size in bucket_sizes
        )
        self._report_grasp_progress(
            bucket_sizes=bucket_sizes,
            max_cache_per_bucket=max_cache_per_bucket,
            finished_buckets=finished_buckets,
            force=finished_buckets == num_cache_buckets,
        )
        if finished_buckets == num_cache_buckets:
            print('done!')
            save_data = torch.cat(self.saved_grasping_states, dim=0)
            os.makedirs('cache', exist_ok=True)
            cache_prefix = getattr(self.cfg, "save_grasp_cache_path", None) or self.cfg.grasp_cache_path or 'cache/sharpa_grasp_linspace'
            name = f'{cache_prefix}_{self.cfg.scale_range[0]}-{self.cfg.scale_range[1]}-{self.cfg.scale_range[2]}.npy'
            np.save(name, save_data.cpu().numpy())
            exit()

        self.scene.reset(env_ids)

        # apply events such as randomization for environments that need a reset
        if self.cfg.events:
            if "reset" in self.event_manager.available_modes:
                env_step_count = self._sim_step_counter // self.cfg.decimation
                self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # reset noise models
        if self.cfg.action_noise_model:
            self._action_noise_model.reset(env_ids)
        if self.cfg.observation_noise_model:
            self._observation_noise_model.reset(env_ids)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

        rand_floats = 2.0 * torch.rand((len(env_ids), self.num_hand_dofs), device=self.device) - 1.0
        
        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, :3] += self.scene.env_origins[env_ids]
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        self.rb_forces[env_ids, :] = 0.0

        self.reset_height_lower[env_ids] = self.cfg.reset_height_lower
        self.reset_height_upper[env_ids] = self.cfg.reset_height_upper

        # reset hand
        dof_pos = self.hand.data.default_joint_pos[env_ids] + 0.15 * rand_floats
        dof_pos = saturate(dof_pos, self.hand_dof_lower_limits[env_ids], self.hand_dof_upper_limits[env_ids],)
        dof_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self._refresh_lab()

        self.object_pos_prev[env_ids] = self.object_pos[env_ids]
        self.object_rot_prev[env_ids] = self.object_rot[env_ids]

        # reset data buffers
        self.last_contacts[env_ids] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def _report_grasp_progress(
        self,
        bucket_sizes: list[int],
        max_cache_per_bucket: int,
        finished_buckets: int,
        force: bool = False,
    ) -> None:
        """Print a throttled progress bar with collection rate and ETA."""
        now = time.monotonic()
        report_interval = float(self.cfg.grasp_progress_interval_s)
        if not force and now - self._grasp_last_progress_time < report_interval:
            return
        self._grasp_last_progress_time = now

        num_cache_buckets = len(bucket_sizes)
        target_total = max_cache_per_bucket * num_cache_buckets
        saved_total = sum(bucket_sizes)
        progress = min(saved_total / target_total, 1.0)
        bar_width = 24
        filled_width = min(int(progress * bar_width), bar_width)
        progress_bar = "#" * filled_width + "-" * (bar_width - filled_width)

        elapsed = max(now - self._grasp_progress_start_time, 1.0e-6)
        collection_rate = saved_total / elapsed
        if collection_rate > 0.0:
            eta_seconds = (target_total - saved_total) / collection_rate
            eta_text = self._format_duration(eta_seconds)
        else:
            eta_text = "--"

        num_scales = int(self.cfg.scale_range[2])
        per_scale_target = max_cache_per_bucket * self._grasp_angle_bins
        per_scale_sizes = [
            sum(
                bucket_sizes[
                    scale_id
                    * self._grasp_angle_bins : (scale_id + 1)
                    * self._grasp_angle_bins
                ]
            )
            for scale_id in range(num_scales)
        ]
        scale_text = ",".join(str(size) for size in per_scale_sizes)
        min_bucket = min(bucket_sizes)
        max_bucket = max(bucket_sizes)

        print(
            f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] grasp cache '
            f"[{progress_bar}] {saved_total}/{target_total} "
            f"({100.0 * progress:5.1f}%) | "
            f"buckets {finished_buckets}/{num_cache_buckets} | "
            f"scales [{scale_text}]/{per_scale_target} | "
            f"bucket min/max {min_bucket}/{max_bucket} | "
            f"rate {collection_rate:.1f}/s | ETA {eta_text}",
            flush=True,
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(int(seconds), 0)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:d}h{minutes:02d}m"
        if minutes > 0:
            return f"{minutes:d}m{seconds:02d}s"
        return f"{seconds:d}s"


@torch.jit.script
def quat_to_rot(quaternion: torch.Tensor):
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    angle = 2 * torch.acos(quaternion[:, 0])
    return angle
