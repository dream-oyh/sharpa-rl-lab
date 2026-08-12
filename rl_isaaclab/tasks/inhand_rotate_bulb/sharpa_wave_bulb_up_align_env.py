# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""In-hand task for aligning an inverted bulb with world negative Z."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from rl_isaaclab.tasks.inhand_rotate.sharpa_wave_env import (
    SharpaWaveInhandRotateEnv,
    quat_rotate,
)
from rl_isaaclab.utils.reward_logging import (
    EPISODE_REWARD_INFO_KEY,
    EPISODE_REWARD_TERMS,
)

from .sharpa_wave_bulb_up_align_env_cfg import SharpaWaveBulbUpAlignEnvCfg


class SharpaWaveInhandBulbUpAlignEnv(SharpaWaveInhandRotateEnv):
    """Align the bulb local +Z direction with world ``(0, 0, -1)``."""

    cfg: SharpaWaveBulbUpAlignEnvCfg

    def __init__(
        self,
        cfg: SharpaWaveBulbUpAlignEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        # These attributes must exist before DirectRLEnv construction because
        # it may dispatch overridden observation methods during initialization.
        self._alignment_angle_viewport_window = None
        self._alignment_angle_scene_view = None
        self._alignment_angle_scene_view_registered = False
        self._alignment_angle_transforms = []
        self._alignment_angle_labels = []
        self._up_angle_cache_indices = None

        super().__init__(cfg, render_mode, **kwargs)
        if self.cfg.alignment_reward_zero_angle <= self.cfg.alignment_success_tolerance:
            raise ValueError(
                "alignment_reward_zero_angle must be greater than "
                "alignment_success_tolerance."
            )
        if self.cfg.alignment_position_tolerance <= 0.0:
            raise ValueError("alignment_position_tolerance must be positive.")
        self._alignment_success_cosine = math.cos(
            self.cfg.alignment_success_tolerance
        )
        self._alignment_zero_cosine = math.cos(
            self.cfg.alignment_reward_zero_angle
        )
        self._success_hold_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._alignment_just_completed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._alignment_completed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._filter_initial_alignment_cache()
        if self.cfg.up_angle_curriculum:
            self._setup_up_angle_curriculum()
        self._setup_alignment_angle_labels()

    def _setup_up_angle_curriculum(self) -> None:
        """Index the grasp cache by final Up-alignment angle."""
        total_bins = int(self.cfg.up_angle_curriculum_total_bins)
        initial_bins = int(self.cfg.up_angle_curriculum_initial_bins)
        angle_max = float(self.cfg.up_angle_curriculum_angle_max)
        if total_bins <= 0:
            raise ValueError("up_angle_curriculum_total_bins must be positive.")
        if angle_max <= 0.0:
            raise ValueError("up_angle_curriculum_angle_max must be positive.")
        if not 1 <= initial_bins <= total_bins:
            raise ValueError(
                "up_angle_curriculum_initial_bins must be in "
                f"[1, {total_bins}], got {initial_bins}."
            )
        if int(self.cfg.up_angle_curriculum_min_stage_steps) <= 0:
            raise ValueError("up_angle_curriculum_min_stage_steps must be positive.")
        frontier_fraction = float(self.cfg.up_angle_curriculum_frontier_fraction)
        if not 0.0 <= frontier_fraction <= 1.0:
            raise ValueError(
                "up_angle_curriculum_frontier_fraction must be in [0, 1]."
            )
        success_threshold = float(self.cfg.up_angle_curriculum_success_threshold)
        if not 0.0 <= success_threshold <= 1.0:
            raise ValueError(
                "up_angle_curriculum_success_threshold must be in [0, 1]."
            )
        if int(self.cfg.up_angle_curriculum_min_frontier_episodes) <= 0:
            raise ValueError(
                "up_angle_curriculum_min_frontier_episodes must be positive."
            )

        num_scales = int(self.cfg.scale_range[2])
        self._up_angle_cache_indices = []
        self._up_angle_cache_angles = []
        for scale_id in range(num_scales):
            start = scale_id * self.bucket_grasp
            stop = start + self.bucket_grasp
            cache_indices = torch.arange(start, stop, device=self.device)
            angles = self._alignment_angle(
                self.saved_grasping_states[start:stop, 25:29]
            )
            valid_angle = angles <= angle_max
            cache_indices = cache_indices[valid_angle]
            angles = angles[valid_angle]
            if len(cache_indices) == 0:
                raise ValueError(
                    f"Scale bucket {scale_id} has no cache poses with Up-angle "
                    f"<= {math.degrees(angle_max):g} deg."
                )
            angle_bin_ids = torch.floor(
                angles / angle_max * total_bins
            ).long().clamp_max(total_bins - 1)
            scale_bins = [
                cache_indices[angle_bin_ids == angle_bin_id]
                for angle_bin_id in range(total_bins)
            ]
            scale_angle_bins = [
                angles[angle_bin_ids == angle_bin_id]
                for angle_bin_id in range(total_bins)
            ]
            empty_bins = [
                angle_bin_id
                for angle_bin_id, indices in enumerate(scale_bins)
                if len(indices) == 0
            ]
            if empty_bins:
                raise ValueError(
                    f"Scale bucket {scale_id} has no cache poses in Up-angle "
                    f"bins {empty_bins}."
                )
            self._up_angle_cache_indices.append(scale_bins)
            self._up_angle_cache_angles.append(scale_angle_bins)
            print(
                "[INFO] Up-angle curriculum cache filter: "
                f"scale_bucket={scale_id} kept={len(cache_indices)}/"
                f"{stop - start} poses with angle <= "
                f"{math.degrees(angle_max):g} deg",
                flush=True,
            )

        self._up_angle_curriculum_bins = initial_bins
        self._up_angle_curriculum_gravity_ready = False
        self._up_angle_curriculum_last_advance_step = int(
            self.common_step_counter
        )
        self._up_angle_curriculum_frontier_episodes = 0
        self._up_angle_curriculum_frontier_successes = 0
        self._episode_up_angle_bin = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        print(
            "[INFO] Up-angle curriculum: gravity first | "
            f"active_bins={initial_bins}/{total_bins} | "
            f"initial_max_angle={math.degrees(angle_max) * initial_bins / total_bins:g} deg | "
            f"angle_max={math.degrees(angle_max):g} deg | "
            f"frontier_fraction={frontier_fraction:.2f} | "
            f"success_threshold={success_threshold:.2f} | "
            f"min_stage_steps={self.cfg.up_angle_curriculum_min_stage_steps}",
            flush=True,
        )
        fixed_angle_deg = getattr(self.cfg, "up_angle_curriculum_fixed_angle_deg", None)
        if fixed_angle_deg is not None:
            fixed_angle = max(0.0, min(math.radians(float(fixed_angle_deg)), angle_max))
            fixed_bin = min(
                int(math.floor(fixed_angle / angle_max * total_bins)),
                total_bins - 1,
            )
            print(
                "[INFO] Up-angle reset override: "
                f"requested={float(fixed_angle_deg):g} deg | "
                f"sampling_bin={fixed_bin}/{total_bins - 1}",
                flush=True,
            )

    def _sample_grasp_cache_rows(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Focus reset sampling on the frontier while rehearsing easier bins."""
        if self._up_angle_cache_indices is None:
            return super()._sample_grasp_cache_rows(env_ids)

        active_bins = int(self._up_angle_curriculum_bins)
        total_bins = int(self.cfg.up_angle_curriculum_total_bins)
        fixed_angle_deg = getattr(self.cfg, "up_angle_curriculum_fixed_angle_deg", None)
        if fixed_angle_deg is not None:
            angle_max = float(self.cfg.up_angle_curriculum_angle_max)
            fixed_angle = max(0.0, min(math.radians(float(fixed_angle_deg)), angle_max))
            fixed_bin = int(math.floor(fixed_angle / angle_max * total_bins))
            fixed_bin = min(fixed_bin, total_bins - 1)
            sampled_angle_bins = torch.full(
                (len(env_ids),), fixed_bin, dtype=torch.long, device=self.device
            )
            active_bins = total_bins
        elif active_bins >= total_bins:
            # Once the curriculum is complete, restore a uniform distribution
            # over the entire angle range.
            sampled_angle_bins = torch.randint(
                0, total_bins, size=(len(env_ids),), device=self.device
            )
        elif active_bins == 1:
            sampled_angle_bins = torch.zeros(
                len(env_ids), dtype=torch.long, device=self.device
            )
        else:
            # Keep half of resets focused on the newest/hardest bin and use
            # the other half to prevent forgetting on all earlier bins.
            frontier_mask = torch.rand(len(env_ids), device=self.device) < float(
                self.cfg.up_angle_curriculum_frontier_fraction
            )
            sampled_angle_bins = torch.randint(
                0, active_bins - 1, size=(len(env_ids),), device=self.device
            )
            sampled_angle_bins[frontier_mask] = active_bins - 1

        self._episode_up_angle_bin[env_ids] = sampled_angle_bins
        sampled_scale_ids = self.scale_ids[env_ids].squeeze(-1).to(torch.long)
        sampled_cache_indices = torch.empty(
            len(env_ids), dtype=torch.long, device=self.device
        )
        for scale_id in range(len(self._up_angle_cache_indices)):
            for angle_bin_id in range(active_bins):
                mask = (sampled_scale_ids == scale_id) & (
                    sampled_angle_bins == angle_bin_id
                )
                count = int(mask.sum().item())
                if count == 0:
                    continue
                candidates = self._up_angle_cache_indices[scale_id][angle_bin_id]
                if fixed_angle_deg is not None:
                    candidate_angles = self._up_angle_cache_angles[scale_id][
                        angle_bin_id
                    ]
                    half_bin_width = 0.5 * float(
                        self.cfg.up_angle_curriculum_angle_max
                    ) / total_bins
                    near_target = torch.abs(candidate_angles - fixed_angle) <= half_bin_width
                    if near_target.any():
                        candidates = candidates[near_target]
                choices = torch.randint(
                    0, len(candidates), size=(count,), device=self.device
                )
                sampled_cache_indices[mask] = candidates[choices]
        return self.saved_grasping_states[sampled_cache_indices].clone()

    def _record_up_angle_frontier_episodes(
        self, done: torch.Tensor
    ) -> None:
        """Accumulate completion statistics for the active frontier bin."""
        if (
            not self.cfg.up_angle_curriculum
            or not self._up_angle_curriculum_gravity_ready
            or self._up_angle_curriculum_bins
            >= self.cfg.up_angle_curriculum_total_bins
        ):
            return
        frontier_bin = int(self._up_angle_curriculum_bins) - 1
        frontier_done = done & (self._episode_up_angle_bin == frontier_bin)
        episode_count = int(frontier_done.sum().item())
        if episode_count == 0:
            return
        success_count = int(
            (frontier_done & self._alignment_completed).sum().item()
        )
        self._up_angle_curriculum_frontier_episodes += episode_count
        self._up_angle_curriculum_frontier_successes += success_count

    def _update_up_angle_curriculum(self) -> None:
        """Expand reset-angle bins after gravity training has completed."""
        total_bins = int(self.cfg.up_angle_curriculum_total_bins)
        angle_max = float(self.cfg.up_angle_curriculum_angle_max)
        active_bins = int(self._up_angle_curriculum_bins)
        gravity = self.physics_sim_view.get_gravity()
        gravity_magnitude = math.sqrt(
            float(gravity[0]) ** 2
            + float(gravity[1]) ** 2
            + float(gravity[2]) ** 2
        )
        gravity_ready = (
            gravity_magnitude + 1.0e-4
            >= float(self.cfg.up_angle_curriculum_gravity_target)
        )

        self.extras["up_angle_curriculum_bins"] = float(active_bins)
        self.extras["up_angle_curriculum_max_deg"] = (
            math.degrees(angle_max) * active_bins / total_bins
        )
        self.extras["up_angle_curriculum_gravity_ready"] = float(gravity_ready)
        frontier_episodes = self._up_angle_curriculum_frontier_episodes
        frontier_success_rate = (
            self._up_angle_curriculum_frontier_successes / frontier_episodes
            if frontier_episodes > 0
            else 0.0
        )
        self.extras["up_angle_curriculum_frontier_episodes"] = float(
            frontier_episodes
        )
        self.extras["up_angle_curriculum_frontier_success_rate"] = float(
            frontier_success_rate
        )

        if not self.cfg.up_angle_curriculum or active_bins >= total_bins:
            return
        if not gravity_ready:
            return
        if not self._up_angle_curriculum_gravity_ready:
            self._up_angle_curriculum_gravity_ready = True
            self._up_angle_curriculum_last_advance_step = int(
                self.common_step_counter
            )
            self._up_angle_curriculum_frontier_episodes = 0
            self._up_angle_curriculum_frontier_successes = 0
            print(
                "[CURRICULUM] Gravity target reached; starting Up-angle "
                f"curriculum at {active_bins}/{total_bins} bins.",
                flush=True,
            )
            return

        stage_steps = int(self.cfg.up_angle_curriculum_min_stage_steps)
        if (
            int(self.common_step_counter)
            - self._up_angle_curriculum_last_advance_step
            < stage_steps
        ):
            return

        # Use exactly the same per-step height-failure thresholds as the
        # gravity curriculum in the parent environment.
        height_stable = bool(
            (self.extras["height_reset_upper"] < 5.0e-4).item()
        ) and bool((self.extras["height_reset_lower"] < 5.0e-4).item())
        if not height_stable:
            return

        min_frontier_episodes = int(
            self.cfg.up_angle_curriculum_min_frontier_episodes
        )
        if frontier_episodes < min_frontier_episodes:
            return
        if frontier_success_rate < float(
            self.cfg.up_angle_curriculum_success_threshold
        ):
            return

        bin_increment = max(int(self.cfg.up_angle_curriculum_bins_per_step), 1)
        completed_frontier_bin = active_bins - 1
        completed_success_rate = frontier_success_rate
        completed_episode_count = frontier_episodes
        self._up_angle_curriculum_bins = min(
            active_bins + bin_increment, total_bins
        )
        self._up_angle_curriculum_last_advance_step = int(
            self.common_step_counter
        )
        max_angle = (
            math.degrees(float(self.cfg.up_angle_curriculum_angle_max))
            * self._up_angle_curriculum_bins
            / total_bins
        )
        self.extras["up_angle_curriculum_bins"] = float(
            self._up_angle_curriculum_bins
        )
        self.extras["up_angle_curriculum_max_deg"] = max_angle
        self._up_angle_curriculum_frontier_episodes = 0
        self._up_angle_curriculum_frontier_successes = 0
        self.extras["up_angle_curriculum_frontier_episodes"] = 0.0
        self.extras["up_angle_curriculum_frontier_success_rate"] = 0.0
        print(
            "[CURRICULUM] Expanded Up-angle reset range: "
            f"{self._up_angle_curriculum_bins}/{total_bins} bins "
            f"(angle < {max_angle:g} deg) | completed_frontier_bin="
            f"{completed_frontier_bin} | success_rate="
            f"{completed_success_rate:.3f} "
            f"({completed_episode_count} episodes).",
            flush=True,
        )

    def _setup_alignment_angle_labels(self) -> None:
        """Create one camera-facing angle label above each GUI environment."""
        if (
            not self.cfg.debug_show_alignment_angle_text
            or not self.sim.has_gui()
            or self._alignment_angle_scene_view is not None
        ):
            return

        try:
            import carb
            import omni.ui as ui
            import omni.ui.scene as sc
            from omni.kit.viewport.utility import get_active_viewport_window

            viewport_window = get_active_viewport_window()
            if viewport_window is None:
                carb.log_warn(
                    "Bulb alignment angle labels were requested, but no active "
                    "viewport was found."
                )
                return

            self._alignment_angle_viewport_window = viewport_window
            frame_name = f"SharpaBulbAlignmentAngles-{id(self):x}"
            with viewport_window.get_frame(frame_name):
                scene_view = sc.SceneView()
                self._alignment_angle_scene_view = scene_view
                with scene_view.scene:
                    for _ in range(self.num_envs):
                        label_transform = sc.Transform(
                            look_at=sc.Transform.LookAt.CAMERA,
                            transform=sc.Matrix44.get_translation_matrix(
                                0.0, 0.0, 0.0
                            ),
                        )
                        with label_transform:
                            with sc.Transform(scale_to=sc.Space.SCREEN):
                                # Scene labels render in front of subsequently
                                # declared backgrounds at the same depth.
                                angle_label = sc.Label(
                                    "angle: --.- deg",
                                    size=self.cfg.alignment_angle_text_size,
                                    color=[0.05, 0.05, 0.05, 1.0],
                                    alignment=ui.Alignment.CENTER,
                                )
                                sc.Rectangle(
                                    width=self.cfg.alignment_angle_text_box_width,
                                    height=self.cfg.alignment_angle_text_box_height,
                                    color=[1.0, 0.88, 0.18, 0.88],
                                    wireframe=False,
                                )
                        self._alignment_angle_transforms.append(label_transform)
                        self._alignment_angle_labels.append(angle_label)

            viewport_window.viewport_api.add_scene_view(scene_view)
            self._alignment_angle_scene_view_registered = True
            self._update_alignment_angle_labels()
        except Exception as error:
            # A missing/late viewport should not make play or headless training
            # fail. Keep the warning visible so GUI setup problems are obvious.
            try:
                import carb

                carb.log_warn(f"Failed to create bulb alignment angle labels: {error}")
            finally:
                self._destroy_alignment_angle_labels()

    def _update_alignment_angle_labels(self) -> None:
        """Move labels with the bulbs and refresh their displayed angle."""
        if self._alignment_angle_scene_view is None:
            return

        import omni.ui.scene as sc

        label_positions = self.object.data.root_pos_w.clone()
        label_positions[:, 2] += self.cfg.alignment_angle_text_height
        angles_deg = torch.rad2deg(self._alignment_angle(self.object_rot))

        positions = label_positions.detach().cpu().tolist()
        angles = angles_deg.detach().cpu().tolist()
        for label_transform, angle_label, position, angle in zip(
            self._alignment_angle_transforms,
            self._alignment_angle_labels,
            positions,
            angles,
        ):
            label_transform.transform = sc.Matrix44.get_translation_matrix(*position)
            angle_label.text = f"angle: {angle:5.1f} deg"

    def _destroy_alignment_angle_labels(self) -> None:
        """Detach the overlay from the viewport before the simulation closes."""
        scene_view = self._alignment_angle_scene_view
        viewport_window = self._alignment_angle_viewport_window
        if scene_view is not None:
            scene_view.scene.clear()
            if (
                viewport_window is not None
                and self._alignment_angle_scene_view_registered
            ):
                viewport_window.viewport_api.remove_scene_view(scene_view)
            scene_view.destroy()

        self._alignment_angle_scene_view = None
        self._alignment_angle_viewport_window = None
        self._alignment_angle_scene_view_registered = False
        self._alignment_angle_transforms.clear()
        self._alignment_angle_labels.clear()

    def close(self) -> None:
        self._destroy_alignment_angle_labels()
        super().close()

    def _get_observations(self) -> dict:
        observations = super()._get_observations()
        # Updating once per policy step avoids a device-to-host copy at every
        # physics substep while remaining visually smooth during play.
        self._update_alignment_angle_labels()
        return observations

    def _target_up(self, count: int) -> torch.Tensor:
        target = torch.tensor(
            self.cfg.object_target_up_axis_w,
            dtype=torch.float32,
            device=self.device,
        )
        target = target / torch.clamp(torch.linalg.vector_norm(target), min=1.0e-6)
        return target.expand(count, -1)

    def _alignment_angle(self, object_rot: torch.Tensor) -> torch.Tensor:
        local_up = torch.tensor(
            self.cfg.object_up_axis, dtype=torch.float32, device=self.device
        ).expand(len(object_rot), -1)
        local_up = local_up / torch.clamp(
            torch.linalg.vector_norm(local_up, dim=-1, keepdim=True), min=1.0e-6
        )
        object_up = quat_rotate(object_rot, local_up)
        cosine = torch.clamp(
            (object_up * self._target_up(len(object_rot))).sum(dim=-1), -1.0, 1.0
        )
        return torch.acos(cosine)

    def _filter_initial_alignment_cache(self) -> None:
        """Drop near-solved resets while retaining equal scale-bucket sizes."""
        if self.saved_grasping_states is None:
            raise RuntimeError("The bulb up-alignment task requires a grasp cache.")

        threshold = self.cfg.min_initial_alignment_error_deg / 180.0 * torch.pi
        num_scale_buckets = int(self.cfg.scale_range[2])
        filtered_buckets = []
        for scale_id in range(num_scale_buckets):
            start = scale_id * self.bucket_grasp
            bucket = self.saved_grasping_states[start : start + self.bucket_grasp]
            keep = self._alignment_angle(bucket[:, 25:29]) >= threshold
            filtered = bucket[keep]
            if len(filtered) == 0:
                raise ValueError(
                    "No cache poses remain for scale bucket "
                    f"{scale_id} at min_initial_alignment_error_deg="
                    f"{self.cfg.min_initial_alignment_error_deg:g}."
                )
            filtered_buckets.append(filtered)

        # The parent reset sampler assumes a single fixed bucket width.
        self.bucket_grasp = min(len(bucket) for bucket in filtered_buckets)
        self.saved_grasping_states = torch.cat(
            [bucket[: self.bucket_grasp] for bucket in filtered_buckets], dim=0
        )
        print(
            "[INFO] Bulb up-alignment cache: "
            f"{self.bucket_grasp} rows/scale with initial error >= "
            f"{self.cfg.min_initial_alignment_error_deg:g} deg",
            flush=True,
        )

    def _get_rewards(self) -> torch.Tensor:
        """Reward alignment while regularizing toward the default hand shape."""
        self.extras.pop(EPISODE_REWARD_INFO_KEY, None)

        alignment_angle = self._alignment_angle(self.object_rot)
        previous_alignment_angle = self._alignment_angle(self.object_rot_prev)
        alignment_progress = previous_alignment_angle - alignment_angle
        hand_pose_reference = (
            self.reset_hand_dof_pos
            if self.cfg.pos_diff_reference_reset_pose
            else self.hand.data.default_joint_pos
        )[:, self.actuated_dof_indices]
        hand_pose_error = (
            (
                self.hand_dof_pos[:, self.actuated_dof_indices]
                - hand_pose_reference
            )
            ** 2
        ).sum(dim=-1)
        object_pos_error = torch.linalg.vector_norm(
            self.object_pos - self.object_default_pose[:, :3], dim=-1
        )
        object_base_z = self.object_pos[:, 2]
        object_base_z_low_error = torch.clamp(
            float(self.cfg.object_base_z_min) - object_base_z,
            min=0.0,
        )
        object_z_diff = torch.abs(
            object_base_z - self.object_default_pose[:, 2]
        )

        object_tip_local_pos = torch.tensor(
            self.cfg.object_tip_local_pos, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        object_tip_pos = self.object_pos + quat_rotate(
            self.object_rot, object_tip_local_pos
        )
        default_object_tip_pos = self.object_default_pose[:, :3] + quat_rotate(
            self.object_default_pose[:, 3:7], object_tip_local_pos
        )
        object_tip_z_diff = torch.abs(
            object_tip_pos[:, 2] - default_object_tip_pos[:, 2]
        )

        up_alignment = torch.cos(alignment_angle)
        alignment_score = torch.clamp(
            (up_alignment - self._alignment_zero_cosine)
            / (self._alignment_success_cosine - self._alignment_zero_cosine),
            min=0.0,
            max=1.0,
        )
        position_score = torch.clamp(
            1.0
            - object_pos_error / float(self.cfg.alignment_position_tolerance),
            min=0.0,
            max=1.0,
        )
        gated_position_score = alignment_score * position_score
        completion_bonus = (
            self._alignment_just_completed.float()
            * self.cfg.alignment_completion_bonus
        )
        zero_reward = torch.zeros_like(alignment_angle)

        reward_terms = {
            "rotate_reward": (
                alignment_progress * self.cfg.alignment_progress_reward_scale
            ),
            "object_linvel_penalty": zero_reward.clone(),
            "pos_diff_penalty": (
                hand_pose_error * self.cfg.pos_diff_penalty_scale
            ),
            "torque_penalty": zero_reward.clone(),
            "work_penalty": zero_reward.clone(),
            "object_pos_diff": zero_reward.clone(),
            "position_reward": (
                gated_position_score * self.cfg.object_pos_reward_scale
            ),
            "object_z_penalty": (
                -object_base_z_low_error.square()
                * self.cfg.object_base_z_low_penalty_scale
            ),
            "object_tip_z_penalty": zero_reward.clone(),
            "object_up_alignment_reward": (
                alignment_score * self.cfg.object_up_alignment_reward_scale
            ),
            "success_reward": completion_bonus,
        }
        total_reward = torch.stack(tuple(reward_terms.values()), dim=0).sum(dim=0)
        reward_terms["total_reward"] = total_reward

        self._step_reward_terms = reward_terms
        for term in EPISODE_REWARD_TERMS:
            self._episode_reward_sums[term] += reward_terms[term]

        self.extras["alignment_error_deg"] = torch.rad2deg(alignment_angle).mean()
        self.extras["up_alignment"] = up_alignment.mean()
        self.extras["object_pos_error"] = object_pos_error.mean()
        self.extras["object_base_z"] = object_base_z.mean()
        self.extras["object_base_z_low_error"] = object_base_z_low_error.mean()
        self.extras["object_z_diff"] = object_z_diff.mean()
        self.extras["object_tip_z_diff"] = object_tip_z_diff.mean()
        self.extras["gravity_x"] = self.physics_sim_view.get_gravity()[0]
        self.extras["gravity_y"] = self.physics_sim_view.get_gravity()[1]
        self.extras["gravity_z"] = self.physics_sim_view.get_gravity()[2]
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        failed, time_out = super()._get_dones()
        alignment_angle = self._alignment_angle(self.object_rot)
        within_tolerance = alignment_angle <= self.cfg.alignment_success_tolerance
        self._success_hold_count = torch.where(
            within_tolerance,
            self._success_hold_count + 1,
            torch.zeros_like(self._success_hold_count),
        )
        aligned = self._success_hold_count >= self.cfg.alignment_success_hold_steps
        self._alignment_just_completed = (
            aligned & ~self._alignment_completed & ~failed
        )
        self._alignment_completed |= self._alignment_just_completed
        if self.cfg.up_angle_curriculum:
            self._record_up_angle_frontier_episodes(failed | time_out)
            self._update_up_angle_curriculum()
        return failed, time_out

    def _settle_reset_states(self, env_ids: torch.Tensor) -> None:
        settle_steps = max(int(self.cfg.reset_settle_physics_steps), 0)
        if settle_steps == 0 or len(env_ids) == 0:
            return

        for _ in range(settle_steps):
            self._refresh_lab()
            if self.cfg.torque_control:
                torques = (
                    self.p_gain[env_ids]
                    * (self.cur_targets[env_ids] - self.hand_dof_pos[env_ids])
                    - self.d_gain[env_ids] * self.hand_dof_vel[env_ids]
                )
                self.hand.set_joint_effort_target(
                    torques[:, self.actuated_dof_indices],
                    joint_ids=self.actuated_dof_indices,
                    env_ids=env_ids,
                )
            else:
                self.hand.set_joint_position_target(
                    self.cur_targets[env_ids][:, self.actuated_dof_indices],
                    joint_ids=self.actuated_dof_indices,
                    env_ids=env_ids,
                )
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        zero_root_velocity = torch.zeros((len(env_ids), 6), device=self.device)
        self.object.write_root_velocity_to_sim(zero_root_velocity, env_ids=env_ids)
        self.hand.write_root_velocity_to_sim(zero_root_velocity, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(
            self.hand.data.joint_pos[env_ids].clone(),
            torch.zeros_like(self.hand.data.joint_vel[env_ids]),
            env_ids=env_ids,
        )
        self.hand.set_joint_position_target(self.cur_targets[env_ids], env_ids=env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)
        self._refresh_lab()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        resolved_env_ids = (
            self.hand._ALL_INDICES
            if env_ids is None
            else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        )
        self._settle_reset_states(resolved_env_ids)

        # Anchor translation rewards at the sampled cache pose while retaining
        # the fixed world negative-Z orientation target.
        self.object_default_pose[resolved_env_ids, :3] = self.object_pos[
            resolved_env_ids
        ]
        self.object_pos_prev[resolved_env_ids] = self.object_pos[resolved_env_ids]
        self.object_rot_prev[resolved_env_ids] = self.object_rot[resolved_env_ids]
        if hasattr(self, "_success_hold_count"):
            self._success_hold_count[resolved_env_ids] = 0
        if hasattr(self, "_alignment_just_completed"):
            self._alignment_just_completed[resolved_env_ids] = False
        if hasattr(self, "_alignment_completed"):
            self._alignment_completed[resolved_env_ids] = False
