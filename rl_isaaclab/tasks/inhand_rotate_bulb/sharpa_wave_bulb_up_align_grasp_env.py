# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Full-orientation grasp-cache generator for the bulb up-alignment task."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils.math import quat_from_angle_axis, quat_mul

from .sharpa_wave_bulb_new_grasp_env import (
    SharpaWaveInhandRotateBulbNewGraspEnv,
)
from .sharpa_wave_bulb_up_align_grasp_env_cfg import (
    SharpaWaveBulbUpAlignGraspEnvCfg,
)


class SharpaWaveInhandBulbUpAlignGraspEnv(
    SharpaWaveInhandRotateBulbNewGraspEnv
):
    """Sample the bulb's full orientation while re-closing seeded grasps."""

    cfg: SharpaWaveBulbUpAlignGraspEnvCfg

    def __init__(
        self,
        cfg: SharpaWaveBulbUpAlignGraspEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        # Each scale block contains environments assigned round-robin to every
        # angle bin, ensuring all bins continue receiving reset attempts.
        self._reset_angle_bin_ids = (
            torch.arange(self.num_envs, device=self.device)
            % self._grasp_angle_bins
        )
        print(
            "[INFO] Bulb up-alignment grasp sampling: "
            f"{self._grasp_angle_bins} equal angle bins over [0, 180] deg | "
            "axial roll over [0, 360) deg | orientation tilt does not reset",
            flush=True,
        )

    def _sample_object_orientations(
        self, env_ids: torch.Tensor
    ) -> torch.Tensor:
        """Sample full rotations with equal probability in every angle bin."""
        count = len(env_ids)
        angle_bin_ids = self._reset_angle_bin_ids[env_ids]
        angle_bin_width = torch.pi / self._grasp_angle_bins
        target_angle = (
            angle_bin_ids.float() + torch.rand(count, device=self.device)
        ) * angle_bin_width
        azimuth = 2.0 * torch.pi * torch.rand(count, device=self.device)

        # Desired world direction of the bulb's local +Z. Its angle from the
        # task target (world -Z) is exactly target_angle.
        sin_angle = torch.sin(target_angle)
        desired_up = torch.stack(
            (
                sin_angle * torch.cos(azimuth),
                sin_angle * torch.sin(azimuth),
                -torch.cos(target_angle),
            ),
            dim=-1,
        )

        local_up = torch.zeros((count, 3), device=self.device)
        local_up[:, 2] = 1.0
        alignment_axis = torch.cross(local_up, desired_up, dim=-1)
        axis_norm = torch.linalg.vector_norm(
            alignment_axis, dim=-1, keepdim=True
        )
        fallback_axis = torch.zeros_like(alignment_axis)
        fallback_axis[:, 0] = 1.0
        alignment_axis = torch.where(
            axis_norm > 1.0e-6,
            alignment_axis / torch.clamp(axis_norm, min=1.0e-6),
            fallback_axis,
        )
        alignment_angle = torch.acos(torch.clamp(desired_up[:, 2], -1.0, 1.0))
        align_quat = quat_from_angle_axis(alignment_angle, alignment_axis)

        # Bulb heading is also randomized, rather than sampling only its axis.
        axial_roll = 2.0 * torch.pi * torch.rand(count, device=self.device)
        roll_quat = quat_from_angle_axis(axial_roll, local_up)
        return quat_mul(align_quat, roll_quat)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

        # DirectRLEnv resets during parent construction before these sampler
        # fields exist. The wrapper's subsequent reset uses full orientations.
        if not hasattr(self, "_reset_angle_bin_ids"):
            return

        resolved_env_ids = (
            self.hand._ALL_INDICES
            if env_ids is None
            else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        )
        object_state = self.object.data.root_state_w[resolved_env_ids].clone()
        object_state[:, 3:7] = self._sample_object_orientations(resolved_env_ids)

        position_jitter = float(self.cfg.grasp_position_jitter)
        if position_jitter > 0.0:
            object_state[:, :3] += (
                2.0
                * torch.rand(
                    (len(resolved_env_ids), 3), device=self.device
                )
                - 1.0
            ) * position_jitter
        object_state[:, 7:] = 0.0
        self.object.write_root_state_to_sim(object_state, env_ids=resolved_env_ids)
        self.rb_forces[resolved_env_ids] = 0.0

        self._refresh_lab()
        self.object_pos_prev[resolved_env_ids] = self.object_pos[resolved_env_ids]
        self.object_rot_prev[resolved_env_ids] = self.object_rot[resolved_env_ids]
