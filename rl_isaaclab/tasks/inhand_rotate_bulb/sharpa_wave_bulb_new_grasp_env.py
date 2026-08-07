# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import carb
from isaaclab.utils.math import saturate

from rl_isaaclab.tasks.inhand_rotate.sharpa_wave_grasp_env import (
    SharpaWaveInhandRotateGraspEnv,
)

from .sharpa_wave_bulb_new_grasp_env_cfg import SharpaWaveBulbNewGraspEnvCfg


class SharpaWaveInhandRotateBulbNewGraspEnv(SharpaWaveInhandRotateGraspEnv):
    """Revalidate stable old-bulb grasps against the new physical geometry."""

    cfg: SharpaWaveBulbNewGraspEnvCfg

    def __init__(
        self,
        cfg: SharpaWaveBulbNewGraspEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        seed_cache = np.load(self.cfg.seed_grasp_cache_file, allow_pickle=False)
        configured_seed_scale_count = self.cfg.seed_grasp_cache_scale_count
        num_seed_scales = (
            int(self.cfg.scale_range[2])
            if configured_seed_scale_count is None
            else int(configured_seed_scale_count)
        )
        if seed_cache.ndim != 2 or seed_cache.shape[1] != 29:
            raise ValueError(f"Expected seed grasp cache shape (N, 29), got {seed_cache.shape}.")
        if seed_cache.shape[0] % num_seed_scales != 0:
            raise ValueError(
                f"Seed cache rows ({seed_cache.shape[0]}) must be divisible by "
                f"{num_seed_scales}."
            )
        self.seed_bucket_size = seed_cache.shape[0] // num_seed_scales
        fixed_seed_scale_id = self.cfg.seed_grasp_cache_scale_id
        if fixed_seed_scale_id is not None:
            fixed_seed_scale_id = int(fixed_seed_scale_id)
            if not 0 <= fixed_seed_scale_id < num_seed_scales:
                raise ValueError(
                    f"seed_grasp_cache_scale_id={fixed_seed_scale_id} is outside "
                    f"[0, {num_seed_scales})."
                )
            start = fixed_seed_scale_id * self.seed_bucket_size
            seed_cache = seed_cache[start : start + self.seed_bucket_size]
            self._fixed_seed_scale_id = fixed_seed_scale_id
        else:
            if int(self.cfg.scale_range[2]) != num_seed_scales:
                raise ValueError(
                    "The active scale count must match the seed-cache scale count "
                    "unless seed_grasp_cache_scale_id is explicitly selected."
                )
            self._fixed_seed_scale_id = None
        self.seed_grasps = torch.as_tensor(
            seed_cache, dtype=torch.float32, device=self.device
        )
        self.gravity_all_directions = [carb.Float3(0.0, 0.0, -9.81)]
        self.seed_flexion_dof_ids = torch.as_tensor(
            [
                index
                for index, name in enumerate(self.hand.joint_names)
                if (
                    "_FE" in name
                    or "_PIP" in name
                    or "_DIP" in name
                    or name.endswith("_IP")
                    or name.endswith("_CMC")
                )
            ],
            dtype=torch.long,
            device=self.device,
        )
        print(
            f"[INFO] New-bulb grasp seeds: {self.cfg.seed_grasp_cache_file} | "
            f"rows_per_scale={self.seed_bucket_size} | "
            f"selected_scale={self._fixed_seed_scale_id} | "
            f"flexion_dofs={len(self.seed_flexion_dof_ids)} | "
            f"closure={self.cfg.seed_flexion_offset_range}",
            flush=True,
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

        # DirectRLEnv may reset once during the parent constructor, before the
        # seed cache is loaded. The next reset from the wrapper uses seeds.
        if not hasattr(self, "seed_grasps"):
            return

        if env_ids is None:
            resolved_env_ids = self.hand._ALL_INDICES
        else:
            resolved_env_ids = torch.as_tensor(
                env_ids, dtype=torch.long, device=self.device
            )
        if self._fixed_seed_scale_id is None:
            scale_ids = self.scale_ids[resolved_env_ids].squeeze(-1).long()
        else:
            scale_ids = torch.zeros(
                len(resolved_env_ids), dtype=torch.long, device=self.device
            )
        row_in_bucket = torch.randint(
            self.seed_bucket_size,
            (len(resolved_env_ids),),
            device=self.device,
        )
        seed_indices = scale_ids * self.seed_bucket_size + row_in_bucket
        seed_states = self.seed_grasps[seed_indices]

        dof_pos = seed_states[:, : self.num_hand_dofs].clone()
        closure_lower, closure_upper = self.cfg.seed_flexion_offset_range
        closure = torch.empty(
            (len(resolved_env_ids), 1), device=self.device
        ).uniform_(closure_lower, closure_upper)
        dof_pos[:, self.seed_flexion_dof_ids] += closure
        dof_pos = saturate(
            dof_pos,
            self.hand_dof_lower_limits[resolved_env_ids],
            self.hand_dof_upper_limits[resolved_env_ids],
        )
        dof_vel = torch.zeros_like(dof_pos)
        self.prev_targets[resolved_env_ids] = dof_pos
        self.cur_targets[resolved_env_ids] = dof_pos
        self.hand.set_joint_position_target(dof_pos, env_ids=resolved_env_ids)
        self.hand.write_joint_state_to_sim(
            dof_pos, dof_vel, env_ids=resolved_env_ids
        )

        object_state = self.object.data.default_root_state[resolved_env_ids].clone()
        object_state[:, :3] = (
            seed_states[:, 22:25] + self.scene.env_origins[resolved_env_ids]
        )
        object_state[:, 3:7] = seed_states[:, 25:29]
        object_state[:, 7:] = 0.0
        self.object.write_root_state_to_sim(
            object_state, env_ids=resolved_env_ids
        )
        self.rb_forces[resolved_env_ids] = 0.0

        self._refresh_lab()
        self.object_pos_prev[resolved_env_ids] = self.object_pos[resolved_env_ids]
        self.object_rot_prev[resolved_env_ids] = self.object_rot[resolved_env_ids]
        self.last_contacts[resolved_env_ids] = 0.0
        self.proprio_hist_buf[resolved_env_ids] = 0.0
        self.at_reset_buf[resolved_env_ids] = 1
