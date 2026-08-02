# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import omni.usd
import torch
from isaaclab.assets import RigidObject
from isaaclab.utils.math import quat_conjugate, quat_mul
from pxr import Gf, Sdf, UsdPhysics

from rl_isaaclab.tasks.inhand_rotate.sharpa_wave_env import (
    SharpaWaveInhandRotateEnv,
    quat_rotate,
)

from .sharpa_wave_bulb_socket_env_cfg import SharpaWaveBulbSocketEnvCfg


class SharpaWaveInhandRotateBulbSocketEnv(SharpaWaveInhandRotateEnv):
    """Bulb rotation task constrained by a passive revolute socket joint.

    The original grasp cache is sampled by the parent environment.  At every
    reset, the sampled hand--bulb pair is rigidly transformed onto the fixed
    socket pose.  The socket axis and the fixed wrist base remain at their
    authored poses; only the cached finger joints and configured physical
    parameters vary.
    """

    cfg: SharpaWaveBulbSocketEnvCfg

    def __init__(self, cfg: SharpaWaveBulbSocketEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        cache = np.load(self.cfg.socket_grasp_cache_file, allow_pickle=False)
        cache_scales = np.asarray(self.cfg.socket_grasp_cache_scales, dtype=np.float32)
        if cache.ndim != 2 or cache.shape[1] != 29:
            raise ValueError(f"Expected grasp cache shape (N, 29), got {cache.shape}.")
        if cache.shape[0] % len(cache_scales) != 0:
            raise ValueError(
                f"Cache rows ({cache.shape[0]}) must be divisible by scale buckets ({len(cache_scales)})."
            )
        expected_scales = np.linspace(
            self.cfg.scale_range[0], self.cfg.scale_range[1], self.cfg.scale_range[2], dtype=np.float32
        )
        if not np.allclose(cache_scales, expected_scales, atol=1.0e-6):
            raise ValueError(
                f"Cache scales {cache_scales.tolist()} do not match task scales {expected_scales.tolist()}."
            )

        source_bucket_size = cache.shape[0] // len(cache_scales)
        nominal_object_quat = np.asarray(
            self.cfg.socket_anchor_cfg.init_state.rot, dtype=np.float32
        )
        nominal_object_quat /= np.linalg.norm(nominal_object_quat)
        selected_buckets = []
        selected_counts = []
        for scale_id in range(len(cache_scales)):
            bucket = cache[
                scale_id * source_bucket_size : (scale_id + 1) * source_bucket_size
            ]
            object_quat = bucket[:, 25:29]
            object_quat = object_quat / np.linalg.norm(object_quat, axis=-1, keepdims=True)
            quat_dot = np.clip(
                np.abs(object_quat @ nominal_object_quat), 0.0, 1.0
            )
            angle_deg = np.rad2deg(2.0 * np.arccos(quat_dot))
            if self.cfg.socket_wrist_pose_dr:
                selected = bucket[
                    angle_deg <= float(self.cfg.socket_wrist_pose_dr_max_angle_deg) + 1.0e-5
                ]
            else:
                selected = bucket[np.argmin(angle_deg) : np.argmin(angle_deg) + 1]
            if len(selected) == 0:
                raise ValueError(
                    f"No scale-{cache_scales[scale_id]:.1f} cache grasps satisfy the configured "
                    f"wrist-pose DR limit ({self.cfg.socket_wrist_pose_dr_max_angle_deg} deg)."
                )
            selected_buckets.append(selected)
            selected_counts.append(len(selected))

        # The parent reset sampler assumes equally sized, contiguous scale buckets.
        bucket_size = min(selected_counts)
        cache = np.concatenate([bucket[:bucket_size] for bucket in selected_buckets], axis=0)
        self.saved_grasping_states = torch.as_tensor(
            cache, dtype=torch.float32, device=self.device
        )
        self.bucket_grasp = bucket_size

        self.socket_reference_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.socket_axis_w = torch.zeros((self.num_envs, 3), device=self.device)
        print(
            f"[INFO] Bulb socket cache: {self.cfg.socket_grasp_cache_file} | "
            f"scales={cache_scales.tolist()} | rows_per_scale={bucket_size} | "
            f"wrist_pose_dr={self.cfg.socket_wrist_pose_dr} | "
            f"max_angle_deg={self.cfg.socket_wrist_pose_dr_max_angle_deg} | "
            f"global_so3_dr={self.cfg.reset_random_quat}"
        )

    def _setup_scene(self):
        # Spawn the source anchor and joint before the parent clones env_0.  The
        # relationship targets are internal to env_0 and are remapped by cloning.
        self.socket_anchor = RigidObject(self.cfg.socket_anchor_cfg)

        stage = omni.usd.get_context().get_stage()
        env_path = "/World/envs/env_0"
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{env_path}/bulb_socket_revolute_joint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{env_path}/socket_anchor")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{env_path}/object")])
        joint.CreateAxisAttr(UsdPhysics.Tokens.z)
        pivot = float(self.cfg.socket_joint_offset_z)
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, pivot))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, pivot))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        joint.CreateCollisionEnabledAttr(False)

        damping = float(self.cfg.socket_joint_damping)
        if damping > 0.0:
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(damping)
            drive.CreateTargetVelocityAttr(0.0)
            drive.CreateMaxForceAttr(float("inf"))

        super()._setup_scene()
        self.scene.rigid_objects["socket_anchor"] = self.socket_anchor

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            resolved_env_ids = self.hand._ALL_INDICES
        else:
            resolved_env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        # The parent restores the selected cache row.  Map that bulb pose onto
        # the fixed socket and apply the same transform to the authored fixed
        # wrist root.  With both wrist DR flags disabled, one near-nominal cache
        # row is used per size bucket.
        super()._reset_idx(resolved_env_ids)

        sampled_bulb_pose_w = self.object.data.root_link_pose_w[resolved_env_ids].clone()
        socket_pose_w = self._get_socket_base_pose_w(resolved_env_ids)

        socket_state_w = self.socket_anchor.data.default_root_state[resolved_env_ids].clone()
        socket_state_w[:, :7] = socket_pose_w
        socket_state_w[:, 7:] = 0.0
        self.socket_anchor.write_root_state_to_sim(socket_state_w, env_ids=resolved_env_ids)

        delta_quat = quat_mul(
            socket_pose_w[:, 3:7], quat_conjugate(sampled_bulb_pose_w[:, 3:7])
        )

        hand_pose_w = self.hand.data.root_link_pose_w[resolved_env_ids].clone()
        hand_pose_w[:, :3] = socket_pose_w[:, :3] + quat_rotate(
            delta_quat, hand_pose_w[:, :3] - sampled_bulb_pose_w[:, :3]
        )
        hand_pose_w[:, 3:7] = quat_mul(delta_quat, hand_pose_w[:, 3:7])
        self.hand.write_root_pose_to_sim(hand_pose_w, env_ids=resolved_env_ids)

        bulb_state_w = self.object.data.default_root_state[resolved_env_ids].clone()
        bulb_state_w[:, :7] = socket_pose_w
        bulb_state_w[:, 7:] = 0.0
        self.object.write_root_state_to_sim(bulb_state_w, env_ids=resolved_env_ids)

        socket_pose = socket_pose_w.clone()
        socket_pose[:, :3] -= self.scene.env_origins[resolved_env_ids]
        self.object_default_pose[resolved_env_ids] = socket_pose
        height_half_range = 0.5 * (self.cfg.reset_height_upper - self.cfg.reset_height_lower)
        self.reset_height_lower[resolved_env_ids] = socket_pose[:, 2] - height_half_range
        self.reset_height_upper[resolved_env_ids] = socket_pose[:, 2] + height_half_range

        self._refresh_lab()
        self.object_pos_prev[resolved_env_ids] = self.object_pos[resolved_env_ids]
        self.object_rot_prev[resolved_env_ids] = self.object_rot[resolved_env_ids]

        if hasattr(self, "socket_reference_pose"):
            self.socket_reference_pose[resolved_env_ids] = socket_pose_w

            local_z = torch.zeros((len(resolved_env_ids), 3), device=self.device)
            local_z[:, 2] = 1.0
            joint_axis_w = quat_rotate(socket_pose_w[:, 3:7], local_z)

            # A revolute axis is geometrically unsigned.  Pick the sign closest
            # to the original task reward axis.
            reward_axis_w = torch.tensor(
                self.cfg.rot_axis, dtype=torch.float32, device=self.device
            ).expand_as(joint_axis_w)
            flip = torch.where(
                torch.sum(joint_axis_w * reward_axis_w, dim=-1, keepdim=True) < 0.0,
                -1.0,
                1.0,
            )
            self.socket_axis_w[resolved_env_ids] = joint_axis_w * flip
            self.rot_axis[resolved_env_ids] = self.socket_axis_w[resolved_env_ids]

    def _get_socket_base_pose_w(self, env_ids: torch.Tensor) -> torch.Tensor:
        socket_pose_w = self.socket_anchor.data.default_root_state[env_ids, :7].clone()
        socket_pose_w[:, :3] += self.scene.env_origins[env_ids]
        return socket_pose_w
