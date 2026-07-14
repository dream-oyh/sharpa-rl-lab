# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize sampled SharpaWave-and-bulb poses from a grasp cache.

The cache layout is ``[22 hand joint positions, 3 object positions,
4 object quaternion values]``.  Each selected row is shown in one Isaac Lab
environment.  The green arrow is the bulb's current up axis, the blue arrow is
its reference up axis, and the red arrow is its heading axis.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visualize sampled bulb grasp-cache poses.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of cache samples/environments to show.")
parser.add_argument(
    "--cache",
    type=str,
    default="cache/sharpa_bulb_grasp_high_1p0",
    help="Bulb cache prefix or a complete .npy cache path.",
)
parser.add_argument("--seed", type=int, default=42, help="Seed used to sample cache rows without replacement.")
parser.add_argument("--env_spacing", type=float, default=0.75, help="Spacing between displayed environments in meters.")
parser.add_argument("--vector_length", type=float, default=0.15, help="Length of the up/heading arrows in meters.")
parser.add_argument("--vector_thickness", type=float, default=0.008, help="Thickness of the arrows in meters.")
parser.add_argument(
    "--max_frames",
    type=int,
    default=None,
    help="Optional number of frames to render; by default the viewer runs until it is closed.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Prevent unrelated arguments from being consumed by libraries imported below.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below this point runs after Isaac Sim has started."""

import gymnasium as gym
import numpy as np
import torch

import rl_isaaclab.tasks.inhand_rotate_bulb  # noqa: F401
from rl_isaaclab.tasks.inhand_rotate_bulb.sharpa_wave_bulb_env_cfg import SharpaWaveBulbEnvCfg


TASK_NAME = "Isaac-Inhand-Rotate-Sharpa-Wave-Bulb-v0"
CACHE_STATE_DIM = 29
HAND_DOF_DIM = 22


def _validate_args() -> None:
    if args_cli.num_envs <= 0:
        raise ValueError(f"--num_envs must be positive, got {args_cli.num_envs}.")
    if args_cli.env_spacing <= 0.0:
        raise ValueError(f"--env_spacing must be positive, got {args_cli.env_spacing}.")
    if args_cli.vector_length <= 0.0:
        raise ValueError(f"--vector_length must be positive, got {args_cli.vector_length}.")
    if args_cli.vector_thickness <= 0.0:
        raise ValueError(f"--vector_thickness must be positive, got {args_cli.vector_thickness}.")
    if args_cli.max_frames is not None and args_cli.max_frames <= 0:
        raise ValueError(f"--max_frames must be positive, got {args_cli.max_frames}.")


def _resolve_cache(cfg: SharpaWaveBulbEnvCfg) -> tuple[str, str]:
    """Return ``(cache_prefix, cache_file)`` accepted by both the env and NumPy."""
    scale_lower, scale_upper, scale_count = cfg.scale_range
    suffix = f"_{scale_lower}-{scale_upper}-{scale_count}.npy"
    requested_path = os.path.abspath(os.path.expanduser(args_cli.cache))

    if requested_path.endswith(".npy"):
        if not requested_path.endswith(suffix):
            raise ValueError(
                f"Cache file must end with '{suffix}' for scale_range={cfg.scale_range}; "
                f"got '{requested_path}'."
            )
        cache_file = requested_path
        cache_prefix = requested_path[: -len(suffix)]
    else:
        cache_prefix = requested_path
        cache_file = f"{cache_prefix}{suffix}"

    if not os.path.isfile(cache_file):
        raise FileNotFoundError(f"Grasp cache does not exist: {cache_file}")
    return cache_prefix, cache_file


def _sample_cache(cache_file: str) -> tuple[np.ndarray, np.ndarray]:
    cache = np.load(cache_file, allow_pickle=False)
    if cache.ndim != 2 or cache.shape[1] != CACHE_STATE_DIM:
        raise ValueError(
            f"Expected cache shape (N, {CACHE_STATE_DIM}), got {cache.shape} in '{cache_file}'."
        )
    if args_cli.num_envs > cache.shape[0]:
        raise ValueError(
            f"Requested {args_cli.num_envs} environments, but the cache contains only {cache.shape[0]} rows."
        )
    if not np.isfinite(cache).all():
        raise ValueError(f"Cache contains NaN or infinite values: {cache_file}")

    quaternion_norms = np.linalg.norm(cache[:, 25:29], axis=1)
    if np.any(quaternion_norms < 1.0e-6):
        bad_row = int(np.flatnonzero(quaternion_norms < 1.0e-6)[0])
        raise ValueError(f"Cache row {bad_row} contains an invalid zero-length object quaternion.")

    rng = np.random.default_rng(args_cli.seed)
    sample_indices = rng.choice(cache.shape[0], size=args_cli.num_envs, replace=False)
    return np.asarray(cache[sample_indices], dtype=np.float32), sample_indices


def _set_camera_light() -> None:
    """Use the active viewport camera light when a GUI is available."""
    if args_cli.headless:
        return

    import carb
    import omni.kit.actions.core

    action_registry = omni.kit.actions.core.get_action_registry()
    action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera")
    if action is not None:
        action.execute()
    else:
        carb.settings.get_settings().set_bool("/rtx/useViewLightingMode", True)


def _write_cache_poses(raw_env, sampled_states: torch.Tensor) -> None:
    """Write the selected cache rows to their corresponding environments."""
    env_ids = torch.arange(raw_env.num_envs, dtype=torch.long, device=raw_env.device)
    joint_pos = sampled_states[:, :HAND_DOF_DIM]
    joint_vel = torch.zeros_like(joint_pos)

    hand_root_state = raw_env.hand.data.default_root_state.clone()
    hand_root_state[:, :3] += raw_env.scene.env_origins
    hand_root_state[:, 7:] = 0.0
    raw_env.hand.write_root_state_to_sim(hand_root_state, env_ids=env_ids)
    raw_env.hand.set_joint_position_target(joint_pos, env_ids=env_ids)
    raw_env.hand.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    raw_env.prev_targets[:] = joint_pos
    raw_env.cur_targets[:] = joint_pos

    object_root_state = raw_env.object.data.default_root_state.clone()
    object_root_state[:, :3] = sampled_states[:, 22:25] + raw_env.scene.env_origins
    object_root_state[:, 3:7] = sampled_states[:, 25:29]
    object_root_state[:, 7:] = 0.0
    raw_env.object.write_root_pose_to_sim(object_root_state[:, :7], env_ids=env_ids)
    raw_env.object.write_root_velocity_to_sim(object_root_state[:, 7:], env_ids=env_ids)


def main() -> None:
    _validate_args()

    cfg = SharpaWaveBulbEnvCfg()
    cache_prefix, cache_file = _resolve_cache(cfg)
    sampled_cache, sample_indices = _sample_cache(cache_file)

    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.env_spacing = args_cli.env_spacing
    cfg.sim.device = args_cli.device if args_cli.device is not None else cfg.sim.device
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.seed = args_cli.seed
    cfg.grasp_cache_path = cache_prefix
    cfg.reset_random_quat = False
    cfg.randomize_pd_gains = False
    cfg.randomize_friction = False
    cfg.randomize_com = False
    cfg.randomize_mass = False
    cfg.randomize_joint_pos_offset = False
    cfg.gravity_curriculum = False
    cfg.force_scale = 0.0
    cfg.enable_tactile = False
    cfg.enable_contact_pos = False
    cfg.debug_show_axes = False
    cfg.debug_show_object_pos = False
    cfg.debug_show_object_vectors = True
    cfg.vis_object_vector_length = args_cli.vector_length
    cfg.vis_object_vector_thickness = args_cli.vector_thickness

    # A kinematic bulb plus explicit pose writes keeps every cache sample frozen
    # while the viewport continues rendering.
    cfg.object_cfg.spawn.rigid_props.kinematic_enabled = True
    cfg.object_cfg.spawn.rigid_props.disable_gravity = True

    env = None
    try:
        env = gym.make(TASK_NAME, cfg=cfg, render_mode=None)
        raw_env = env.unwrapped
        env.reset(seed=args_cli.seed)

        sampled_states = torch.from_numpy(sampled_cache).to(device=raw_env.device)
        _write_cache_poses(raw_env, sampled_states)
        raw_env.sim.forward()
        raw_env.scene.update(cfg.sim.dt)
        raw_env._refresh_lab()
        _set_camera_light()

        print(f"[INFO] Loaded cache: {cache_file}")
        print(f"[INFO] Displaying {args_cli.num_envs} sampled rows: {sample_indices.tolist()}")
        print("[INFO] Vector legend: up = green, reference up = blue, heading = red")
        print("[INFO] Close the Isaac Sim window or press Ctrl+C to exit.")

        frame = 0
        while simulation_app.is_running() and (args_cli.max_frames is None or frame < args_cli.max_frames):
            _write_cache_poses(raw_env, sampled_states)
            raw_env.sim.step(render=True)
            raw_env.scene.update(cfg.sim.dt)
            raw_env._refresh_lab()
            frame += 1
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
