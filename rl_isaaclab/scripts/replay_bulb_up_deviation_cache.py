# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a policy from bulb cache poses with a large up-axis deviation."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replay large-up-deviation bulb cache poses.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of replay environments.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Inhand-Rotate-Sharpa-Wave-Bulb-v0",
    help="Isaac Lab bulb task.",
)
parser.add_argument(
    "--cache",
    type=str,
    default="cache/sharpa_bulb_grasp_high_1p0",
    help="Source bulb cache prefix or complete .npy path.",
)
parser.add_argument(
    "--filtered_cache",
    type=str,
    default=None,
    help="Filtered cache prefix or complete .npy path. An automatic name is used when omitted.",
)
parser.add_argument(
    "--min_up_deviation_deg",
    type=float,
    default=15.0,
    help="Minimum angle between the cached and reference up vectors.",
)
parser.add_argument("--load_path", type=str, required=True, help="PPO or ProprioAdapt checkpoint to replay.")
parser.add_argument(
    "--algorithm",
    type=str,
    choices=("PPO", "ProprioAdapt"),
    default=None,
    help="Policy type. Defaults to the task agent configuration.",
)
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument(
    "--show_object_vectors",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Show current up (green), reference up (blue), and heading (red).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Optional replay step limit; by default replay runs until the viewer is closed.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Keep Hydra from creating an ``outputs/`` directory for this replay-only tool.
sys.argv = [sys.argv[0], "hydra.run.dir=.", "hydra.output_subdir=null"] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below this point runs after Isaac Sim has started."""

from datetime import datetime

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

from rl_isaaclab.algo.padapt.padapt import ProprioAdapt
from rl_isaaclab.algo.ppo.ppo import PPO
from rl_isaaclab.wrapper.config_wrapper import ConfigWrapper
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper

import rl_isaaclab.tasks.inhand_rotate  # noqa: F401
import rl_isaaclab.tasks.inhand_rotate_bulb  # noqa: F401


CACHE_STATE_DIM = 29
OBJECT_QUATERNION_SLICE = slice(25, 29)


def _validate_args() -> None:
    if args_cli.num_envs <= 0:
        raise ValueError(f"--num_envs must be positive, got {args_cli.num_envs}.")
    if not 0.0 <= args_cli.min_up_deviation_deg <= 180.0:
        raise ValueError(
            f"--min_up_deviation_deg must be within [0, 180], got {args_cli.min_up_deviation_deg}."
        )
    if args_cli.max_steps is not None and args_cli.max_steps <= 0:
        raise ValueError(f"--max_steps must be positive, got {args_cli.max_steps}.")


def _cache_suffix(env_cfg: DirectRLEnvCfg) -> str:
    scale_lower, scale_upper, scale_count = env_cfg.scale_range
    if scale_count != 1:
        raise ValueError(
            "This bulb replay script expects a single object scale, "
            f"but scale_range is {env_cfg.scale_range}."
        )
    return f"_{scale_lower}-{scale_upper}-{scale_count}.npy"


def _resolve_source_cache(env_cfg: DirectRLEnvCfg) -> tuple[str, str]:
    suffix = _cache_suffix(env_cfg)
    requested = os.path.abspath(os.path.expanduser(args_cli.cache))
    if requested.endswith(".npy"):
        if not requested.endswith(suffix):
            raise ValueError(f"Source cache must end with '{suffix}', got '{requested}'.")
        source_file = requested
        source_prefix = requested[: -len(suffix)]
    else:
        source_prefix = requested
        source_file = f"{source_prefix}{suffix}"

    if not os.path.isfile(source_file):
        raise FileNotFoundError(f"Source cache does not exist: {source_file}")
    return source_prefix, source_file


def _angle_tag(angle_deg: float) -> str:
    return f"{angle_deg:g}".replace(".", "p")


def _resolve_filtered_cache(env_cfg: DirectRLEnvCfg, source_prefix: str) -> tuple[str, str]:
    suffix = _cache_suffix(env_cfg)
    if args_cli.filtered_cache is None:
        filtered_prefix = f"{source_prefix}_up_dev_ge_{_angle_tag(args_cli.min_up_deviation_deg)}deg"
        return filtered_prefix, f"{filtered_prefix}{suffix}"

    requested = os.path.abspath(os.path.expanduser(args_cli.filtered_cache))
    if requested.endswith(".npy"):
        if not requested.endswith(suffix):
            raise ValueError(f"Filtered cache must end with '{suffix}', got '{requested}'.")
        return requested[: -len(suffix)], requested
    return requested, f"{requested}{suffix}"


def _quat_rotate_wxyz(quaternions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate vectors by normalized WXYZ quaternions."""
    quaternions = np.asarray(quaternions, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-8):
        bad_index = int(np.flatnonzero(norms.reshape(-1) < 1.0e-8)[0])
        raise ValueError(f"Cache row {bad_index} contains a zero-length object quaternion.")
    quaternions = quaternions / norms

    quaternion_vector = quaternions[..., 1:4]
    if vectors.ndim == 1:
        vectors = np.broadcast_to(vectors, quaternion_vector.shape)
    twice_cross = 2.0 * np.cross(quaternion_vector, vectors)
    return vectors + quaternions[..., :1] * twice_cross + np.cross(quaternion_vector, twice_cross)


def _extract_large_deviation_cache(
    env_cfg: DirectRLEnvCfg,
    source_file: str,
    filtered_file: str,
) -> tuple[np.ndarray, np.ndarray]:
    cache = np.load(source_file, allow_pickle=False)
    if cache.ndim != 2 or cache.shape[1] != CACHE_STATE_DIM:
        raise ValueError(f"Expected cache shape (N, {CACHE_STATE_DIM}), got {cache.shape}.")
    if not np.isfinite(cache).all():
        raise ValueError(f"Source cache contains NaN or infinite values: {source_file}")

    local_up = np.asarray(env_cfg.object_up_axis, dtype=np.float64)
    local_up_norm = np.linalg.norm(local_up)
    if local_up_norm < 1.0e-8:
        raise ValueError(f"object_up_axis must be non-zero, got {env_cfg.object_up_axis}.")
    local_up /= local_up_norm

    reference_quaternion = np.asarray(env_cfg.object_cfg.init_state.rot, dtype=np.float64)[None, :]
    reference_up = _quat_rotate_wxyz(reference_quaternion, local_up)[0]
    cached_up = _quat_rotate_wxyz(cache[:, OBJECT_QUATERNION_SLICE], local_up)
    cosine = np.clip(cached_up @ reference_up, -1.0, 1.0)
    deviation_deg = np.rad2deg(np.arccos(cosine))

    selected_indices = np.flatnonzero(deviation_deg >= args_cli.min_up_deviation_deg)
    if selected_indices.size == 0:
        raise ValueError(
            f"No cache pose has up deviation >= {args_cli.min_up_deviation_deg:g} deg. "
            f"Available maximum is {deviation_deg.max():.3f} deg."
        )

    filtered_cache = np.ascontiguousarray(cache[selected_indices])
    if os.path.abspath(source_file) == os.path.abspath(filtered_file):
        raise ValueError("Filtered cache path must differ from the source cache path.")
    os.makedirs(os.path.dirname(filtered_file), exist_ok=True)
    np.save(filtered_file, filtered_cache)

    selected_angles = deviation_deg[selected_indices]
    print(f"[INFO] Source cache: {source_file}")
    print(
        f"[INFO] Up deviation in source: min={deviation_deg.min():.3f} deg, "
        f"mean={deviation_deg.mean():.3f} deg, max={deviation_deg.max():.3f} deg"
    )
    print(
        f"[INFO] Selected {selected_indices.size}/{cache.shape[0]} poses with "
        f"up deviation >= {args_cli.min_up_deviation_deg:g} deg"
    )
    print(
        f"[INFO] Selected deviation: min={selected_angles.min():.3f} deg, "
        f"mean={selected_angles.mean():.3f} deg, max={selected_angles.max():.3f} deg"
    )
    print(f"[INFO] Filtered reset cache: {filtered_file}")
    return filtered_cache, selected_angles


def _set_camera_light() -> None:
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


def _policy_actions(agent: PPO | ProprioAdapt, algorithm: str, obs_dict: dict) -> torch.Tensor:
    if algorithm == "ProprioAdapt":
        input_dict = {
            "obs": agent.running_mean_std(obs_dict["obs"]),
            "proprio_hist": agent.sa_mean_std(obs_dict["proprio_hist"]),
        }
    else:
        input_dict = {
            "obs": agent.running_mean_std(obs_dict["obs"]),
            "priv_info": obs_dict["priv_info"],
        }
    return torch.clamp(agent.model.act_inference(input_dict), -1.0, 1.0)


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict) -> None:
    _validate_args()

    checkpoint_path = os.path.abspath(os.path.expanduser(args_cli.load_path))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    source_prefix, source_file = _resolve_source_cache(env_cfg)
    filtered_prefix, filtered_file = _resolve_filtered_cache(env_cfg, source_prefix)
    filtered_cache, _ = _extract_large_deviation_cache(env_cfg, source_file, filtered_file)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.gravity = (0.0, 0.0, -9.81)
    env_cfg.grasp_cache_path = filtered_prefix
    env_cfg.reset_random_quat = False
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_friction = False
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.randomize_joint_pos_offset = False
    env_cfg.gravity_curriculum = False
    env_cfg.force_scale = 0.0
    env_cfg.debug_show_axes = False
    env_cfg.debug_show_object_pos = False
    env_cfg.debug_show_object_vectors = args_cli.show_object_vectors

    algorithm = args_cli.algorithm if args_cli.algorithm is not None else agent_cfg["algo"]
    agent_cfg["algo"] = algorithm
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else agent_cfg["device"]
    agent_cfg["seed"] = args_cli.seed
    agent_cfg["load_path"] = checkpoint_path
    agent_cfg["algorithm"]["num_actors"] = args_cli.num_envs
    agent_cfg["algorithm"]["minibatch_size"] = min(args_cli.num_envs * 8, 32768)
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    env = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if not args_cli.headless:
            _set_camera_light()
        env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)

        log_dir = os.path.join(
            "logs",
            agent_cfg["algorithm"]["experiment_name"],
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
        agent_cls = {"PPO": PPO, "ProprioAdapt": ProprioAdapt}[algorithm]
        agent = agent_cls(env, output_dir=log_dir, full_config=config, create_output_dir=False)
        print(f"[INFO] Loading {algorithm} checkpoint: {checkpoint_path}")
        agent.restore_test(checkpoint_path)
        agent.set_eval()

        print(f"[INFO] Replaying resets from {filtered_cache.shape[0]} filtered cache poses")
        print("[INFO] Close the Isaac Sim window or press Ctrl+C to exit.")
        obs_dict = env.reset()
        step = 0
        while simulation_app.is_running() and (args_cli.max_steps is None or step < args_cli.max_steps):
            actions = _policy_actions(agent, algorithm, obs_dict)
            obs_dict, _, _, _ = env.step(actions)
            step += 1
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
