#!/usr/bin/env python3
"""Evaluate bulb scales and store the first episode in one structured memmap."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Inhand-Rotate-Sharpa-Wave-Bulb-v0")
parser.add_argument("--load_path", required=True, help="PPO or ProprioAdapt checkpoint.")
parser.add_argument("--algorithm", choices=("PPO", "ProprioAdapt"), default="ProprioAdapt")
parser.add_argument("--cache", default=None, help="Optional scale-range grasp-cache prefix.")
parser.add_argument("--output_dir", default=None, help="Output directory; a timestamped path is used by default.")
parser.add_argument("--num_envs_per_scale", type=int, default=100)
parser.add_argument("--scales", type=float, nargs="+", default=None, help="Scale bins to evaluate; defaults to 0.7 0.8 0.9 1.0 1.1.")
parser.add_argument("--bulb_model", choices=("original", "old", "new"), default=None, help="Bulb asset model to load.")
parser.add_argument("--episode_steps", type=int, default=400)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--curve_count", type=int, default=50)
parser.add_argument("--randomize_friction", action=argparse.BooleanOptionalAction, default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], "hydra.run.dir=.", "hydra.output_subdir=null"] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import DirectRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

from rl_isaaclab.algo.padapt.padapt import ProprioAdapt
from rl_isaaclab.algo.ppo.ppo import PPO
from rl_isaaclab.utils.bulb_scale_eval import (
    MEMMAP_DTYPE,
    analyze_dataset,
    create_memmap,
    write_metadata,
)
from rl_isaaclab.wrapper.config_wrapper import ConfigWrapper
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper

import rl_isaaclab.tasks.inhand_rotate  # noqa: F401,E402
import rl_isaaclab.tasks.inhand_rotate_bulb  # noqa: F401,E402


DEFAULT_SCALES = (0.7, 0.8, 0.9, 1.0, 1.1)


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


def _reshape_cpu(tensor: torch.Tensor, num_scales: int, num_envs_per_scale: int) -> np.ndarray:
    shape = (num_scales, num_envs_per_scale) + tuple(tensor.shape[1:])
    return tensor.detach().reshape(shape).cpu().numpy()


def _record_state(data: np.memmap, raw_env, actions: torch.Tensor, active: torch.Tensor, step: int) -> None:
    num_scales = data.shape[0]
    num_envs_per_scale = data.shape[1]
    active_np = _reshape_cpu(active, num_scales, num_envs_per_scale).astype(bool)
    data["valid"][:, :, step] = active_np
    state_fields = {
        "object_pos": raw_env.object_pos,
        "object_quat_wxyz": raw_env.object_rot,
        "target_quat_wxyz": raw_env.object_default_pose[:, 3:7],
        "object_linvel_w": raw_env.object_linvel,
        "object_angvel_w": raw_env.object_angvel,
        "object_up_w": raw_env.object_up_w,
        "target_up_w": raw_env.object_target_up_w,
        "action": actions,
        "episode_step": raw_env.episode_length_buf,
    }
    for name, tensor in state_fields.items():
        values = _reshape_cpu(tensor, num_scales, num_envs_per_scale)
        data[name][:, :, step][active_np] = values[active_np]


def _scale_range_from_scales(scales: tuple[float, ...]) -> list[float | int]:
    if len(scales) == 1:
        scale = float(scales[0])
        return [scale, scale, 1]
    expected = np.linspace(scales[0], scales[-1], len(scales))
    if not np.allclose(np.asarray(scales), expected, rtol=0.0, atol=1.0e-6):
        raise ValueError("--scales must be a single value or evenly spaced values.")
    return [float(scales[0]), float(scales[-1]), len(scales)]


def _validate_args() -> None:
    if args_cli.num_envs_per_scale <= 0:
        raise ValueError("--num_envs_per_scale must be positive.")
    if args_cli.episode_steps <= 0:
        raise ValueError("--episode_steps must be positive.")
    if not 0 < args_cli.curve_count <= args_cli.num_envs_per_scale:
        raise ValueError("--curve_count must be within [1, num_envs_per_scale].")


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict) -> None:
    _validate_args()
    checkpoint_path = Path(args_cli.load_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    scales = tuple(float(value) for value in (args_cli.scales or DEFAULT_SCALES))

    output_dir = Path(args_cli.output_dir).expanduser().resolve() if args_cli.output_dir else Path(
        "eval_data", f"bulb_scale_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    ).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty or absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_envs = len(scales) * args_cli.num_envs_per_scale
    env_cfg.scene.num_envs = total_envs
    env_cfg.seed = args_cli.seed
    # SharpaWaveInhandRotateEnv checks timeout against max_episode_length - 1.
    # Add one configured step so the first episode contains exactly the requested
    # number of policy transitions before truncation.
    policy_step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    env_cfg.episode_length_s = (args_cli.episode_steps + 1) * policy_step_dt
    if args_cli.bulb_model is not None:
        env_cfg.set_bulb_model(args_cli.bulb_model)
    env_cfg.scale_range = _scale_range_from_scales(scales)
    env_cfg.events.randomize_scale.params["scale_range"] = env_cfg.scale_range
    if args_cli.cache is not None:
        env_cfg.grasp_cache_path = args_cli.cache
    env_cfg.reset_random_quat = False
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_friction = args_cli.randomize_friction
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.randomize_joint_pos_offset = False
    env_cfg.gravity_curriculum = False
    env_cfg.force_scale = 0.0
    env_cfg.sim.gravity = (0.0, 0.0, -9.81)
    env_cfg.debug_show_axes = False
    env_cfg.debug_show_object_pos = False
    env_cfg.debug_show_object_vectors = False

    agent_cfg["algo"] = args_cli.algorithm
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else agent_cfg["device"]
    agent_cfg["seed"] = args_cli.seed
    agent_cfg["load_path"] = str(checkpoint_path)
    agent_cfg["algorithm"]["num_actors"] = total_envs
    agent_cfg["algorithm"]["minibatch_size"] = min(total_envs * 8, 32768)
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    env = None
    data = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
        raw_env = env.unwrapped
        if raw_env.max_episode_length - 1 != args_cli.episode_steps:
            raise ValueError(
                f"Environment timeout threshold is {raw_env.max_episode_length - 1}, "
                f"but --episode_steps is {args_cli.episode_steps}."
            )

        agent_class = {"PPO": PPO, "ProprioAdapt": ProprioAdapt}[args_cli.algorithm]
        agent = agent_class(env, output_dir=str(output_dir), full_config=config, create_output_dir=False)
        print(f"[INFO] Loading {args_cli.algorithm} checkpoint: {checkpoint_path}")
        agent.restore_test(str(checkpoint_path))
        agent.set_eval()

        shape = (len(scales), args_cli.num_envs_per_scale, args_cli.episode_steps)
        data = create_memmap(output_dir, shape)
        metadata = {
            "format_version": 1,
            "complete": False,
            "memmap_file": "bulb_eval.memmap",
            "dtype": MEMMAP_DTYPE.descr,
            "shape": list(shape),
            "layout": ["scale_bin", "environment", "policy_step"],
            "scales": list(scales),
            "num_envs_per_scale": args_cli.num_envs_per_scale,
            "episode_steps": args_cli.episode_steps,
            "environment_max_episode_length": int(raw_env.max_episode_length),
            "timeout_threshold_steps": int(raw_env.max_episode_length - 1),
            "step_dt_s": float(raw_env.step_dt),
            "task": args_cli.task,
            "algorithm": args_cli.algorithm,
            "checkpoint": str(checkpoint_path),
            "cache_prefix": str(env_cfg.grasp_cache_path),
            "seed": args_cli.seed,
            "gravity": list(env_cfg.sim.gravity),
            "randomize_friction": bool(env_cfg.randomize_friction),
            "success_definition": "First episode reaches timeout without an earlier termination.",
            "state_timing": "State is sampled immediately before the action at each policy step.",
            "angle_definition": "Wrapped XYZ Euler angles of target_quaternion^-1 * object_quaternion.",
        }
        write_metadata(output_dir, metadata)

        obs_dict = env.reset()
        active = torch.ones(total_envs, dtype=torch.bool, device=raw_env.device)
        print(f"[INFO] Evaluating {len(scales)} scale bins x {args_cli.num_envs_per_scale} envs x {args_cli.episode_steps} steps")
        with torch.inference_mode():
            for step in range(args_cli.episode_steps):
                actions = _policy_actions(agent, args_cli.algorithm, obs_dict)
                _record_state(data, raw_env, actions, active, step)
                obs_dict, rewards, dones, infos = env.step(actions)

                active_np = _reshape_cpu(active, len(scales), args_cli.num_envs_per_scale).astype(bool)
                data["reward"][:, :, step][active_np] = _reshape_cpu(rewards, len(scales), args_cli.num_envs_per_scale)[active_np]
                truncated = infos.get("time_outs", torch.zeros_like(dones, dtype=torch.bool)).bool()
                terminated = dones.bool() & ~truncated
                data["truncated"][:, :, step] = _reshape_cpu(truncated & active, len(scales), args_cli.num_envs_per_scale)
                data["terminated"][:, :, step] = _reshape_cpu(terminated & active, len(scales), args_cli.num_envs_per_scale)
                active &= ~dones.bool()
                if (step + 1) % 50 == 0 or step + 1 == args_cli.episode_steps:
                    print(f"[INFO] step {step + 1}/{args_cli.episode_steps}, active first episodes: {int(active.sum())}")
                    data.flush()

        data.flush()
        metadata["complete"] = True
        metadata["remaining_active_after_rollout"] = int(active.sum().item())
        write_metadata(output_dir, metadata)
        rows = analyze_dataset(output_dir, curve_count=args_cli.curve_count)
        print(f"[INFO] Results written to: {output_dir}")
        for row in rows:
            print(
                f"[RESULT] scale={row['scale']:.1f} success={row['success_count']}/{row['num_envs']} "
                f"({100.0 * row['success_rate']:.1f}%) omega_z_mean={row['omega_z_rad_s_mean']:.4f} rad/s"
            )
    finally:
        if data is not None:
            data.flush()
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
