# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate grasp caches, using all environments on one object scale at a time."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from isaaclab.app import AppLauncher


_GRASP_PLAN_MARKER = "__SHARPA_GRASP_PLAN__="
_original_cli_args = sys.argv[1:].copy()

parser = argparse.ArgumentParser(description="Generate a stable grasp cache.")
parser.add_argument(
    "--num_envs",
    type=int,
    default=8192,
    help="Number of environments used for each object scale.",
)
parser.add_argument("--task", type=str, required=True, help="Grasp task name.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--max_agent_steps", type=int, default=None)
parser.add_argument("--algorithm", type=str, default=None)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument(
    "--parallel_scales",
    action="store_true",
    default=False,
    help="Use the legacy behavior that divides environments between all scales.",
)

# Internal arguments used by the sequential parent process.
parser.add_argument("--grasp_worker", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--print_grasp_plan", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--grasp_scale", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--grasp_scale_index", type=int, default=None, help=argparse.SUPPRESS)
parser.add_argument(
    "--grasp_original_scale_count", type=int, default=None, help=argparse.SUPPRESS
)
parser.add_argument(
    "--grasp_worker_cache_size", type=int, default=None, help=argparse.SUPPRESS
)
parser.add_argument(
    "--grasp_worker_output_prefix", type=str, default=None, help=argparse.SUPPRESS
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def _worker_command(*extra_args: str) -> list[str]:
    return [
        sys.executable,
        os.path.abspath(__file__),
        *_original_cli_args,
        "--grasp_worker",
        *extra_args,
    ]


def _probe_grasp_plan() -> dict:
    with tempfile.TemporaryDirectory(prefix="sharpa_grasp_probe_") as probe_dir:
        result = subprocess.run(
            _worker_command(
                "--print_grasp_plan", f"hydra.run.dir={probe_dir}"
            ),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    plan_line = next(
        (
            line[len(_GRASP_PLAN_MARKER) :]
            for line in result.stdout.splitlines()
            if line.startswith(_GRASP_PLAN_MARKER)
        ),
        None,
    )
    if result.returncode != 0 or plan_line is None:
        print(result.stdout, end="", flush=True)
        raise RuntimeError("Failed to read the grasp task's scale plan.")
    return json.loads(plan_line)


def _scale_values(scale_range: list[float | int]) -> list[float]:
    lower, upper, count_value = scale_range
    count = int(count_value)
    if count <= 0:
        raise ValueError(f"Scale count must be positive, got {scale_range}.")
    if count == 1:
        return [float(lower)]
    step = (float(upper) - float(lower)) / (count - 1)
    return [round(float(lower) + step * index, 10) for index in range(count)]


def _run_sequential_collection() -> int:
    print(
        f"[SEQUENTIAL GRASP] Reading scale plan for {args_cli.task}...",
        flush=True,
    )
    plan = _probe_grasp_plan()
    scale_range = plan["scale_range"]
    scales = _scale_values(scale_range)
    num_scales = len(scales)
    requested_total = int(plan["grasp_cache_size"])
    angle_bins = int(plan["grasp_angle_bins"])
    output_prefix = str(plan["output_prefix"])

    requested_per_scale = requested_total // num_scales
    poses_per_angle_bucket = requested_per_scale // angle_bins
    if poses_per_angle_bucket <= 0:
        raise ValueError(
            f"grasp_cache_size={requested_total} is too small for "
            f"{num_scales} scales x {angle_bins} angle bins."
        )
    effective_per_scale = poses_per_angle_bucket * angle_bins
    effective_total = effective_per_scale * num_scales

    print(
        "[SEQUENTIAL GRASP] "
        f"task={args_cli.task} | scales={scales} | "
        f"envs_per_scale={args_cli.num_envs} | "
        f"target={effective_total} ({effective_per_scale}/scale, "
        f"{poses_per_angle_bucket}/angle-bin)",
        flush=True,
    )

    output_arrays = []
    with tempfile.TemporaryDirectory(prefix="sharpa_grasp_scales_") as temp_dir:
        for scale_index, scale in enumerate(scales):
            worker_prefix = str(Path(temp_dir) / f"scale_{scale_index}")
            print(
                f"[SEQUENTIAL GRASP] Starting scale {scale_index + 1}/"
                f"{num_scales}: {scale:g} with {args_cli.num_envs} envs",
                flush=True,
            )
            command = _worker_command(
                "--grasp_scale",
                repr(scale),
                "--grasp_scale_index",
                str(scale_index),
                "--grasp_original_scale_count",
                str(num_scales),
                "--grasp_worker_cache_size",
                str(effective_per_scale),
                "--grasp_worker_output_prefix",
                worker_prefix,
            )
            subprocess.run(command, check=True)

            worker_path = Path(f"{worker_prefix}_{scale}-{scale}-1.npy")
            if not worker_path.is_file():
                raise FileNotFoundError(
                    f"Scale worker completed without producing {worker_path}."
                )
            scale_cache = np.load(worker_path, allow_pickle=False)
            if scale_cache.shape != (effective_per_scale, 29):
                raise ValueError(
                    f"Scale {scale:g} produced cache shape {scale_cache.shape}; "
                    f"expected ({effective_per_scale}, 29)."
                )
            output_arrays.append(scale_cache)
            print(
                f"[SEQUENTIAL GRASP] Finished scale {scale_index + 1}/"
                f"{num_scales}: {scale:g} ({len(scale_cache)} poses)",
                flush=True,
            )

        merged_cache = np.concatenate(output_arrays, axis=0)

    lower, upper, count = scale_range
    output_path = Path(f"{output_prefix}_{lower}-{upper}-{int(count)}.npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, merged_cache)
    print(
        f"[SEQUENTIAL GRASP] Done: {output_path} | shape={merged_cache.shape}",
        flush=True,
    )
    return 0


if not args_cli.grasp_worker and not args_cli.parallel_scales:
    raise SystemExit(_run_sequential_collection())

# Workers and the explicit legacy mode launch Isaac Sim in this process.
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_isaaclab.tasks.inhand_rotate
import rl_isaaclab.tasks.inhand_rotate_bulb
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    if args_cli.grasp_scale is not None:
        worker_scale_range = [args_cli.grasp_scale, args_cli.grasp_scale, 1]
        env_cfg.scale_range = worker_scale_range
        env_cfg.events.randomize_scale.params["scale_range"] = worker_scale_range
        env_cfg.grasp_cache_size = args_cli.grasp_worker_cache_size
        env_cfg.save_grasp_cache_path = args_cli.grasp_worker_output_prefix
        # Preserve an explicit task-level seed mapping.  This is needed when a
        # single-scale collector reads its bucket from a legacy multi-scale
        # seed cache (for example scale 1.0 from a five-scale bulb cache).
        if (
            hasattr(env_cfg, "seed_grasp_cache_scale_id")
            and env_cfg.seed_grasp_cache_scale_id is None
        ):
            env_cfg.seed_grasp_cache_scale_id = args_cli.grasp_scale_index
            env_cfg.seed_grasp_cache_scale_count = (
                args_cli.grasp_original_scale_count
            )

    if args_cli.print_grasp_plan:
        output_prefix = (
            getattr(env_cfg, "save_grasp_cache_path", None)
            or env_cfg.grasp_cache_path
            or "cache/sharpa_grasp_linspace"
        )
        print(
            _GRASP_PLAN_MARKER
            + json.dumps(
                {
                    "scale_range": list(env_cfg.scale_range),
                    "grasp_cache_size": int(env_cfg.grasp_cache_size),
                    "grasp_angle_bins": int(env_cfg.grasp_angle_bins),
                    "output_prefix": output_prefix,
                    "seed_scale_id": getattr(
                        env_cfg, "seed_grasp_cache_scale_id", None
                    ),
                    "seed_scale_count": getattr(
                        env_cfg, "seed_grasp_cache_scale_count", None
                    ),
                }
            ),
            flush=True,
        )
        return

    shutil.rmtree("outputs", ignore_errors=True)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    agent_cfg["algorithm"]["minibatch_size"] = min(args_cli.num_envs * 8, 32768)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)

    env.reset()
    while True:
        actions = env.zero_actions()
        env.step(actions)


if __name__ == "__main__":
    main()
    simulation_app.close()
