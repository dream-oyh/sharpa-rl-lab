# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Check whether an object's root position lies inside the hand grasp region."""

import argparse
import builtins
import functools
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect object root position relative to the hand and fingertips.")
parser.add_argument(
    "--which",
    type=str,
    default="both",
    choices=("cylinder", "bulb", "both"),
    help="Which object task to inspect.",
)
parser.add_argument("--num_envs", type=int, default=512, help="Number of sampled reset states.")
parser.add_argument("--steps", type=int, default=0, help="Number of zero-action steps after reset before measuring.")
parser.add_argument("--margin", type=float, default=0.0, help="Margin added to the fingertip AABB check in meters.")
parser.add_argument("--reward_terms", action="store_true", default=False, help="Print raw and weighted reward terms.")
parser.add_argument("--cylinder_cache", type=str, default=None, help="Override cylinder grasp cache prefix.")
parser.add_argument("--bulb_cache", type=str, default=None, help="Override bulb grasp cache prefix.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_apply_inverse

from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper

import rl_isaaclab.tasks.inhand_rotate
import rl_isaaclab.tasks.inhand_rotate_bulb
from rl_isaaclab.tasks.inhand_rotate.sharpa_wave_env_cfg import SharpaWaveEnvCfg
from rl_isaaclab.tasks.inhand_rotate_bulb.sharpa_wave_bulb_env_cfg import SharpaWaveBulbEnvCfg

print = functools.partial(builtins.print, flush=True)


TASKS = {
    "cylinder": (
        "Isaac-Inhand-Rotate-Sharpa-Wave-v0",
        SharpaWaveEnvCfg,
        "cylinder_cache",
    ),
    "bulb": (
        "Isaac-Inhand-Rotate-Sharpa-Wave-Bulb-v0",
        SharpaWaveBulbEnvCfg,
        "bulb_cache",
    ),
}


def _summarize(name: str, values: torch.Tensor, unit: str = "m"):
    values = values.detach().float().cpu()
    print(
        f"  {name}: mean={values.mean().item(): .6f} {unit}, "
        f"std={values.std(unbiased=False).item(): .6f}, "
        f"min={values.min().item(): .6f}, max={values.max().item(): .6f}"
    )


def _summarize_vec(name: str, values: torch.Tensor, unit: str = "m"):
    values = values.detach().float().cpu()
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    min_v = values.min(dim=0).values
    max_v = values.max(dim=0).values
    print(f"  {name} [{unit}]")
    print(f"    mean: x={mean[0]: .6f}, y={mean[1]: .6f}, z={mean[2]: .6f}")
    print(f"    std : x={std[0]: .6f}, y={std[1]: .6f}, z={std[2]: .6f}")
    print(f"    min : x={min_v[0]: .6f}, y={min_v[1]: .6f}, z={min_v[2]: .6f}")
    print(f"    max : x={max_v[0]: .6f}, y={max_v[1]: .6f}, z={max_v[2]: .6f}")


def _make_env(which: str):
    task_name, cfg_cls, cache_arg = TASKS[which]
    print(f"creating env: {task_name}")
    env_cfg = cfg_cls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.reset_random_quat = False
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_friction = False
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.randomize_joint_pos_offset = False
    env_cfg.gravity_curriculum = False
    cache_override = getattr(args_cli, cache_arg)
    if cache_override is not None:
        env_cfg.grasp_cache_path = cache_override
    env = gym.make(task_name, cfg=env_cfg, render_mode=None)
    return GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().mean().item()
    return float(value)


def _print_reward_terms(raw_env):
    extras = raw_env.extras
    specs = [
        ("rotate_reward", "rotate_reward_scale"),
        ("object_linvel_penalty", "object_linvel_penalty_scale"),
        ("pos_diff_penalty", "pos_diff_penalty_scale"),
        ("torque_penalty", "torque_penalty_scale"),
        ("work_penalty", "work_penalty_scale"),
        ("object_pos_diff", "object_pos_reward_scale"),
        ("object_axis_align_penalty", "object_axis_align_penalty_scale"),
    ]
    print("reward term contributions:")
    weighted_sum = 0.0
    for term_name, scale_name in specs:
        if term_name not in extras:
            continue
        raw = _to_float(extras[term_name])
        scale = float(getattr(raw_env.cfg, scale_name))
        weighted = raw * scale
        weighted_sum += weighted
        print(f"  {term_name}: raw={raw: .6f}, scale={scale: .6f}, weighted={weighted: .6f}")
    if "object_axis_align_angle" in extras:
        print(f"  object_axis_align_angle: raw={_to_float(extras['object_axis_align_angle']): .6f} rad")
    if "total_reward" in extras:
        print(f"  total_reward extras: {_to_float(extras['total_reward']): .6f}")
    print(f"  weighted sum from listed terms: {weighted_sum: .6f}")


def _measure(which: str):
    print(f"\n=== {which} ===")
    env = _make_env(which)
    print(f"env created: {which}")
    raw_env = env.unwrapped

    for _ in range(args_cli.steps):
        _, _, done, _ = env.step(env.zero_actions())
        if done.any():
            raw_env.reset()

    raw_env._refresh_lab()
    num_envs = raw_env.num_envs
    device = raw_env.device
    env_origins = raw_env.scene.env_origins

    object_root_l = raw_env.object.data.root_pos_w - env_origins
    hand_root_l = raw_env.hand.data.root_pos_w - env_origins
    hand_quat_w = raw_env.hand.data.root_quat_w
    fingertip_l = raw_env.hand.data.body_pos_w[:, raw_env.finger_bodies] - env_origins[:, None, :]

    object_root_h = quat_apply_inverse(hand_quat_w, object_root_l - hand_root_l)
    fingertip_h = quat_apply_inverse(
        hand_quat_w[:, None, :].expand(-1, raw_env.num_fingertips, -1).reshape(-1, 4),
        (fingertip_l - hand_root_l[:, None, :]).reshape(-1, 3),
    ).reshape(num_envs, raw_env.num_fingertips, 3)

    fingertip_min_h = fingertip_h.min(dim=1).values
    fingertip_max_h = fingertip_h.max(dim=1).values
    inside_axis = (object_root_h >= fingertip_min_h - args_cli.margin) & (
        object_root_h <= fingertip_max_h + args_cli.margin
    )
    inside_tip_aabb = inside_axis.all(dim=-1)

    fingertip_center_h = fingertip_h.mean(dim=1)
    root_to_tip_center_h = object_root_h - fingertip_center_h
    root_to_tip_center_dist = torch.norm(root_to_tip_center_h, dim=-1)

    palm_to_tip_center = fingertip_center_h
    denom = torch.clamp((palm_to_tip_center * palm_to_tip_center).sum(dim=-1), min=1.0e-9)
    palm_to_tip_alpha = (object_root_h * palm_to_tip_center).sum(dim=-1) / denom
    palm_to_tip_radial_dist = torch.norm(object_root_h - palm_to_tip_alpha[:, None] * palm_to_tip_center, dim=-1)
    between_palm_and_tips = (palm_to_tip_alpha >= 0.0) & (palm_to_tip_alpha <= 1.0)

    print(f"task num_envs: {num_envs}")
    print(f"cache prefix: {raw_env.cfg.grasp_cache_path}")
    print(f"scale_range: {raw_env.cfg.scale_range}")
    print(f"zero-action steps before measurement: {args_cli.steps}")
    print(f"fingertip AABB margin: {args_cli.margin:.4f} m")
    print(f"inside fingertip AABB ratio: {inside_tip_aabb.float().mean().item():.4f}")
    print(
        "inside per hand-axis ratio: "
        f"x={inside_axis[:, 0].float().mean().item():.4f}, "
        f"y={inside_axis[:, 1].float().mean().item():.4f}, "
        f"z={inside_axis[:, 2].float().mean().item():.4f}"
    )
    print(f"between palm root and fingertip-center ratio: {between_palm_and_tips.float().mean().item():.4f}")
    _summarize_vec("object root in hand-root frame", object_root_h)
    _summarize_vec("fingertip center in hand-root frame", fingertip_center_h)
    _summarize_vec("object root minus fingertip center", root_to_tip_center_h)
    _summarize("distance(root, fingertip center)", root_to_tip_center_dist)
    _summarize("palm-to-fingertip-center alpha", palm_to_tip_alpha, unit="")
    _summarize("radial distance to palm-fingertip-center line", palm_to_tip_radial_dist)
    _summarize_vec("fingertip AABB min in hand-root frame", fingertip_min_h)
    _summarize_vec("fingertip AABB max in hand-root frame", fingertip_max_h)
    if args_cli.reward_terms:
        _print_reward_terms(raw_env)

    env.close()


def main():
    targets = ("cylinder", "bulb") if args_cli.which == "both" else (args_cli.which,)
    for which in targets:
        _measure(which)


if __name__ == "__main__":
    main()
    simulation_app.close()
