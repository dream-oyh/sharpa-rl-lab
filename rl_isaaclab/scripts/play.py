# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse
import sys
import shutil
import select
import termios
import tty
from contextlib import contextmanager

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--cache", type=str, default=None, help="Cache path.")
parser.add_argument(
    "--scales",
    type=float,
    nargs="+",
    default=None,
    help="Scale bins to play. Use a single value for a single-scale cache.",
)
parser.add_argument(
    "--bulb_model",
    choices=("original", "old", "new"),
    default=None,
    help="Bulb asset model to load for bulb tasks.",
)
parser.add_argument("--load_path", type=str, default=None, help="Checkpoint path.")
parser.add_argument("--max_agent_steps", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--algorithm", type=str, default=None, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint.")
parser.add_argument(
    "--reset_random_quat",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override global reset rotation DR; by default use the task configuration.",
)
parser.add_argument(
    "--show_object_vectors",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Draw the object's up (green), target up (blue), and heading (red) vectors (default: enabled).",
)
parser.add_argument(
    "--print_object_angvel",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Print the object's world-frame and rotation-axis angular velocities.",
)
parser.add_argument(
    "--disable_object_pos_error_obs",
    action="store_true",
    default=False,
    help="Disable object position error in policy observations for old checkpoints.",
)
parser.add_argument(
    "--angvel_print_interval",
    type=int,
    default=20,
    help="Number of policy steps between angular-velocity prints.",
)
parser.add_argument(
    "--wait_for_space_before_inference",
    action="store_true",
    default=False,
    help="Step zero actions after reset until the terminal receives a space key.",
)
parser.add_argument(
    "--wait_print_interval",
    type=int,
    default=20,
    help="Number of zero-action wait steps between status prints.",
)
parser.add_argument(
    "--up_angle_reset_deg",
    type=float,
    default=None,
    help="For up-align curriculum tasks, sample resets only from the bin containing this Up-angle.",
)
parser.add_argument("--video", action="store_true", help="Record a headless RGB video.")
parser.add_argument("--video_length", type=int, default=400, help="Recorded video length in policy steps.")
parser.add_argument(
    "--video_dir",
    type=str,
    default="videos/play",
    help="Directory that receives the recorded MP4.",
)
parser.add_argument("--camera_light", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--camera_origin_type",
    choices=("world", "env", "asset_root", "asset_body"),
    default="env",
    help="Frame used by the video/GUI camera eye and lookat.",
)
parser.add_argument("--camera_env_index", type=int, default=0, help="Environment index for env-relative camera origin.")
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=(0.38, -0.42, 0.82),
    help="Camera eye used for video/GUI play.",
)
parser.add_argument(
    "--camera_lookat",
    type=float,
    nargs=3,
    default=(-0.09559, -0.00517, 0.64),
    help="Camera target used for video/GUI play.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rl_isaaclab.algo.ppo.ppo import PPO
from rl_isaaclab.algo.padapt.padapt import ProprioAdapt
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
from rl_isaaclab.wrapper.config_wrapper import ConfigWrapper

from isaaclab.envs import DirectRLEnvCfg

import rl_isaaclab.tasks.inhand_rotate
import rl_isaaclab.tasks.inhand_rotate_bulb
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def set_camera_light():
    """Use the active viewport's camera light instead of lights authored in the stage."""
    import carb
    import omni.kit.actions.core

    action_registry = omni.kit.actions.core.get_action_registry()
    camera_light_action = action_registry.get_action(
        "omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera"
    )
    if camera_light_action is not None:
        camera_light_action.execute()
    else:
        # Keep camera lighting functional if the viewport lighting menu extension is unavailable.
        carb.settings.get_settings().set_bool("/rtx/useViewLightingMode", True)
        carb.log_warn("Viewport lighting action is unavailable; enabled camera light through RTX settings.")


@contextmanager
def terminal_cbreak_if_available():
    """Temporarily read single keypresses from stdin when running in a TTY."""
    if not sys.stdin.isatty():
        yield False
        return
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def space_pressed() -> bool:
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return False
    return sys.stdin.read(1) == " "


def policy_action(agent, algorithm: str, obs_dict: dict) -> torch.Tensor:
    if algorithm == "ProprioAdapt":
        input_dict = {
            "obs": agent.running_mean_std(obs_dict["obs"]),
            "proprio_hist": agent.sa_mean_std(obs_dict["proprio_hist"].detach()),
        }
    else:
        input_dict = {
            "obs": agent.running_mean_std(obs_dict["obs"]),
            "priv_info": obs_dict["priv_info"],
        }
    return torch.clamp(agent.model.act_inference(input_dict), -1.0, 1.0)


def scale_range_from_scales(scales: list[float]) -> list[float | int]:
    if len(scales) == 1:
        scale = float(scales[0])
        return [scale, scale, 1]
    if len(scales) < 1:
        raise ValueError("--scales must contain at least one value.")
    step = (scales[-1] - scales[0]) / (len(scales) - 1)
    expected = [scales[0] + step * index for index in range(len(scales))]
    if any(abs(float(actual) - float(target)) > 1.0e-6 for actual, target in zip(scales, expected)):
        raise ValueError("--scales must be a single value or evenly spaced values.")
    return [float(scales[0]), float(scales[-1]), len(scales)]


def alignment_angle_text(raw_env) -> str:
    if not hasattr(raw_env, "_alignment_angle"):
        return ""
    raw_env._refresh_lab()
    angle = torch.rad2deg(raw_env._alignment_angle(raw_env.object_rot))
    return f", angle mean/min/max={angle.mean().item():.2f}/{angle.min().item():.2f}/{angle.max().item():.2f} deg"


def wait_with_zero_actions(env: GymStyleEnvWrapper, obs_dict: dict) -> dict:
    raw_env = env.unwrapped
    if not sys.stdin.isatty():
        print(
            "[WAIT] stdin is not a TTY; cannot read a space key. Starting inference immediately.",
            flush=True,
        )
        return obs_dict

    print("[WAIT] Stepping zero actions. Press SPACE in this terminal to start policy inference.", flush=True)
    step = 0
    with terminal_cbreak_if_available():
        while simulation_app.is_running():
            if space_pressed():
                print(f"[WAIT] SPACE received after {step} zero-action steps. Starting inference.", flush=True)
                return obs_dict
            obs_dict, _, dones, _ = env.step(env.zero_actions())
            step += 1
            if step % args_cli.wait_print_interval == 0 or dones.any():
                print(
                    f"[WAIT] zero_step={step}, dones={int(dones.sum().item())}/{len(dones)}"
                    f"{alignment_angle_text(raw_env)}",
                    flush=True,
                )
    return obs_dict


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    if args_cli.angvel_print_interval <= 0:
        raise ValueError(
            f"--angvel_print_interval must be positive, got {args_cli.angvel_print_interval}."
        )
    if args_cli.wait_print_interval <= 0:
        raise ValueError(
            f"--wait_print_interval must be positive, got {args_cli.wait_print_interval}."
        )
    if args_cli.video_length <= 0:
        raise ValueError(f"--video_length must be positive, got {args_cli.video_length}.")
    shutil.rmtree('outputs/', ignore_errors=True)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["algorithm"]["max_agent_steps"] = args_cli.max_agent_steps if args_cli.max_agent_steps is not None else agent_cfg["algorithm"]["max_agent_steps"]
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg['seed']
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else agent_cfg["device"]
    agent_cfg["algo"] = args_cli.algorithm if args_cli.algorithm is not None else agent_cfg["algo"]
    agent_cfg["load_path"] = args_cli.load_path if args_cli.load_path is not None else agent_cfg["load_path"]
    if args_cli.bulb_model is not None:
        if not hasattr(env_cfg, "set_bulb_model"):
            raise ValueError(
                f"--bulb_model is only supported by bulb tasks, got task {args_cli.task!r}."
            )
        env_cfg.set_bulb_model(args_cli.bulb_model)
    if args_cli.scales is not None:
        env_cfg.scale_range = scale_range_from_scales(args_cli.scales)
        env_cfg.events.randomize_scale.params["scale_range"] = env_cfg.scale_range
    if args_cli.reset_random_quat is not None:
        env_cfg.reset_random_quat = args_cli.reset_random_quat
    if args_cli.disable_object_pos_error_obs and getattr(
        env_cfg, "include_object_pos_error_in_policy_obs", False
    ):
        env_cfg.include_object_pos_error_in_policy_obs = False
        env_cfg.observation_space = int(env_cfg.observation_space) - 9
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_friction = True
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.randomize_joint_pos_offset = False
    env_cfg.sim.gravity = (0, 0, -9.81)
    env_cfg.gravity_curriculum = False
    env_cfg.debug_show_object_vectors = args_cli.show_object_vectors
    env_cfg.debug_print_object_angvel = args_cli.print_object_angvel
    env_cfg.debug_print_object_angvel_interval = args_cli.angvel_print_interval
    if args_cli.video:
        env_cfg.viewer.origin_type = args_cli.camera_origin_type
        env_cfg.viewer.env_index = args_cli.camera_env_index
        env_cfg.viewer.eye = tuple(args_cli.camera_eye)
        env_cfg.viewer.lookat = tuple(args_cli.camera_lookat)
    if args_cli.up_angle_reset_deg is not None:
        if not hasattr(env_cfg, "up_angle_curriculum_fixed_angle_deg"):
            raise ValueError(
                f"--up_angle_reset_deg is only supported by up-angle curriculum tasks, got {args_cli.task!r}."
            )
        env_cfg.up_angle_curriculum_fixed_angle_deg = args_cli.up_angle_reset_deg
        env_cfg.up_angle_curriculum_initial_bins = env_cfg.up_angle_curriculum_total_bins
    env_cfg.grasp_cache_path = args_cli.cache if args_cli.cache is not None else env_cfg.grasp_cache_path
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    # specify directory for logging experiments
    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg["algorithm"]["experiment_name"]))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if (not args_cli.headless or args_cli.video) and args_cli.camera_light:
        set_camera_light()
    if args_cli.video:
        video_dir = os.path.abspath(os.path.expanduser(args_cli.video_dir))
        os.makedirs(video_dir, exist_ok=True)
        print(f"[INFO] Recording {args_cli.video_length} policy steps to: {video_dir}")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
            name_prefix="play",
        )
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir=log_dir, full_config=config, create_output_dir=False)
    
    # load the checkpoint
    resume_path = agent_cfg["load_path"]
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    agent.restore_test(resume_path)
    if args_cli.video:
        agent.set_eval()
        obs_dict = env.reset()
        with torch.inference_mode():
            for _ in range(args_cli.video_length):
                action = policy_action(agent, agent_cfg["algo"], obs_dict)
                obs_dict, _, _, _ = env.step(action)
    elif args_cli.wait_for_space_before_inference:
        agent.set_eval()
        obs_dict = env.reset()
        obs_dict = wait_with_zero_actions(env, obs_dict)
        with torch.inference_mode():
            while simulation_app.is_running():
                action = policy_action(agent, agent_cfg["algo"], obs_dict)
                obs_dict, _, _, _ = env.step(action)
    else:
        agent.test()

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
