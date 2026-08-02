# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse
import sys
import shutil

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--cache", type=str, default=None, help="Cache path.")
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
    "--angvel_print_interval",
    type=int,
    default=20,
    help="Number of policy steps between angular-velocity prints.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

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


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    if args_cli.angvel_print_interval <= 0:
        raise ValueError(
            f"--angvel_print_interval must be positive, got {args_cli.angvel_print_interval}."
        )
    shutil.rmtree('outputs/')
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["algorithm"]["max_agent_steps"] = args_cli.max_agent_steps if args_cli.max_agent_steps is not None else agent_cfg["algorithm"]["max_agent_steps"]
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg['seed']
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else agent_cfg["device"]
    agent_cfg["algo"] = args_cli.algorithm if args_cli.algorithm is not None else agent_cfg["algo"]
    agent_cfg["load_path"] = args_cli.load_path if args_cli.load_path is not None else agent_cfg["load_path"]
    if args_cli.reset_random_quat is not None:
        env_cfg.reset_random_quat = args_cli.reset_random_quat
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
    env_cfg.grasp_cache_path = args_cli.cache if args_cli.cache is not None else env_cfg.grasp_cache_path
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    # specify directory for logging experiments
    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg["algorithm"]["experiment_name"]))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if not args_cli.headless:
        set_camera_light()
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir=log_dir, full_config=config, create_output_dir=False)
    
    # load the checkpoint
    resume_path = agent_cfg["load_path"]
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    agent.restore_test(resume_path)
    agent.test()

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
