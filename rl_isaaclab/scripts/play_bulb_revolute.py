# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a bulb policy with the bulb constrained to a fixed revolute joint.

The socket is represented by an invisible, collision-free kinematic anchor.  A
USD revolute joint connects that anchor to the existing rigid bulb and leaves
only rotation about the bulb's cached local Z axis free.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replay a bulb policy with a fixed revolute-joint socket.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Inhand-Rotate-Sharpa-Wave-Bulb-v0",
    help="Task whose environment and agent configurations are used.",
)
parser.add_argument(
    "--cache",
    type=str,
    default="cache/sharpa_bulb_grasp_high_1p0_angle25_0.7-1.1-5.npy",
    help="Complete five-scale bulb cache file.",
)
parser.add_argument(
    "--cache_scale",
    type=float,
    default=1.0,
    help="Object-scale bucket selected from the cache.",
)
parser.add_argument(
    "--cache_row",
    type=int,
    default=1815,
    help="Row inside the selected scale bucket (default: a near-axis 1.0-scale grasp).",
)
parser.add_argument(
    "--load_path",
    type=str,
    default="logs/bulb_debug/2026-07-27_13-26-55/stage1_nn/best.pth",
    help="PPO checkpoint to replay.",
)
parser.add_argument(
    "--joint_offset_z",
    type=float,
    default=-0.046,
    help="Revolute-joint origin along the bulb's local Z axis, in meters.",
)
parser.add_argument(
    "--joint_damping",
    type=float,
    default=0.0,
    help="Optional passive angular-drive damping. Zero leaves the joint fully passive.",
)
parser.add_argument(
    "--print_interval",
    type=int,
    default=20,
    help="Policy-step interval for constraint diagnostics.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Optional policy-step limit. By default, replay continues until the viewer closes.",
)
parser.add_argument("--video", action="store_true", help="Record one headless RGB video.")
parser.add_argument(
    "--camera_light",
    action="store_true",
    help="Use Isaac Sim camera lighting for both viewport display and headless RGB recording.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Recorded video length in policy steps.",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default="videos/bulb_revolute",
    help="Directory that receives the recorded MP4.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# Keep this visualization utility from creating Hydra output directories.
sys.argv = [sys.argv[0], "hydra.run.dir=.", "hydra.output_subdir=null"] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below this point runs after Isaac Sim has started."""

from datetime import datetime

import gymnasium as gym
import numpy as np
import omni.usd
import torch
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

from rl_isaaclab.algo.ppo.ppo import PPO
from rl_isaaclab.tasks.inhand_rotate.sharpa_wave_env import SharpaWaveInhandRotateEnv, quat_rotate
from rl_isaaclab.wrapper.config_wrapper import ConfigWrapper
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper

import rl_isaaclab.tasks.inhand_rotate  # noqa: F401,E402
import rl_isaaclab.tasks.inhand_rotate_bulb  # noqa: F401,E402


CACHE_STATE_DIM = 29
OBJECT_POS_SLICE = slice(22, 25)
OBJECT_QUAT_SLICE = slice(25, 29)


class BulbRevoluteReplayEnv(SharpaWaveInhandRotateEnv):
    """Bulb task with an invisible fixed anchor and a passive revolute joint."""

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        cache_state = np.asarray(cfg.revolute_cache_state, dtype=np.float32)
        if cache_state.shape != (CACHE_STATE_DIM,):
            raise ValueError(f"Expected one ({CACHE_STATE_DIM},) cache row, got {cache_state.shape}.")

        # The parent loads caches after scene construction.  Disable its file
        # lookup and inject the exact selected row once all parent buffers exist.
        cfg.grasp_cache_path = None
        super().__init__(cfg, render_mode, **kwargs)
        self.saved_grasping_states = torch.as_tensor(
            cache_state, dtype=torch.float32, device=self.device
        ).reshape(1, CACHE_STATE_DIM)
        self.bucket_grasp = 1
        self.bucket_env = 1

        cached_quat = self.saved_grasping_states[:, OBJECT_QUAT_SLICE]
        local_z = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32, device=self.device).reshape(1, 3)
        self.revolute_axis_w = quat_rotate(cached_quat, local_z)
        self.revolute_axis_w /= torch.clamp(
            torch.norm(self.revolute_axis_w, dim=-1, keepdim=True), min=1.0e-6
        )
        self.revolute_reference_pos = self.saved_grasping_states[:, OBJECT_POS_SLICE].clone()

    def _setup_scene(self):
        super()._setup_scene()

        stage = omni.usd.get_context().get_stage()
        object_pos = tuple(float(value) for value in self.cfg.revolute_object_pos)
        object_quat = tuple(float(value) for value in self.cfg.revolute_object_quat)
        joint_offset_z = float(self.cfg.revolute_joint_offset_z)
        joint_damping = float(self.cfg.revolute_joint_damping)

        for env_id in range(self.cfg.scene.num_envs):
            env_path = f"/World/envs/env_{env_id}"
            env_origin = self.scene.env_origins[env_id].detach().cpu().tolist()
            anchor_position = tuple(object_pos[index] + env_origin[index] for index in range(3))

            anchor_path = f"{env_path}/bulb_socket_anchor"
            anchor_prim = UsdGeom.Xform.Define(stage, anchor_path).GetPrim()
            anchor_xform = UsdGeom.Xformable(anchor_prim)
            anchor_xform.AddTranslateOp().Set(Gf.Vec3d(*anchor_position))
            anchor_xform.AddOrientOp().Set(
                Gf.Quatf(object_quat[0], Gf.Vec3f(*object_quat[1:4]))
            )
            rigid_api = UsdPhysics.RigidBodyAPI.Apply(anchor_prim)
            rigid_api.CreateRigidBodyEnabledAttr(True)
            rigid_api.CreateKinematicEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(anchor_prim).CreateMassAttr(1.0)

            joint_path = f"{env_path}/bulb_socket_revolute_joint"
            joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
            joint.CreateBody0Rel().SetTargets([Sdf.Path(anchor_path)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{env_path}/object")])
            joint.CreateAxisAttr(UsdPhysics.Tokens.z)
            joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, joint_offset_z))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, joint_offset_z))
            joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
            joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
            joint.CreateCollisionEnabledAttr(False)

            if joint_damping > 0.0:
                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(0.0)
                drive.CreateDampingAttr(joint_damping)
                drive.CreateTargetVelocityAttr(0.0)
                drive.CreateMaxForceAttr(float("inf"))


def _set_camera_light() -> None:
    import carb
    import omni.kit.actions.core

    # The RTX renderer watches this setting in both interactive and headless modes.
    carb.settings.get_settings().set_bool("/rtx/useViewLightingMode", True)
    action_registry = omni.kit.actions.core.get_action_registry()
    action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera")
    if action is not None:
        action.execute()
    print("[INFO] Camera light enabled: /rtx/useViewLightingMode=True")


def _load_cache_row(env_cfg: DirectRLEnvCfg) -> tuple[np.ndarray, int, np.ndarray]:
    cache_path = os.path.abspath(os.path.expanduser(args_cli.cache))
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"Cache file does not exist: {cache_path}")

    cache = np.load(cache_path, allow_pickle=False)
    if cache.ndim != 2 or cache.shape[1] != CACHE_STATE_DIM:
        raise ValueError(f"Expected cache shape (N, {CACHE_STATE_DIM}), got {cache.shape}.")
    if not np.isfinite(cache).all():
        raise ValueError(f"Cache contains NaN or infinite values: {cache_path}")

    scale_lower, scale_upper, scale_count = env_cfg.scale_range
    scale_count = int(scale_count)
    if cache.shape[0] % scale_count != 0:
        raise ValueError(
            f"Cache row count {cache.shape[0]} is not divisible by scale count {scale_count}."
        )
    scales = np.linspace(float(scale_lower), float(scale_upper), scale_count)
    scale_id = int(np.argmin(np.abs(scales - args_cli.cache_scale)))
    if not np.isclose(scales[scale_id], args_cli.cache_scale, atol=1.0e-6):
        raise ValueError(f"Requested scale {args_cli.cache_scale:g} is not in {scales.tolist()}.")

    rows_per_scale = cache.shape[0] // scale_count
    if not 0 <= args_cli.cache_row < rows_per_scale:
        raise ValueError(
            f"--cache_row must be in [0, {rows_per_scale - 1}], got {args_cli.cache_row}."
        )
    global_row = scale_id * rows_per_scale + args_cli.cache_row
    return np.ascontiguousarray(cache[global_row]), global_row, scales


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict) -> None:
    if args_cli.print_interval <= 0:
        raise ValueError(f"--print_interval must be positive, got {args_cli.print_interval}.")
    if args_cli.max_steps is not None and args_cli.max_steps <= 0:
        raise ValueError(f"--max_steps must be positive, got {args_cli.max_steps}.")
    if args_cli.video_length <= 0:
        raise ValueError(f"--video_length must be positive, got {args_cli.video_length}.")
    if args_cli.joint_damping < 0.0:
        raise ValueError(f"--joint_damping must be non-negative, got {args_cli.joint_damping}.")

    checkpoint_path = os.path.abspath(os.path.expanduser(args_cli.load_path))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    cache_state, global_cache_row, cache_scales = _load_cache_row(env_cfg)
    object_pos = cache_state[OBJECT_POS_SLICE]
    object_quat = cache_state[OBJECT_QUAT_SLICE]

    # One hand and one bulb, using the exact 1.0-scale cache row without any
    # startup or reset randomization.
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 1.0
    env_cfg.scale_range = [args_cli.cache_scale, args_cli.cache_scale, 1]
    env_cfg.events.randomize_scale.params["scale_range"] = env_cfg.scale_range
    env_cfg.object_cfg.spawn.scale = (args_cli.cache_scale,) * 3
    env_cfg.object_cfg.init_state.pos = tuple(float(value) for value in object_pos)
    env_cfg.object_cfg.init_state.rot = tuple(float(value) for value in object_quat)
    env_cfg.revolute_cache_state = tuple(float(value) for value in cache_state)
    env_cfg.revolute_object_pos = tuple(float(value) for value in object_pos)
    env_cfg.revolute_object_quat = tuple(float(value) for value in object_quat)
    env_cfg.revolute_joint_offset_z = args_cli.joint_offset_z
    env_cfg.revolute_joint_damping = args_cli.joint_damping

    env_cfg.seed = 42
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.gravity = (0.0, 0.0, -9.81)
    env_cfg.reset_random_quat = False
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_friction = False
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.randomize_joint_pos_offset = False
    env_cfg.gravity_curriculum = False
    env_cfg.force_scale = 0.0
    env_cfg.joint_noise_scale = 0.0
    env_cfg.debug_show_axes = False
    env_cfg.debug_show_object_pos = True
    env_cfg.debug_show_object_vectors = True
    env_cfg.viewer.eye = (0.35, -0.35, 0.85)
    env_cfg.viewer.lookat = (float(object_pos[0]), float(object_pos[1]), float(object_pos[2]))
    env_cfg.viewer.resolution = (1280, 720)

    agent_cfg["algo"] = "PPO"
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else agent_cfg["device"]
    agent_cfg["seed"] = 42
    agent_cfg["load_path"] = checkpoint_path
    agent_cfg["algorithm"]["num_actors"] = 1
    agent_cfg["algorithm"]["minibatch_size"] = 8
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    print(f"[INFO] Cache: {os.path.abspath(args_cli.cache)}")
    print(f"[INFO] Available cache scales: {cache_scales.tolist()}")
    print(f"[INFO] Selected scale={args_cli.cache_scale:g}, global row={global_cache_row}")
    print(f"[INFO] Cached bulb pose: pos={object_pos.tolist()}, quat_wxyz={object_quat.tolist()}")
    print(
        f"[INFO] Invisible socket joint: local Z axis, local pivot=(0, 0, {args_cli.joint_offset_z:g}) m, "
        f"damping={args_cli.joint_damping:g}"
    )

    env = None
    try:
        env = BulbRevoluteReplayEnv(
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.video else None,
        )
        if args_cli.camera_light:
            _set_camera_light()
        if args_cli.video:
            video_dir = os.path.abspath(os.path.expanduser(args_cli.video_dir))
            print(f"[INFO] Recording {args_cli.video_length} policy steps to: {video_dir}")
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_dir,
                step_trigger=lambda step: step == 0,
                video_length=args_cli.video_length,
                disable_logger=True,
                name_prefix="bulb-revolute",
            )
        env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)

        log_dir = os.path.join(
            "logs",
            agent_cfg["algorithm"]["experiment_name"],
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
        agent = PPO(env, output_dir=log_dir, full_config=config, create_output_dir=False)
        print(f"[INFO] Loading PPO checkpoint: {checkpoint_path}")
        agent.restore_test(checkpoint_path)
        agent.set_eval()

        obs_dict = env.reset()
        raw_env = env.unwrapped
        print("[INFO] Revolute replay is running. Close the viewer or press Ctrl+C to exit.")
        step = 0
        with torch.inference_mode():
            requested_steps = args_cli.video_length if args_cli.video else args_cli.max_steps
            while simulation_app.is_running() and (requested_steps is None or step < requested_steps):
                input_dict = {
                    "obs": agent.running_mean_std(obs_dict["obs"]),
                    "priv_info": obs_dict["priv_info"],
                }
                actions = torch.clamp(agent.model.act_inference(input_dict), -1.0, 1.0)
                obs_dict, _, _, _ = env.step(actions)
                step += 1

                if step % args_cli.print_interval == 0:
                    position_error = torch.norm(
                        raw_env.object_pos - raw_env.revolute_reference_pos, dim=-1
                    )
                    current_axis_w = quat_rotate(
                        raw_env.object_rot,
                        torch.tensor((0.0, 0.0, 1.0), device=raw_env.device).reshape(1, 3),
                    )
                    axis_alignment = torch.sum(current_axis_w * raw_env.revolute_axis_w, dim=-1)
                    axial_angvel = torch.sum(
                        raw_env.object_angvel * raw_env.revolute_axis_w, dim=-1
                    )
                    off_axis_angvel = torch.norm(
                        raw_env.object_angvel
                        - axial_angvel.unsqueeze(-1) * raw_env.revolute_axis_w,
                        dim=-1,
                    )
                    print(
                        f"[REVOLUTE][step={step}] axial_w={axial_angvel.item():+.4f} rad/s | "
                        f"off_axis_w={off_axis_angvel.item():.6f} rad/s | "
                        f"position_error={position_error.item():.6f} m | "
                        f"axis_alignment={axis_alignment.item():.6f}"
                    )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
