"""Run a small physics smoke test for the free bulb and socket guide ring."""

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=5)
parser.add_argument("--physics_steps", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import carb
import omni.usd
import torch

import rl_isaaclab.tasks.inhand_rotate_bulb  # noqa: F401,E402
from rl_isaaclab.tasks.inhand_rotate_bulb.sharpa_wave_bulb_ring_env_cfg import (  # noqa: E402
    SharpaWaveBulbRingEnvCfg,
)


TASK_NAME = "Isaac-Inhand-Rotate-Sharpa-Wave-Bulb-Ring-v0"


def main():
    if args_cli.num_envs < 1:
        raise ValueError("--num_envs must be positive.")
    if args_cli.physics_steps < 1:
        raise ValueError("--physics_steps must be positive.")

    cfg = SharpaWaveBulbRingEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.debug_show_axes = False
    cfg.debug_show_object_pos = False
    cfg.debug_show_object_vectors = False

    env = None
    try:
        print("[SMOKE] Creating ring environment...", flush=True)
        env = gym.make(TASK_NAME, cfg=cfg, render_mode=None)
        raw_env = env.unwrapped
        print("[SMOKE] Resetting ring environment...", flush=True)
        env.reset()
        print("[SMOKE] Checking stage and reward...", flush=True)

        stage = omni.usd.get_context().get_stage()
        joint_prim = stage.GetPrimAtPath(
            "/World/envs/env_0/bulb_socket_revolute_joint"
        )
        guide_prim = stage.GetPrimAtPath(
            "/World/envs/env_0/socket_anchor/guide_collision"
        )
        if joint_prim.IsValid():
            raise AssertionError("Ring task unexpectedly contains a revolute joint.")
        if not guide_prim.IsValid():
            raise AssertionError("Socket guide collision mesh was not spawned.")

        actions = torch.zeros(
            (raw_env.num_envs, raw_env.cfg.action_space), device=raw_env.device
        )
        _, reward, terminated, truncated, _ = env.step(actions)
        if not torch.isfinite(reward).all():
            raise AssertionError("Non-finite reward produced by the ring task.")
        required_terms = {
            "rotate_reward",
            "object_linvel_penalty",
            "pos_diff_penalty",
            "torque_penalty",
            "work_penalty",
            "object_pos_diff",
            "object_tip_z_penalty",
            "object_up_alignment_reward",
        }
        if not required_terms.issubset(raw_env._step_reward_terms):
            missing = required_terms.difference(raw_env._step_reward_terms)
            raise AssertionError(f"Missing in-hand reward terms: {sorted(missing)}")

        if not raw_env.cfg.gravity_curriculum:
            raise AssertionError("Gravity curriculum is disabled in the ring task.")
        initial_gravity_z = float(raw_env.physics_sim_view.get_gravity()[2])
        if abs(initial_gravity_z + 0.05) > 1.0e-6:
            raise AssertionError(
                f"Unexpected curriculum start gravity: {initial_gravity_z} m/s^2"
            )

        # Exercise the same curriculum branch used by training. With a stable
        # reset state and a step count above the warm-up, gravity should advance
        # by one 0.05 m/s^2 increment.
        saved_step_counter = raw_env.common_step_counter
        raw_env.common_step_counter = 1001
        raw_env._get_dones()
        ramped_gravity_z = float(raw_env.physics_sim_view.get_gravity()[2])
        if abs(ramped_gravity_z + 0.10) > 1.0e-6:
            raise AssertionError(
                "Gravity curriculum did not advance by one increment: "
                f"{initial_gravity_z} -> {ramped_gravity_z} m/s^2"
            )
        raw_env.common_step_counter = saved_step_counter
        raw_env.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -0.05))

        # Teleport the bulb away from both the fixed-base hand and the guide,
        # then advance raw physics without the RL auto-reset path. A genuinely
        # free bulb must accelerate downward.
        object_state_w = raw_env.object.data.root_state_w.clone()
        object_state_w[:, 0] += 0.30
        object_state_w[:, 7:] = 0.0
        raw_env.object.write_root_state_to_sim(object_state_w)
        raw_env.scene.update(dt=raw_env.physics_dt)
        initial_z = raw_env.object.data.root_pos_w[:, 2].clone()
        for _ in range(args_cli.physics_steps):
            raw_env.sim.step()
            raw_env.scene.update(dt=raw_env.physics_dt)
        final_z = raw_env.object.data.root_pos_w[:, 2].clone()
        drop = initial_z - final_z
        if not torch.all(drop > 1.0e-6):
            raise AssertionError(
                f"Bulb did not fall under gravity in every environment: {drop.tolist()}"
            )

        # Move the bulb and kinematic guide together away from the hand, then
        # launch the bulb sideways. The thread colliders should hit the guide's
        # inner wall instead of crossing it.
        env.reset()
        ring_state_w = raw_env.socket_anchor.data.root_state_w.clone()
        ring_state_w[:, 0] += 0.30
        ring_state_w[:, 7:] = 0.0
        raw_env.socket_anchor.write_root_state_to_sim(ring_state_w)
        object_state_w = raw_env.object.data.root_state_w.clone()
        object_state_w[:, 0] += 0.30
        object_state_w[:, 7:] = 0.0
        object_state_w[:, 7] = 0.5
        raw_env.object.write_root_state_to_sim(object_state_w)
        raw_env.scene.update(dt=raw_env.physics_dt)
        for _ in range(5):
            raw_env.sim.step()
            raw_env.scene.update(dt=raw_env.physics_dt)
        relative_xy = (
            raw_env.object.data.root_pos_w[:, :2]
            - raw_env.socket_anchor.data.root_pos_w[:, :2]
        )
        radial_after_impact = torch.norm(relative_xy, dim=-1)
        if not torch.all(radial_after_impact < 0.006):
            raise AssertionError(
                "Bulb crossed the guide wall during lateral impact: "
                f"{radial_after_impact.tolist()}"
            )

        object_scales = []
        ring_scales = []
        for env_id in range(raw_env.num_envs):
            object_scales.append(
                tuple(
                    stage.GetPrimAtPath(f"/World/envs/env_{env_id}/object")
                    .GetAttribute("xformOp:scale")
                    .Get()
                )
            )
            ring_scales.append(
                tuple(
                    stage.GetPrimAtPath(f"/World/envs/env_{env_id}/socket_anchor")
                    .GetAttribute("xformOp:scale")
                    .Get()
                )
            )
        if object_scales != ring_scales:
            raise AssertionError(
                f"Bulb/ring scale mismatch: {object_scales} != {ring_scales}"
            )

        print(
            "[PASS] ring smoke test | "
            f"envs={raw_env.num_envs} | "
            f"reward_mean={reward.mean().item():+.5f} | "
            f"terminated={terminated.sum().item()} | "
            f"truncated={truncated.sum().item()} | "
            "gravity_ramp=("
            f"{initial_gravity_z:.2f} -> {ramped_gravity_z:.2f}) m/s^2 | "
            f"drop_range=({drop.min().item():.6f}, {drop.max().item():.6f}) m | "
            "impact_radial_range=("
            f"{radial_after_impact.min().item():.6f}, "
            f"{radial_after_impact.max().item():.6f}) m | "
            f"scales={object_scales}",
            flush=True,
        )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
