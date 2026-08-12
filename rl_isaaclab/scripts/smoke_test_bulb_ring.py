"""Run a small physics smoke test for the free bulb and socket guide ring."""

import argparse
import math
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
from isaaclab.utils.math import quat_apply

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

        object_root_path = "/World/envs/env_0/object"
        visual_prim = stage.GetPrimAtPath(f"{object_root_path}/visual")
        if not visual_prim.IsValid():
            raise AssertionError("New physical E27 visual mesh was not spawned.")
        collision_prims = [
            stage.GetPrimAtPath(f"{object_root_path}/collision_{index}")
            for index in range(5)
        ]
        if not all(prim.IsValid() for prim in collision_prims):
            raise AssertionError("Expected five physical E27 convex colliders.")
        authored_mass = float(
            stage.GetPrimAtPath(object_root_path)
            .GetAttribute("physics:mass")
            .Get()
        )
        if abs(authored_mass - 0.0336) > 1.0e-7:
            raise AssertionError(
                f"Unexpected authored bulb mass: {authored_mass} kg"
            )
        sampled_masses = raw_env.object.root_physx_view.get_masses().reshape(-1)
        if raw_env.cfg.randomize_mass:
            raise AssertionError("Ring task unexpectedly enables mass DR.")
        if not torch.allclose(
            sampled_masses,
            torch.full_like(sampled_masses, 0.0336),
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise AssertionError(
                f"Bulb mass is not fixed at 33.6 g: {sampled_masses.tolist()}"
            )

        thread_specs = (
            ("thread_ring_0", -0.051),
            ("thread_ring_1", -0.044),
            ("thread_ring_2", -0.037),
        )
        for name, expected_z in thread_specs:
            thread_prim = stage.GetPrimAtPath(
                f"{object_root_path}/thread_collision/{name}"
            )
            if not thread_prim.IsValid():
                raise AssertionError(f"Missing synthetic thread collider: {name}")
            radius = float(thread_prim.GetAttribute("radius").Get())
            height = float(thread_prim.GetAttribute("height").Get())
            translate = thread_prim.GetAttribute("xformOp:translate").Get()
            if abs(radius - 0.0125) > 1.0e-8 or abs(height - 0.002) > 1.0e-8:
                raise AssertionError(
                    f"Unexpected {name} dimensions: radius={radius}, height={height}"
                )
            if translate is None or abs(float(translate[2]) - expected_z) > 1.0e-8:
                raise AssertionError(f"Unexpected {name} Z placement: {translate}")

        guide_segments = [
            stage.GetPrimAtPath(
                f"/World/envs/env_0/socket_anchor/guide_collision/segment_{index:02d}"
            )
            for index in range(16)
        ]
        if not all(prim.IsValid() for prim in guide_segments):
            raise AssertionError("Expected all 16 socket-guide collision segments.")
        guide_points = [
            point
            for prim in guide_segments
            for point in prim.GetAttribute("points").Get()
        ]
        guide_radii = [math.hypot(float(point[0]), float(point[1])) for point in guide_points]
        guide_z = [float(point[2]) for point in guide_points]
        guide_inner_apothem = min(guide_radii) * math.cos(math.pi / 16)
        guide_outer_radius = max(guide_radii)
        guide_height = max(guide_z) - min(guide_z)
        if (
            abs(guide_inner_apothem - 0.0135) > 1.0e-7
            or abs(guide_outer_radius - 0.018) > 1.0e-7
            or abs(guide_height - 0.018) > 1.0e-7
        ):
            raise AssertionError(
                "Unexpected socket-guide dimensions: "
                f"ID={2 * guide_inner_apothem}, OD={2 * guide_outer_radius}, "
                f"height={guide_height}"
            )

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
        if not raw_env.cfg.pos_diff_reference_reset_pose:
            raise AssertionError("Ring task is not using its reset grasp as position reference.")
        expected_pos_diff_penalty = (
            (
                raw_env.hand_dof_pos[:, raw_env.actuated_dof_indices]
                - raw_env.reset_hand_dof_pos[:, raw_env.actuated_dof_indices]
            )
            ** 2
        ).sum(-1) * raw_env.cfg.pos_diff_penalty_scale
        if not torch.allclose(
            raw_env._step_reward_terms["pos_diff_penalty"],
            expected_pos_diff_penalty,
            atol=1.0e-6,
            rtol=1.0e-5,
        ):
            raise AssertionError(
                "Position penalty is not referenced to the cache-sampled reset grasp."
            )

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
        if not torch.all(radial_after_impact < 0.008):
            raise AssertionError(
                "Bulb root moved implausibly far during lateral impact: "
                f"{radial_after_impact.tolist()}"
            )

        # The rigid-body root lies 37--51 mm above the three thread rings, so
        # root XY motion is amplified when the bulb tilts. Check collision at
        # the actual thread-ring centers instead of treating root XY as wall
        # penetration.
        scale_values = torch.linspace(
            raw_env.cfg.scale_range[0],
            raw_env.cfg.scale_range[1],
            raw_env.cfg.scale_range[2],
            device=raw_env.device,
        )
        env_scales = scale_values[
            raw_env.scale_ids.squeeze(-1).to(torch.long)
        ]
        local_thread_centers = torch.zeros(
            (raw_env.num_envs, 3, 3), device=raw_env.device
        )
        local_thread_centers[:, :, 2] = (
            torch.tensor([-0.051, -0.044, -0.037], device=raw_env.device)
            * env_scales.unsqueeze(-1)
        )
        object_quat = raw_env.object.data.root_quat_w[:, None, :].expand(-1, 3, -1)
        thread_centers_w = raw_env.object.data.root_pos_w[:, None, :] + quat_apply(
            object_quat.reshape(-1, 4), local_thread_centers.reshape(-1, 3)
        ).reshape(raw_env.num_envs, 3, 3)
        thread_relative_xy = (
            thread_centers_w[:, :, :2]
            - raw_env.socket_anchor.data.root_pos_w[:, None, :2]
        )
        thread_radial_after_impact = torch.norm(thread_relative_xy, dim=-1)
        if not torch.all(thread_radial_after_impact < 0.004):
            raise AssertionError(
                "Thread colliders crossed the guide wall during lateral impact: "
                f"{thread_radial_after_impact.tolist()}"
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
            "guide=(ID 0.027, OD 0.036, H 0.018) m | "
            "thread=(D 0.025, H 0.002) m | "
            f"mass={sampled_masses.mean().item() * 1000:.1f} g | "
            f"drop_range=({drop.min().item():.6f}, {drop.max().item():.6f}) m | "
            "impact_radial_range=("
            f"{radial_after_impact.min().item():.6f}, "
            f"{radial_after_impact.max().item():.6f}) m | "
            "thread_radial_range=("
            f"{thread_radial_after_impact.min().item():.6f}, "
            f"{thread_radial_after_impact.max().item():.6f}) m | "
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
