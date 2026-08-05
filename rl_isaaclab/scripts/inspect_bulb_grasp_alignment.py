"""Inspect bulb/hand alignment and grasp success conditions after reset."""

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-Bulb-New-v0",
)
parser.add_argument("--num_envs", type=int, default=5)
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import omni.usd
import torch
from pxr import Usd, UsdGeom

import rl_isaaclab.tasks.inhand_rotate_bulb  # noqa: F401,E402


def _contact_metrics(raw_env):
    forces = torch.cat(
        [
            raw_env._contact_sensor[index]
            .data.force_matrix_w[:, 0, 0, :]
            .unsqueeze(1)
            for index in range(10)
        ],
        dim=1,
    )
    force_norm = torch.norm(forces, dim=-1)
    return force_norm, (force_norm > 0.5).sum(dim=-1)


def main():
    cfg_entry = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    module_name, class_name = cfg_entry.split(":")
    module = __import__(module_name, fromlist=[class_name])
    cfg = getattr(module, class_name)()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.debug_show_axes = False
    cfg.debug_show_object_pos = False

    env = None
    try:
        env = gym.make(args_cli.task, cfg=cfg, render_mode=None)
        raw_env = env.unwrapped
        env.reset()
        raw_env._refresh_lab()

        stage = omni.usd.get_context().get_stage()
        bbox = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        ).ComputeWorldBound(stage.GetPrimAtPath("/World/envs/env_0/object"))
        bbox_range = bbox.GetRange()
        fingertip_distance = torch.norm(
            raw_env.fingertip_pos - raw_env.object_pos.unsqueeze(1), dim=-1
        )
        force_norm, contact_count = _contact_metrics(raw_env)
        print(
            "[ALIGN] reset | "
            f"object_pos={raw_env.object_pos[0].tolist()} | "
            f"fingertips={raw_env.fingertip_pos[0].tolist()} | "
            f"distance={fingertip_distance[0].tolist()} | "
            f"force={force_norm[0].tolist()} | contacts={contact_count[0].item()} | "
            f"bbox_min={tuple(bbox_range.GetMin())} | "
            f"bbox_max={tuple(bbox_range.GetMax())}",
            flush=True,
        )

        actions = torch.zeros(
            (raw_env.num_envs, raw_env.cfg.action_space), device=raw_env.device
        )
        for step in range(args_cli.steps):
            _, _, terminated, truncated, _ = env.step(actions)
            raw_env._refresh_lab()
            force_norm, contact_count = _contact_metrics(raw_env)
            if step in {0, 4, 9, args_cli.steps - 1}:
                print(
                    f"[ALIGN] step={step + 1} | "
                    f"object_z=({raw_env.object_pos[:, 2].min().item():.6f}, "
                    f"{raw_env.object_pos[:, 2].max().item():.6f}) | "
                    f"max_force={force_norm.max().item():.4f} | "
                    f"contacts_ge3={(contact_count >= 3).sum().item()} | "
                    f"terminated={terminated.sum().item()} | "
                    f"truncated={truncated.sum().item()}",
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
