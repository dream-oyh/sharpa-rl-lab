"""Memmap schema and offline analysis for fixed-scale bulb evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


MEMMAP_FILENAME = "bulb_eval.memmap"
METADATA_FILENAME = "meta.json"
MEMMAP_DTYPE = np.dtype(
    [
        ("valid", np.uint8),
        ("object_pos", np.float32, (3,)),
        ("object_quat_wxyz", np.float32, (4,)),
        ("target_quat_wxyz", np.float32, (4,)),
        ("object_linvel_w", np.float32, (3,)),
        ("object_angvel_w", np.float32, (3,)),
        ("object_up_w", np.float32, (3,)),
        ("target_up_w", np.float32, (3,)),
        ("action", np.float32, (22,)),
        ("reward", np.float32),
        ("terminated", np.uint8),
        ("truncated", np.uint8),
        ("episode_step", np.int16),
    ]
)


def create_memmap(output_dir: str | Path, shape: tuple[int, int, int]) -> np.memmap:
    """Create a new structured memmap and initialize invalid floating values to NaN."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data = np.memmap(output_path / MEMMAP_FILENAME, dtype=MEMMAP_DTYPE, mode="w+", shape=shape)
    data["valid"] = 0
    data["terminated"] = 0
    data["truncated"] = 0
    data["episode_step"] = -1
    for name in MEMMAP_DTYPE.names:
        if np.issubdtype(MEMMAP_DTYPE[name].base, np.floating):
            data[name] = np.nan
    data.flush()
    return data


def write_metadata(output_dir: str | Path, metadata: dict) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / METADATA_FILENAME).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.write("\n")


def load_dataset(output_dir: str | Path) -> tuple[np.memmap, dict]:
    output_path = Path(output_dir)
    with (output_path / METADATA_FILENAME).open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    shape = tuple(int(value) for value in metadata["shape"])
    data = np.memmap(output_path / MEMMAP_FILENAME, dtype=MEMMAP_DTYPE, mode="r", shape=shape)
    return data, metadata


def _quat_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def relative_euler_xyz_deg(current_wxyz: np.ndarray, target_wxyz: np.ndarray) -> np.ndarray:
    """Return wrapped XYZ roll/pitch/yaw of target^-1 * current, in degrees."""
    current_norm = np.linalg.norm(current_wxyz, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(target_wxyz, axis=-1, keepdims=True)
    current = current_wxyz / np.clip(current_norm, 1.0e-8, None)
    target = target_wxyz / np.clip(target_norm, 1.0e-8, None)
    target_inverse = target.copy()
    target_inverse[..., 1:] *= -1.0
    relative = _quat_multiply_wxyz(target_inverse, current)
    w, x, y, z = np.moveaxis(relative, -1, 0)

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.rad2deg(np.stack((roll, pitch, yaw), axis=-1))


def _masked(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.where(valid, values, np.nan)


def _time_mean(values: np.ndarray) -> np.ndarray:
    counts = np.sum(np.isfinite(values), axis=0)
    totals = np.nansum(values, axis=0)
    return np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)


def _finite_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {f"{prefix}_mean": float("nan"), f"{prefix}_min": float("nan"), f"{prefix}_max": float("nan")}
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def analyze_dataset(output_dir: str | Path, curve_count: int = 50) -> list[dict[str, float | int]]:
    """Generate one two-panel trajectory figure per scale and bin-wise summary tables."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    figures_path = output_path / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)
    data, metadata = load_dataset(output_path)
    scales = [float(value) for value in metadata["scales"]]
    num_scales, num_envs, episode_steps = data.shape
    if num_scales != len(scales):
        raise ValueError(f"Scale metadata has {len(scales)} values, but memmap has {num_scales} bins.")

    valid = data["valid"].astype(bool)
    up_projection = np.sum(data["object_up_w"] * data["target_up_w"], axis=-1)
    up_projection = _masked(up_projection, valid)
    omega_z = _masked(data["object_angvel_w"][..., 2], valid)
    euler_deg = relative_euler_xyz_deg(data["object_quat_wxyz"], data["target_quat_wxyz"])
    euler_deg = np.where(valid[..., None], euler_deg, np.nan)
    time_s = np.arange(episode_steps, dtype=np.float64) * float(metadata["step_dt_s"])
    selected_envs = np.linspace(0, num_envs - 1, min(curve_count, num_envs), dtype=np.int64)

    rows: list[dict[str, float | int]] = []
    for scale_index, scale in enumerate(scales):
        bin_valid = valid[scale_index]
        bin_up = up_projection[scale_index]
        bin_omega_z = omega_z[scale_index]
        bin_euler = euler_deg[scale_index]
        terminated = data["terminated"][scale_index].astype(bool) & bin_valid
        truncated = data["truncated"][scale_index].astype(bool) & bin_valid
        success = np.any(truncated, axis=-1) & ~np.any(terminated, axis=-1)
        episode_lengths = np.sum(bin_valid, axis=-1)

        row: dict[str, float | int] = {
            "scale": scale,
            "num_envs": num_envs,
            "success_count": int(np.sum(success)),
            "success_rate": float(np.mean(success)),
            "episode_steps_mean": float(np.mean(episode_lengths)),
        }
        row.update(_finite_stats(bin_up, "up_projection"))
        row.update(_finite_stats(bin_omega_z, "omega_z_rad_s"))
        finite_abs_omega = np.abs(bin_omega_z[np.isfinite(bin_omega_z)])
        row["omega_z_abs_mean_rad_s"] = float(np.mean(finite_abs_omega)) if finite_abs_omega.size else float("nan")
        for axis_index, axis_name in enumerate(("roll", "pitch", "yaw")):
            row.update(_finite_stats(bin_euler[..., axis_index], f"relative_{axis_name}_deg"))
        rows.append(row)

        figure, axis = plt.subplots(figsize=(11.0, 4.5), constrained_layout=True)
        for env_index in selected_envs:
            axis.plot(time_s, bin_up[env_index], color="tab:blue", alpha=0.18, linewidth=0.8)
        axis.plot(time_s, _time_mean(bin_up), color="navy", linewidth=2.2, label=f"mean of {num_envs} envs")
        axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        axis.set_title(f"Bulb scale {scale:.1f}: up projection ({len(selected_envs)} trajectories)")
        axis.set_xlabel("time (s)"); axis.set_ylabel("up · target_up")
        axis.grid(True, alpha=0.25); axis.legend(loc="best")
        figure.savefig(figures_path / f"scale_{scale:.1f}_up_projection.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(11.0, 4.5), constrained_layout=True)
        for env_index in selected_envs:
            axis.plot(time_s, bin_omega_z[env_index], color="tab:orange", alpha=0.18, linewidth=0.8)
        axis.plot(time_s, _time_mean(bin_omega_z), color="darkred", linewidth=2.2, label=f"mean of {num_envs} envs")
        axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        axis.set_title(f"Bulb scale {scale:.1f}: world-Z angular velocity ({len(selected_envs)} trajectories)")
        axis.set_xlabel("time (s)"); axis.set_ylabel("angular velocity (rad/s)")
        axis.grid(True, alpha=0.25); axis.legend(loc="best")
        figure.savefig(figures_path / f"scale_{scale:.1f}_angular_velocity_z.png", dpi=180)
        plt.close(figure)

    fieldnames = list(rows[0].keys())
    with (output_path / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (output_path / "summary.md").open("w", encoding="utf-8") as file:
        file.write("| " + " | ".join(fieldnames) + " |\n")
        file.write("| " + " | ".join("---" for _ in fieldnames) + " |\n")
        for row in rows:
            values = []
            for name in fieldnames:
                value = row[name]
                values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
            file.write("| " + " | ".join(values) + " |\n")

    summary = {
        "rows": rows,
        "definitions": {
            "success": "First episode reaches the environment time limit without an earlier termination.",
            "up_projection": "Dot product of the unit bulb up vector and unit target up vector.",
            "omega_z_rad_s": "World-frame Z component of rigid-body angular velocity.",
            "relative_euler_deg": "Wrapped XYZ roll/pitch/yaw of target_quaternion^-1 * object_quaternion.",
            "aggregate_statistics": "Mean/min/max over all valid environment-time samples in each scale bin.",
        },
        "curve_env_indices": selected_envs.tolist(),
    }
    with (output_path / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False, allow_nan=True)
        file.write("\n")
    return rows
