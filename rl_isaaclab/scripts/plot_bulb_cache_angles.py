#!/usr/bin/env python3
"""Plot the bulb-axis angle distribution stored in a grasp cache."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


CACHE_STATE_DIM = 29
OBJECT_QUATERNION_SLICE = slice(25, 29)
DEFAULT_CACHE = Path(
    "cache/sharpa_bulb_new_grasp_high_1p0_angle25_0.7-1.1-5.npy"
)
SCALE_SUFFIX_RE = re.compile(
    r"_([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"-([-+]?(?:\d+(?:\.\d*)?|\.\d+))-(\d+)\.npy$"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read an (N, 29) Sharpa grasp cache and plot the angle between "
            "the bulb local axis and a world-frame target axis."
        )
    )
    parser.add_argument(
        "cache",
        nargs="?",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"Cache .npy file (default: {DEFAULT_CACHE}).",
    )
    parser.add_argument(
        "--local-axis",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="Bulb-local axis stored by the object quaternion (default: +Z).",
    )
    parser.add_argument(
        "--target-axis",
        nargs=3,
        type=float,
        default=(0.0, 0.0, -1.0),
        metavar=("X", "Y", "Z"),
        help="World target axis (default: -Z for the inverted bulb).",
    )
    parser.add_argument(
        "--undirected-axis",
        action="store_true",
        help="Treat parallel and anti-parallel axes as equivalent (angles 0-90 deg).",
    )
    parser.add_argument("--bins", type=int, default=36, help="Histogram bin count.")
    parser.add_argument(
        "--num-scale-buckets",
        type=int,
        default=None,
        help=(
            "Number of contiguous scale buckets. By default this is inferred "
            "from a filename ending in _LOW-HIGH-COUNT.npy."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: CACHE_STEM_up_angle_distribution.png).",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Saved figure DPI.")
    parser.add_argument("--show", action="store_true", help="Open an interactive plot window.")
    return parser.parse_args()


def _normalize_axis(values: tuple[float, float, float] | list[float], name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if not np.isfinite(axis).all() or norm < 1.0e-8:
        raise ValueError(f"{name} must be a finite, non-zero 3-vector, got {values}.")
    return axis / norm


def _quat_rotate_wxyz(quaternions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate vectors by normalized WXYZ quaternions without SciPy."""
    quaternions = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-8):
        bad_index = int(np.flatnonzero(norms.reshape(-1) < 1.0e-8)[0])
        raise ValueError(f"Cache row {bad_index} contains a zero-length quaternion.")
    quaternions = quaternions / norms
    quaternion_vector = quaternions[:, 1:4]
    vectors = np.broadcast_to(vectors, quaternion_vector.shape)
    twice_cross = 2.0 * np.cross(quaternion_vector, vectors)
    return (
        vectors
        + quaternions[:, :1] * twice_cross
        + np.cross(quaternion_vector, twice_cross)
    )


def bulb_axis_angles_deg(
    cache: np.ndarray,
    local_axis: np.ndarray,
    target_axis: np.ndarray,
    *,
    undirected_axis: bool = False,
) -> np.ndarray:
    """Return one target-axis error angle in degrees for every cache row."""
    object_axis_w = _quat_rotate_wxyz(
        cache[:, OBJECT_QUATERNION_SLICE], local_axis
    )
    cosine = np.clip(object_axis_w @ target_axis, -1.0, 1.0)
    if undirected_axis:
        cosine = np.abs(cosine)
    return np.rad2deg(np.arccos(cosine))


def _infer_scale_buckets(cache_path: Path, requested: int | None) -> tuple[int, list[str]]:
    match = SCALE_SUFFIX_RE.search(cache_path.name)
    inferred_count = int(match.group(3)) if match else 1
    count = requested if requested is not None else inferred_count
    if count <= 0:
        raise ValueError(f"--num-scale-buckets must be positive, got {count}.")

    if match and int(match.group(3)) == count:
        lower, upper = float(match.group(1)), float(match.group(2))
        labels = [f"scale {value:g}" for value in np.linspace(lower, upper, count)]
    else:
        labels = [f"bucket {index}" for index in range(count)]
    return count, labels


def _print_stats(label: str, values: np.ndarray) -> None:
    percentiles = np.percentile(values, (5, 25, 50, 75, 95))
    print(
        f"{label:>12}: n={len(values):6d} | min={values.min():7.3f} | "
        f"p05={percentiles[0]:7.3f} | p25={percentiles[1]:7.3f} | "
        f"median={percentiles[2]:7.3f} | mean={values.mean():7.3f} | "
        f"p75={percentiles[3]:7.3f} | p95={percentiles[4]:7.3f} | "
        f"max={values.max():7.3f} deg"
    )


def main() -> None:
    args = _parse_args()
    if args.bins <= 0:
        raise ValueError(f"--bins must be positive, got {args.bins}.")
    if args.dpi <= 0:
        raise ValueError(f"--dpi must be positive, got {args.dpi}.")

    cache_path = args.cache.expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cache does not exist: {cache_path}")
    cache = np.load(cache_path, allow_pickle=False)
    if cache.ndim != 2 or cache.shape[1] != CACHE_STATE_DIM:
        raise ValueError(f"Expected cache shape (N, {CACHE_STATE_DIM}), got {cache.shape}.")
    if len(cache) == 0 or not np.isfinite(cache).all():
        raise ValueError(f"Cache must be non-empty and finite: {cache_path}")

    local_axis = _normalize_axis(args.local_axis, "--local-axis")
    target_axis = _normalize_axis(args.target_axis, "--target-axis")
    angles = bulb_axis_angles_deg(
        cache,
        local_axis,
        target_axis,
        undirected_axis=args.undirected_axis,
    )

    bucket_count, bucket_labels = _infer_scale_buckets(
        cache_path, args.num_scale_buckets
    )
    if len(cache) % bucket_count != 0:
        raise ValueError(
            f"Cache rows ({len(cache)}) are not divisible by scale buckets ({bucket_count})."
        )
    bucket_angles = np.split(angles, bucket_count)

    print(f"cache: {cache_path}")
    print(f"shape: {cache.shape}, dtype: {cache.dtype}")
    print(f"local axis: {local_axis.tolist()}")
    print(f"target axis: {target_axis.tolist()}")
    print(f"undirected axis: {args.undirected_axis}")
    _print_stats("all", angles)
    for label, values in zip(bucket_labels, bucket_angles, strict=True):
        _print_stats(label, values)

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    angle_domain_max = 90.0 if args.undirected_axis else 180.0
    plot_max = min(
        angle_domain_max,
        max(5.0, np.ceil((angles.max() * 1.05) / 5.0) * 5.0),
    )
    bin_edges = np.linspace(0.0, plot_max, args.bins + 1)
    figure, (hist_axis, cdf_axis) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    hist_axis.hist(
        angles,
        bins=bin_edges,
        density=True,
        color="0.75",
        edgecolor="0.35",
        linewidth=0.5,
        label="all",
    )
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, bucket_count))
    for label, values, color in zip(
        bucket_labels, bucket_angles, colors, strict=True
    ):
        hist_axis.hist(
            values,
            bins=bin_edges,
            density=True,
            histtype="step",
            linewidth=1.3,
            color=color,
            label=label,
        )
        sorted_values = np.sort(values)
        cumulative = np.arange(1, len(values) + 1) / len(values)
        cdf_axis.plot(sorted_values, cumulative, color=color, label=label)

    axis_kind = "axis error" if args.undirected_axis else "directed-axis error"
    hist_axis.set_title(f"Bulb {axis_kind} distribution")
    hist_axis.set_xlabel("Angle to target (deg)")
    hist_axis.set_ylabel("Probability density")
    cdf_axis.set_title("Empirical cumulative distribution")
    cdf_axis.set_xlabel("Angle to target (deg)")
    cdf_axis.set_ylabel("Cumulative probability")
    for axis in (hist_axis, cdf_axis):
        axis.set_xlim(0.0, plot_max)
        axis.grid(alpha=0.25)
    cdf_axis.set_ylim(0.0, 1.0)
    hist_axis.legend(fontsize=8)
    cdf_axis.legend(fontsize=8)
    figure.suptitle(
        f"{cache_path.name}\nlocal {local_axis.tolist()} → world {target_axis.tolist()}",
        fontsize=10,
    )
    figure.tight_layout()

    output_path = (
        args.output.expanduser()
        if args.output is not None
        else cache_path.with_name(f"{cache_path.stem}_up_angle_distribution.png")
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"saved figure: {output_path}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
