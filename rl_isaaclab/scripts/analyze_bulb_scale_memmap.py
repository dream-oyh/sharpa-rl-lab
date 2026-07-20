#!/usr/bin/env python3
"""Analyze an existing fixed-scale bulb evaluation memmap."""

from __future__ import annotations

import argparse

from rl_isaaclab.utils.bulb_scale_eval import analyze_dataset


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output_dir", help="Directory containing bulb_eval.memmap and meta.json.")
parser.add_argument("--curve_count", type=int, default=50, help="Number of per-environment trajectories per figure.")
args = parser.parse_args()

if args.curve_count <= 0:
    raise ValueError(f"--curve_count must be positive, got {args.curve_count}.")

rows = analyze_dataset(args.output_dir, curve_count=args.curve_count)
for row in rows:
    print(
        f"scale={row['scale']:.1f} success={row['success_count']}/{row['num_envs']} "
        f"({100.0 * row['success_rate']:.1f}%) omega_z_mean={row['omega_z_rad_s_mean']:.4f} rad/s"
    )
