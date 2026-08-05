#!/usr/bin/env python3
"""Prepare the downloaded Sketchfab E27 bulb for FoundationPose."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("input", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument("--diameter", type=float, default=0.060, help="Target diameter in metres")
  parser.add_argument("--height", type=float, default=0.109, help="Target height in metres")
  args = parser.parse_args()

  loaded = trimesh.load(args.input, force="scene")
  # The downloaded GLB uses solid glTF materials. Convert them to vertex
  # colours so FoundationPose's Trimesh/NVDiffRast path can consume the merged
  # mesh without a texture atlas.
  for name, geometry in loaded.geometry.items():
    if "Inox" in name:
      rgba = np.array([158, 158, 158, 255], dtype=np.uint8)
    elif "Led" in name:
      rgba = np.array([238, 238, 238, 255], dtype=np.uint8)
    else:
      rgba = np.array([248, 248, 248, 255], dtype=np.uint8)
    geometry.visual = trimesh.visual.ColorVisuals(
      mesh=geometry,
      vertex_colors=np.tile(rgba, (len(geometry.vertices), 1)),
    )
  mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded.copy()
  if not isinstance(mesh, trimesh.Trimesh):
    raise TypeError(f"Expected Trimesh after scene merge, got {type(mesh).__name__}")

  # The Sketchfab scene uses Y as the bulb axis. FoundationPose receives a
  # right-handed model with Z along the bulb axis and the screw-tip centre at
  # the origin. X/Y are scaled independently by less than 1% to match the
  # product-sheet diameter exactly; Z is scaled to the specified total height.
  vertices = np.asarray(mesh.vertices).copy()
  bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
  centre_x = bounds[:, 0].mean()
  centre_z = bounds[:, 2].mean()

  prepared = np.empty_like(vertices)
  prepared[:, 0] = (vertices[:, 0] - centre_x) * args.diameter / np.ptp(vertices[:, 0])
  prepared[:, 1] = -(vertices[:, 2] - centre_z) * args.diameter / np.ptp(vertices[:, 2])
  prepared[:, 2] = (vertices[:, 1] - bounds[0, 1]) * args.height / np.ptp(vertices[:, 1])
  mesh.vertices = prepared
  mesh.remove_unreferenced_vertices()

  args.output.parent.mkdir(parents=True, exist_ok=True)
  mesh.export(args.output)

  check = trimesh.load(args.output, force="mesh", process=False)
  print(f"saved: {args.output}")
  print(f"vertices: {len(check.vertices)}, faces: {len(check.faces)}")
  print(f"bounds (m):\n{check.bounds}")
  print(f"extents (m): {check.extents}")
  print(f"visual: {type(check.visual).__name__}")


if __name__ == "__main__":
  main()
