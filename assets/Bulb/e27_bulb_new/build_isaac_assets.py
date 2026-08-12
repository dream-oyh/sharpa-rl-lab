#!/usr/bin/env python3
"""Build Isaac Sim USDA assets from the prepared physical E27 mesh."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh


# The RL task's historical bulb root lies 59 mm above the screw-tip plane.
# Keep the FoundationPose PLY unchanged and apply this offset only to Isaac
# assets so the existing hand pose, reward point, and socket anchor stay valid.
ISAAC_ORIGIN_OFFSET_Z = -0.059


def _format_vec3(values: np.ndarray) -> str:
    return ",\n            ".join(
        f"({value[0]:.9g}, {value[1]:.9g}, {value[2]:.9g})" for value in values
    )


def _format_ints(values: np.ndarray, width: int = 24) -> str:
    flat = values.reshape(-1).tolist()
    return ",\n            ".join(
        ", ".join(str(value) for value in flat[index : index + width])
        for index in range(0, len(flat), width)
    )


def _mesh_block(
    name: str,
    mesh: trimesh.Trimesh,
    *,
    collision: bool = False,
    colors: np.ndarray | None = None,
) -> str:
    schemas = ""
    physics = ""
    visibility = ""
    if collision:
        schemas = ' (\n        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]\n    )'
        physics = (
            '\n        bool physics:collisionEnabled = 1'
            '\n        uniform token physics:approximation = "convexHull"'
        )
        visibility = '\n        uniform token visibility = "invisible"'

    color_text = ""
    if colors is not None:
        rgb = colors[:, :3].astype(np.float64) / 255.0
        opacity = colors[:, 3].astype(np.float64) / 255.0
        color_text = (
            "\n        color3f[] primvars:displayColor = [\n            "
            + _format_vec3(rgb)
            + '\n        ] (\n            interpolation = "vertex"\n        )'
            + "\n        float[] primvars:displayOpacity = [\n            "
            + ", ".join(f"{value:.6g}" for value in opacity)
            + '\n        ] (\n            interpolation = "vertex"\n        )'
        )

    return f'''    def Mesh "{name}"{schemas}
    {{
        uniform bool doubleSided = 0
        uniform token subdivisionScheme = "none"{visibility}{physics}
        point3f[] points = [
            {_format_vec3(np.asarray(mesh.vertices))}
        ]
        int[] faceVertexCounts = [
            {_format_ints(np.full(len(mesh.faces), 3, dtype=np.int32))}
        ]
        int[] faceVertexIndices = [
            {_format_ints(np.asarray(mesh.faces, dtype=np.int32))}
        ]{color_text}
    }}
'''


def build_bulb(mesh: trimesh.Trimesh, output: Path) -> None:
    mesh = mesh.copy()
    mesh.apply_translation((0.0, 0.0, ISAAC_ORIGIN_OFFSET_Z))
    colors = np.asarray(mesh.visual.vertex_colors)
    components = sorted(mesh.split(only_watertight=False), key=lambda part: len(part.faces), reverse=True)
    collision_components = [
        part
        for part in components
        if len(part.faces) >= 32 and float(np.min(part.extents)) > 1.0e-4
    ]

    blocks = [_mesh_block("visual", mesh, colors=colors)]
    blocks.extend(
        _mesh_block(f"collision_{index}", part, collision=True)
        for index, part in enumerate(collision_components)
    )
    text = '''#usda 1.0
(
    defaultPrim = "E27_Bulb"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "E27_Bulb" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    bool physics:kinematicEnabled = 0
    bool physics:rigidBodyEnabled = 1
    float physics:mass = 0.0336

''' + "\n".join(blocks) + "}\n"
    output.write_text(text)
    print(f"saved bulb asset: {output}")
    print(f"collision convex hulls: {len(collision_components)}")


def _guide_segment(index: int, segments: int, inner_radius: float, outer_radius: float, z_min: float, z_max: float) -> str:
    angle_0 = 2.0 * math.pi * index / segments
    angle_1 = 2.0 * math.pi * (index + 1) / segments
    # Set the inner polygon's apothem to the requested effective radius.
    inner_vertex_radius = inner_radius / math.cos(math.pi / segments)

    def xy(radius: float, angle: float, z: float) -> tuple[float, float, float]:
        return radius * math.cos(angle), radius * math.sin(angle), z

    points = np.asarray(
        [
            xy(inner_vertex_radius, angle_0, z_min),
            xy(outer_radius, angle_0, z_min),
            xy(inner_vertex_radius, angle_1, z_min),
            xy(outer_radius, angle_1, z_min),
            xy(inner_vertex_radius, angle_0, z_max),
            xy(outer_radius, angle_0, z_max),
            xy(inner_vertex_radius, angle_1, z_max),
            xy(outer_radius, angle_1, z_max),
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 4, 6, 2],
            [1, 3, 7, 5],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
        ],
        dtype=np.int32,
    )
    return f'''        def Mesh "segment_{index:02d}" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {{
            uniform bool doubleSided = 0
            uniform token subdivisionScheme = "none"
            bool physics:collisionEnabled = 1
            uniform token physics:approximation = "convexHull"
            point3f[] points = [
                {_format_vec3(points)}
            ]
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [
                {_format_ints(faces)}
            ]
        }}
'''


def build_guide(output: Path) -> None:
    segments = 16
    inner_radius = 0.0135
    outer_radius = 0.018
    z_min = 0.006 + ISAAC_ORIGIN_OFFSET_Z
    z_max = 0.024 + ISAAC_ORIGIN_OFFSET_Z
    segment_blocks = "\n".join(
        _guide_segment(index, segments, inner_radius, outer_radius, z_min, z_max)
        for index in range(segments)
    )
    text = f'''#usda 1.0
(
    defaultPrim = "E27_Socket_Guide"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "E27_Socket_Guide" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    bool physics:kinematicEnabled = 1
    bool physics:rigidBodyEnabled = 1
    float physics:mass = 1

    def Xform "guide_collision" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        rel material:binding = </E27_Socket_Guide/Looks/guide_material>
        rel material:binding:physics = </E27_Socket_Guide/Looks/guide_physics_material> (
            bindMaterialAs = "strongerThanDescendants"
        )
{segment_blocks}
    }}

    def Scope "Looks"
    {{
        def Material "guide_material"
        {{
            token outputs:surface.connect = </E27_Socket_Guide/Looks/guide_material/PreviewSurface.outputs:surface>

            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.18, 0.22, 0.28)
                float inputs:metallic = 0.8
                float inputs:roughness = 0.35
                token outputs:surface
            }}
        }}

        def Material "guide_physics_material" (
            prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
        )
        {{
            float physics:staticFriction = 0.1
            float physics:dynamicFriction = 0.08
            float physics:restitution = 0
            uniform token physxMaterial:frictionCombineMode = "min"
            uniform token physxMaterial:restitutionCombineMode = "min"
        }}
    }}
}}
'''
    output.write_text(text)
    print(f"saved guide asset: {output}")
    print(
        "guide: "
        f"ID={2 * inner_radius * 1000:.2f} mm, "
        f"OD={2 * outer_radius * 1000:.2f} mm, "
        f"height={(z_max - z_min) * 1000:.2f} mm, segments={segments}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=Path(__file__).with_name("e27_bulb_60x109mm.ply"))
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(mesh).__name__}")
    output_dir = args.mesh.parent
    build_bulb(mesh, output_dir / "E27_bulb_rigid.usda")
    build_guide(output_dir / "E27_socket_guide.usda")


if __name__ == "__main__":
    main()
