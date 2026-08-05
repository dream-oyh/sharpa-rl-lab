# E27 Bulb model source

- Model: **E27 Bulb**
- Author: **Axcellence**
- Source: https://sketchfab.com/3d-models/e27-bulb-d475160147c74f57b281e3136e047707
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- Sketchfab model UID: `d475160147c74f57b281e3136e047707`

The source model is marked as downloadable, but Sketchfab requires an
authenticated account for the download. The downloaded GLB is preserved as
`e27_bulb_source.glb`. Keep this attribution when the model or a derivative is
redistributed.

Before FoundationPose inference, convert the asset to a single `trimesh.Trimesh`
mesh and scale its external dimensions to the physical bulb measurements:

- maximum diameter: `0.060 m`
- total height: `0.109 m`

The FoundationPose-ready, vertex-coloured asset is
`e27_bulb_60x109mm.ply`. Its Z axis follows the bulb's symmetry axis, its origin
is at the centre of the screw-tip plane, and all dimensions are expressed in
metres.

It can be regenerated in an environment containing NumPy and Trimesh:

```bash
python prepare_mesh.py e27_bulb_source.glb e27_bulb_60x109mm.ply
```

The Isaac Sim assets are generated from the prepared PLY with:

```bash
python build_isaac_assets.py
```

`E27_bulb_rigid.usda` contains the physical visual mesh and five convex-hull
collision components and uses the measured 33.6 g bulb mass.
`E27_bulb_ring_collision.usda` adds three 25 mm thread contact rings.
`E27_socket_guide.usda` is a 16-segment guide with a 27 mm effective inner
diameter, 36 mm outer diameter, and 18 mm height.

The Isaac assets translate the prepared mesh by `-59 mm` along local Z. This
keeps the existing RL task root convention (`-59 ... +50 mm`) while the source
PLY remains in the FoundationPose screw-tip convention (`0 ... 109 mm`).
