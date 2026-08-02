# Static spherical-bubble export guide

`scientific/manifest.json` and `scientific/arrays.npz` together form the
authoritative source. The manifest describes the immutable study/run/frame/
domain/field/evidence hierarchy.

`exports/bubble_finest.vtp` is ASCII VTK XML PolyData containing geometry,
topology, point fields, cell fields, and global fields. See the
[official VTK XML specification](https://docs.vtk.org/en/v9.6.1/vtk_file_formats/vtkxml_file_format.html).

VTKHDF is intentionally not emitted because neither VTK nor `h5py` was available
in the baseline environment. VTP is stable for this small serial dataset;
VTKHDF should be added only with a tested ParaView round trip.

`bubble_blender_vo.ply` and `bubble_vo.gltf` contain derived geometry, normals,
and display colors. Both are **VO** and cannot restart or validate the study.
`visualization_manifest.json` identifies their classifications and source.
