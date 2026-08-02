"""Blender-compatible visualization-only PLY export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openphenomena.data import Domain


def write_ply(domain: Domain, path: Path) -> Path:
    """Write geometry, normals, and tone-mapped color as ASCII PLY."""

    normals = np.asarray(
        domain.fields["geometry.normal.vertex"].values, dtype=np.float64
    )
    tone_mapped = np.asarray(
        domain.fields["color.tonemapped_srgb"].values, dtype=np.float64
    )
    colors = np.clip(255.0 * tone_mapped, 0.0, 255.0).astype(np.uint8)
    lines = [
        "ply",
        "format ascii 1.0",
        "comment Open Phenomena derived visualization-only artifact",
        f"element vertex {len(domain.positions_m)}",
        "property double x",
        "property double y",
        "property double z",
        "property double nx",
        "property double ny",
        "property double nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {len(domain.faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for position, normal, color in zip(
        domain.positions_m, normals, colors, strict=True
    ):
        values = (*position, *normal, *color)
        lines.append(" ".join(str(item) for item in values))
    lines.extend(
        f"3 {int(face[0])} {int(face[1])} {int(face[2])}" for face in domain.faces
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
