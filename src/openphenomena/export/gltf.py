"""Minimal embedded glTF 2.0 visualization-only export."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from openphenomena.data import Domain


def write_gltf(domain: Domain, path: Path) -> Path:
    """Write positions, normals, tone-mapped vertex colors, and triangles."""

    positions = np.asarray(domain.positions_m, dtype="<f4")
    normals = np.asarray(domain.fields["geometry.normal.vertex"].values, dtype="<f4")
    colors = np.asarray(domain.fields["color.tonemapped_srgb"].values, dtype="<f4")
    indices = np.asarray(domain.faces, dtype="<u4").ravel()
    chunks = [
        positions.tobytes(),
        normals.tobytes(),
        colors.tobytes(),
        indices.tobytes(),
    ]
    offsets: list[int] = []
    payload = bytearray()
    for chunk in chunks:
        while len(payload) % 4:
            payload.append(0)
        offsets.append(len(payload))
        payload.extend(chunk)

    position_min = np.min(positions, axis=0).astype(float).tolist()
    position_max = np.max(positions, axis=0).astype(float).tolist()
    view_lengths = [len(chunk) for chunk in chunks]
    document: dict[str, object] = {
        "asset": {
            "version": "2.0",
            "generator": "Open Phenomena",
            "extras": {"classification": "VO", "sourceDomain": domain.domain_id},
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Static spherical bubble (VO)"}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
                        "indices": 3,
                        "mode": 4,
                    }
                ]
            }
        ],
        "buffers": [
            {
                "byteLength": len(payload),
                "uri": "data:application/octet-stream;base64,"
                + base64.b64encode(payload).decode("ascii"),
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[index], "byteLength": length}
            for index, length in enumerate(view_lengths)
        ],
        "accessors": [
            _accessor(0, 5126, len(positions), "VEC3", position_min, position_max),
            _accessor(1, 5126, len(normals), "VEC3"),
            _accessor(2, 5126, len(colors), "VEC3"),
            _accessor(3, 5125, len(indices), "SCALAR"),
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _accessor(
    view: int,
    component_type: int,
    count: int,
    value_type: str,
    minimum: list[float] | None = None,
    maximum: list[float] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "bufferView": view,
        "componentType": component_type,
        "count": count,
        "type": value_type,
    }
    if minimum is not None:
        result["min"] = minimum
    if maximum is not None:
        result["max"] = maximum
    return result
