"""Deterministic recursively subdivided icosphere meshes."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def create_icosphere(
    refinement_level: int,
    radius_m: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Return an outward-oriented projected icosphere.

    Every refinement splits one triangle into four and projects new edge
    midpoints to the analytical sphere. No randomness is used.
    """

    if refinement_level < 0:
        raise ValueError("refinement_level must be nonnegative")
    if radius_m <= 0.0 or not np.isfinite(radius_m):
        raise ValueError("radius_m must be finite and positive")

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        ],
        dtype=np.float64,
    )
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = np.array(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int64,
    )

    vertex_list = [vertex.copy() for vertex in vertices]
    face_list: list[tuple[int, int, int]] = [
        (int(face[0]), int(face[1]), int(face[2])) for face in faces
    ]
    for _ in range(refinement_level):
        edge_midpoints: dict[tuple[int, int], int] = {}
        refined: list[tuple[int, int, int]] = []

        for first, second, third in face_list:
            ab = _midpoint(first, second, vertex_list, edge_midpoints)
            bc = _midpoint(second, third, vertex_list, edge_midpoints)
            ca = _midpoint(third, first, vertex_list, edge_midpoints)
            refined.extend(
                [
                    (first, ab, ca),
                    (second, bc, ab),
                    (third, ca, bc),
                    (ab, bc, ca),
                ]
            )
        face_list = refined

    positions = radius_m * np.asarray(vertex_list, dtype=np.float64)
    connectivity = np.asarray(face_list, dtype=np.int64)
    cross = np.cross(
        positions[connectivity[:, 1]] - positions[connectivity[:, 0]],
        positions[connectivity[:, 2]] - positions[connectivity[:, 0]],
    )
    centroids = np.mean(positions[connectivity], axis=1)
    inward = np.einsum("ij,ij->i", cross, centroids) < 0.0
    connectivity[inward, 1], connectivity[inward, 2] = (
        connectivity[inward, 2].copy(),
        connectivity[inward, 1].copy(),
    )
    return positions, connectivity


def _midpoint(
    first: int,
    second: int,
    vertices: list[npt.NDArray[np.float64]],
    cache: dict[tuple[int, int], int],
) -> int:
    key = (min(first, second), max(first, second))
    if key not in cache:
        point = vertices[first] + vertices[second]
        point /= np.linalg.norm(point)
        cache[key] = len(vertices)
        vertices.append(point)
    return cache[key]
