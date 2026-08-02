"""Cotan-Laplace and angle-defect operators for triangle meshes.

The mixed-area and curvature formulas follow:

Meyer, Desbrun, Schroder, and Barr, "Discrete Differential-Geometry Operators
for Triangulated 2-Manifolds," Visualization and Mathematics III (2003).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class SurfaceGeometry:
    """Discrete geometric measurements on an oriented triangular surface."""

    face_areas_m2: FloatArray
    face_normals: FloatArray
    vertex_areas_m2: FloatArray
    vertex_normals: FloatArray
    mean_curvature_per_m: FloatArray
    gaussian_curvature_per_m2: FloatArray
    characteristic_edge_length_m: float


def analyze_surface(positions_m: FloatArray, faces: IntArray) -> SurfaceGeometry:
    """Compute mixed-area cotangent and angle-defect geometry."""

    positions = np.asarray(positions_m, dtype=np.float64)
    connectivity = np.asarray(faces, dtype=np.int64)
    face_areas, face_normals = _face_geometry(positions, connectivity)
    vertex_areas = np.zeros(len(positions), dtype=np.float64)
    angle_sums = np.zeros(len(positions), dtype=np.float64)
    normal_accumulator = np.zeros_like(positions)
    edge_weights: dict[tuple[int, int], float] = {}

    for face_index, (i_raw, j_raw, k_raw) in enumerate(connectivity):
        i, j, k = int(i_raw), int(j_raw), int(k_raw)
        a, b, c = positions[i], positions[j], positions[k]
        angles = _triangle_angles(a, b, c)
        angle_sums[[i, j, k]] += angles
        area = face_areas[face_index]
        normal_accumulator[[i, j, k]] += face_normals[face_index] * area

        if np.max(angles) > np.pi / 2.0:
            obtuse_local = int(np.argmax(angles))
            shares = np.full(3, area / 4.0)
            shares[obtuse_local] = area / 2.0
            vertex_areas[[i, j, k]] += shares
        else:
            cot_i, cot_j, cot_k = 1.0 / np.tan(angles)
            lij2 = float(np.dot(a - b, a - b))
            lik2 = float(np.dot(a - c, a - c))
            ljk2 = float(np.dot(b - c, b - c))
            vertex_areas[i] += (cot_k * lij2 + cot_j * lik2) / 8.0
            vertex_areas[j] += (cot_k * lij2 + cot_i * ljk2) / 8.0
            vertex_areas[k] += (cot_j * lik2 + cot_i * ljk2) / 8.0

        _add_edge_weight(edge_weights, i, j, 1.0 / np.tan(angles[2]))
        _add_edge_weight(edge_weights, j, k, 1.0 / np.tan(angles[0]))
        _add_edge_weight(edge_weights, k, i, 1.0 / np.tan(angles[1]))

    if np.any(vertex_areas <= 0.0):
        raise ValueError("surface contains a vertex with nonpositive mixed area")
    vertex_norms = np.linalg.norm(normal_accumulator, axis=1)
    if np.any(vertex_norms == 0.0):
        raise ValueError("surface contains an undefined vertex normal")
    vertex_normals = normal_accumulator / vertex_norms[:, None]

    laplace_position = np.zeros_like(positions)
    edge_lengths: list[float] = []
    for (i, j), weight in edge_weights.items():
        displacement = positions[j] - positions[i]
        laplace_position[i] += weight * displacement
        laplace_position[j] -= weight * displacement
        edge_lengths.append(float(np.linalg.norm(displacement)))
    laplace_position /= 2.0 * vertex_areas[:, None]

    # With outward normals, Delta_s x = -2 H n for a convex sphere. This sign
    # convention therefore makes convex outward-oriented spheres have H > 0.
    mean_curvature = -0.5 * np.einsum("ij,ij->i", laplace_position, vertex_normals)
    gaussian_curvature = (2.0 * np.pi - angle_sums) / vertex_areas
    characteristic_edge_length = float(
        np.sqrt(np.mean(np.square(np.asarray(edge_lengths))))
    )
    return SurfaceGeometry(
        face_areas_m2=_readonly(face_areas),
        face_normals=_readonly(face_normals),
        vertex_areas_m2=_readonly(vertex_areas),
        vertex_normals=_readonly(vertex_normals),
        mean_curvature_per_m=_readonly(mean_curvature),
        gaussian_curvature_per_m2=_readonly(gaussian_curvature),
        characteristic_edge_length_m=characteristic_edge_length,
    )


def _face_geometry(
    positions: FloatArray,
    faces: IntArray,
) -> tuple[FloatArray, FloatArray]:
    edge_a = positions[faces[:, 1]] - positions[faces[:, 0]]
    edge_b = positions[faces[:, 2]] - positions[faces[:, 0]]
    cross = np.cross(edge_a, edge_b)
    double_areas = np.linalg.norm(cross, axis=1)
    if np.any(double_areas == 0.0):
        raise ValueError("surface contains a degenerate triangle")
    return 0.5 * double_areas, cross / double_areas[:, None]


def _triangle_angles(a: FloatArray, b: FloatArray, c: FloatArray) -> FloatArray:
    def angle(first: FloatArray, center: FloatArray, second: FloatArray) -> float:
        left = first - center
        right = second - center
        cosine = np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
        return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    return np.array([angle(b, a, c), angle(a, b, c), angle(a, c, b)])


def _add_edge_weight(
    weights: dict[tuple[int, int], float],
    first: int,
    second: int,
    value: float,
) -> None:
    key = (min(first, second), max(first, second))
    weights[key] = weights.get(key, 0.0) + value


def _readonly(values: FloatArray) -> FloatArray:
    values.flags.writeable = False
    return values
