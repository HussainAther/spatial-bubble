"""Exact discrete functionals for oriented triangular surface meshes.

"Exact" here means exact evaluation of the piecewise-linear discrete model,
up to floating-point roundoff. It does not claim that a triangulation exactly
represents an underlying smooth surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from openphenomena.data import (
    UNQUANTIFIED,
    Fidelity,
    Field,
    FieldAssociation,
    FieldDescriptor,
    FloatArray,
    IntArray,
    Provenance,
    ValidationStatus,
)

_LENGTH = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_AREA = (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_VOLUME = (3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_ENERGY = (2.0, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0)
_FORCE = (1.0, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0)
_DIMENSIONLESS = (0.0,) * 7


@dataclass(frozen=True, slots=True)
class SurfaceFunctionalEvaluation:
    """Immutable SI-valued evaluation of the discrete surface functionals."""

    face_areas_m2: FloatArray
    total_area_m2: float
    oriented_volume_m3: float
    volume_centroid_m: FloatArray
    total_surface_energy_j: float
    area_gradient_m: FloatArray
    volume_gradient_m2: FloatArray
    centroid_jacobian: FloatArray
    surface_energy_gradient_n: FloatArray
    surface_tension_n_per_m: float
    interface_multiplicity: float
    provenance: tuple[Provenance, ...]

    def __post_init__(self) -> None:
        if not self.provenance:
            raise ValueError("surface evaluation requires provenance")
        if not np.isfinite(self.surface_tension_n_per_m):
            raise ValueError("surface tension must be finite")
        if self.surface_tension_n_per_m < 0.0:
            raise ValueError("surface tension must be nonnegative")
        if not np.isfinite(self.interface_multiplicity):
            raise ValueError("interface multiplicity must be finite")
        if self.interface_multiplicity <= 0.0:
            raise ValueError("interface multiplicity must be positive")
        scalar_values = (
            self.total_area_m2,
            self.oriented_volume_m3,
            self.total_surface_energy_j,
        )
        if not all(np.isfinite(value) for value in scalar_values):
            raise ValueError("surface evaluation scalars must be finite")
        if self.total_area_m2 < 0.0 or self.total_surface_energy_j < 0.0:
            raise ValueError("area and surface energy must be nonnegative")
        for name in (
            "face_areas_m2",
            "volume_centroid_m",
            "area_gradient_m",
            "volume_gradient_m2",
            "centroid_jacobian",
            "surface_energy_gradient_n",
        ):
            value = _readonly_copy(getattr(self, name))
            object.__setattr__(self, name, value)
        vertex_count = self.area_gradient_m.shape[0]
        expected_shapes = {
            "face_areas_m2": (len(self.face_areas_m2),),
            "volume_centroid_m": (3,),
            "area_gradient_m": (vertex_count, 3),
            "volume_gradient_m2": (vertex_count, 3),
            "centroid_jacobian": (vertex_count, 3, 3),
            "surface_energy_gradient_n": (vertex_count, 3),
        }
        for name, expected_shape in expected_shapes.items():
            if getattr(self, name).shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}")
        object.__setattr__(self, "provenance", tuple(self.provenance))

    def as_fields(self, coordinate_frame: str) -> Mapping[str, Field]:
        """Expose every result as immutable, unit-bearing scientific fields."""

        if not coordinate_frame:
            raise ValueError("coordinate frame cannot be empty")
        definitions = (
            (
                "geometry.triangle_area",
                self.face_areas_m2,
                FieldAssociation.FACE,
                "m^2",
                _AREA,
                "Exact area of each piecewise-linear triangle.",
            ),
            (
                "geometry.total_area",
                np.array(self.total_area_m2),
                FieldAssociation.GLOBAL,
                "m^2",
                _AREA,
                "Total area of the piecewise-linear surface.",
            ),
            (
                "geometry.oriented_volume",
                np.array(self.oriented_volume_m3),
                FieldAssociation.GLOBAL,
                "m^3",
                _VOLUME,
                "Signed volume enclosed by the oriented triangle mesh.",
            ),
            (
                "geometry.volume_centroid",
                self.volume_centroid_m,
                FieldAssociation.GLOBAL,
                "m",
                _LENGTH,
                "Centroid of the signed enclosed volume.",
            ),
            (
                "energy.surface",
                np.array(self.total_surface_energy_j),
                FieldAssociation.GLOBAL,
                "J",
                _ENERGY,
                "Surface energy m gamma A.",
            ),
            (
                "derivative.area_wrt_position",
                self.area_gradient_m,
                FieldAssociation.VERTEX,
                "m",
                _LENGTH,
                "Analytic gradient of total area with respect to vertex position.",
            ),
            (
                "derivative.volume_wrt_position",
                self.volume_gradient_m2,
                FieldAssociation.VERTEX,
                "m^2",
                _AREA,
                "Analytic gradient of oriented volume with respect to vertex position.",
            ),
            (
                "derivative.centroid_wrt_position",
                self.centroid_jacobian,
                FieldAssociation.VERTEX,
                "1",
                _DIMENSIONLESS,
                "Analytic centroid Jacobian; axes are vertex, output, input.",
            ),
            (
                "derivative.surface_energy_wrt_position",
                self.surface_energy_gradient_n,
                FieldAssociation.VERTEX,
                "N",
                _FORCE,
                "Analytic gradient of surface energy with respect to vertex position.",
            ),
        )
        fields: dict[str, Field] = {}
        for (
            semantic_id,
            values,
            association,
            unit,
            dimension,
            description,
        ) in definitions:
            array = np.asarray(values, dtype=np.float64)
            descriptor = FieldDescriptor(
                semantic_id=semantic_id,
                association=association,
                unit=unit,
                unit_dimension=dimension,
                shape=array.shape,
                dtype=array.dtype.str,
                coordinate_frame=coordinate_frame,
                generating_model="piecewise-linear oriented surface functionals",
                generating_implementation="openphenomena.surface.exact.evaluate_surface_functionals",
                fidelity=Fidelity.ENGINEERING_APPROXIMATION,
                validation_status=ValidationStatus.VERIFIED,
                uncertainty=UNQUANTIFIED,
                description=description,
            )
            fields[semantic_id] = Field(descriptor, array, self.provenance)
        return MappingProxyType(fields)


def triangle_areas(positions_m: FloatArray, faces: IntArray) -> FloatArray:
    """Return exact piecewise-linear triangle areas in square metres."""

    positions, connectivity = _validated_mesh(positions_m, faces)
    first = positions[connectivity[:, 1]] - positions[connectivity[:, 0]]
    second = positions[connectivity[:, 2]] - positions[connectivity[:, 0]]
    return _readonly_copy(0.5 * np.linalg.norm(np.cross(first, second), axis=1))


def total_area(positions_m: FloatArray, faces: IntArray) -> float:
    """Return total piecewise-linear surface area in square metres."""

    return float(np.sum(triangle_areas(positions_m, faces)))


def area_gradient(positions_m: FloatArray, faces: IntArray) -> FloatArray:
    """Return the analytic gradient of total area in metres."""

    positions, connectivity = _validated_mesh(positions_m, faces)
    a, b, c = (positions[connectivity[:, index]] for index in range(3))
    cross = np.cross(b - a, c - a)
    double_area = np.linalg.norm(cross, axis=1)
    if np.any(double_area == 0.0):
        raise ValueError("area gradient is undefined for a degenerate triangle")
    normals = cross / double_area[:, None]
    local = (
        0.5 * np.cross(b - c, normals),
        0.5 * np.cross(c - a, normals),
        0.5 * np.cross(a - b, normals),
    )
    gradient = np.zeros_like(positions)
    for local_index, contribution in enumerate(local):
        np.add.at(gradient, connectivity[:, local_index], contribution)
    return _readonly_copy(gradient)


def oriented_volume(positions_m: FloatArray, faces: IntArray) -> float:
    """Return signed polyhedral volume in cubic metres."""

    positions, connectivity = _validated_mesh(positions_m, faces)
    _validate_closed_oriented_surface(connectivity)
    a, b, c = (positions[connectivity[:, index]] for index in range(3))
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


def volume_gradient(positions_m: FloatArray, faces: IntArray) -> FloatArray:
    """Return the analytic gradient of signed volume in square metres."""

    positions, connectivity = _validated_mesh(positions_m, faces)
    _validate_closed_oriented_surface(connectivity)
    a, b, c = (positions[connectivity[:, index]] for index in range(3))
    local = (np.cross(b, c) / 6.0, np.cross(c, a) / 6.0, np.cross(a, b) / 6.0)
    gradient = np.zeros_like(positions)
    for local_index, contribution in enumerate(local):
        np.add.at(gradient, connectivity[:, local_index], contribution)
    return _readonly_copy(gradient)


def volume_centroid(positions_m: FloatArray, faces: IntArray) -> FloatArray:
    """Return the centroid of signed tetrahedra formed with the origin."""

    positions, connectivity = _validated_mesh(positions_m, faces)
    _validate_closed_oriented_surface(connectivity)
    volume, first_moment = _volume_and_first_moment(positions, connectivity)
    if volume == 0.0:
        raise ValueError("volume centroid is undefined for zero oriented volume")
    return _readonly_copy(first_moment / volume)


def centroid_jacobian(positions_m: FloatArray, faces: IntArray) -> FloatArray:
    """Return analytic ``d centroid / d position`` with shape ``(n, 3, 3)``."""

    positions, connectivity = _validated_mesh(positions_m, faces)
    _validate_closed_oriented_surface(connectivity)
    a, b, c = (positions[connectivity[:, index]] for index in range(3))
    signed_tetra_volumes = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    volume = float(np.sum(signed_tetra_volumes))
    if volume == 0.0:
        raise ValueError("centroid Jacobian is undefined for zero oriented volume")
    vertex_sum = a + b + c
    first_moment = np.sum(signed_tetra_volumes[:, None] * vertex_sum / 4.0, axis=0)
    centroid = first_moment / volume
    volume_local = (
        np.cross(b, c) / 6.0,
        np.cross(c, a) / 6.0,
        np.cross(a, b) / 6.0,
    )
    moment_jacobian = np.zeros((len(positions), 3, 3), dtype=np.float64)
    volume_derivative = np.zeros_like(positions)
    identity = np.eye(3)
    for local_index, volume_contribution in enumerate(volume_local):
        contribution = (
            np.einsum("fi,fj->fij", vertex_sum / 4.0, volume_contribution)
            + signed_tetra_volumes[:, None, None] * identity / 4.0
        )
        np.add.at(moment_jacobian, connectivity[:, local_index], contribution)
        np.add.at(
            volume_derivative,
            connectivity[:, local_index],
            volume_contribution,
        )
    jacobian = (
        moment_jacobian - np.einsum("i,vj->vij", centroid, volume_derivative)
    ) / volume
    return _readonly_copy(jacobian)


def total_surface_energy(
    positions_m: FloatArray,
    faces: IntArray,
    surface_tension_n_per_m: float,
    interface_multiplicity: float = 1.0,
) -> float:
    """Return ``m gamma A`` in joules."""

    _validate_material(surface_tension_n_per_m, interface_multiplicity)
    return (
        interface_multiplicity
        * surface_tension_n_per_m
        * total_area(positions_m, faces)
    )


def surface_energy_gradient(
    positions_m: FloatArray,
    faces: IntArray,
    surface_tension_n_per_m: float,
    interface_multiplicity: float = 1.0,
) -> FloatArray:
    """Return the analytic gradient of ``m gamma A`` in newtons."""

    _validate_material(surface_tension_n_per_m, interface_multiplicity)
    return _readonly_copy(
        interface_multiplicity
        * surface_tension_n_per_m
        * area_gradient(positions_m, faces)
    )


def evaluate_surface_functionals(
    positions_m: FloatArray,
    faces: IntArray,
    surface_tension_n_per_m: float,
    provenance: tuple[Provenance, ...],
    interface_multiplicity: float = 1.0,
) -> SurfaceFunctionalEvaluation:
    """Evaluate the complete backend-independent discrete geometric state."""

    areas = triangle_areas(positions_m, faces)
    area_derivative = area_gradient(positions_m, faces)
    volume = oriented_volume(positions_m, faces)
    centroid = volume_centroid(positions_m, faces)
    volume_derivative = volume_gradient(positions_m, faces)
    centroid_derivative = centroid_jacobian(positions_m, faces)
    energy = total_surface_energy(
        positions_m, faces, surface_tension_n_per_m, interface_multiplicity
    )
    energy_derivative = surface_energy_gradient(
        positions_m, faces, surface_tension_n_per_m, interface_multiplicity
    )
    return SurfaceFunctionalEvaluation(
        face_areas_m2=areas,
        total_area_m2=float(np.sum(areas)),
        oriented_volume_m3=volume,
        volume_centroid_m=centroid,
        total_surface_energy_j=energy,
        area_gradient_m=area_derivative,
        volume_gradient_m2=volume_derivative,
        centroid_jacobian=centroid_derivative,
        surface_energy_gradient_n=energy_derivative,
        surface_tension_n_per_m=surface_tension_n_per_m,
        interface_multiplicity=interface_multiplicity,
        provenance=provenance,
    )


def _validated_mesh(
    positions_m: FloatArray, faces: IntArray
) -> tuple[FloatArray, IntArray]:
    positions = np.asarray(positions_m, dtype=np.float64)
    connectivity = np.asarray(faces, dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_m must have shape (n_vertices, 3)")
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError("faces must have shape (n_faces, 3)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_m must be finite")
    if connectivity.size and (
        int(np.min(connectivity)) < 0 or int(np.max(connectivity)) >= len(positions)
    ):
        raise ValueError("faces contain an out-of-range vertex index")
    return positions, connectivity


def _volume_and_first_moment(
    positions: FloatArray, connectivity: IntArray
) -> tuple[float, FloatArray]:
    a, b, c = (positions[connectivity[:, index]] for index in range(3))
    tetra_volumes = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    volume = float(np.sum(tetra_volumes))
    moment = np.sum(tetra_volumes[:, None] * (a + b + c) / 4.0, axis=0)
    return volume, moment


def _validate_closed_oriented_surface(connectivity: IntArray) -> None:
    """Require every undirected edge exactly twice with opposite directions."""

    edge_balance: dict[tuple[int, int], tuple[int, int]] = {}
    for face in connectivity:
        for first_raw, second_raw in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            first, second = int(first_raw), int(second_raw)
            if first == second:
                raise ValueError("closed surface contains a collapsed edge")
            key = (min(first, second), max(first, second))
            count, balance = edge_balance.get(key, (0, 0))
            direction = 1 if (first, second) == key else -1
            edge_balance[key] = (count + 1, balance + direction)
    if not edge_balance or any(
        count != 2 or balance != 0 for count, balance in edge_balance.values()
    ):
        raise ValueError(
            "oriented volume requires a closed, consistently oriented manifold"
        )


def _validate_material(surface_tension: float, multiplicity: float) -> None:
    if not np.isfinite(surface_tension) or surface_tension < 0.0:
        raise ValueError("surface tension must be finite and nonnegative")
    if not np.isfinite(multiplicity) or multiplicity <= 0.0:
        raise ValueError("interface multiplicity must be finite and positive")


def _readonly_copy(values: FloatArray) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("surface evaluation arrays must be finite")
    result.flags.writeable = False
    return result
