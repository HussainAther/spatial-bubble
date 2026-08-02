from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from openphenomena import Provenance
from openphenomena.mesh import create_icosphere
from openphenomena.surface import (
    EquilibriumScales,
    area_gradient,
    centroid_jacobian,
    evaluate_surface_functionals,
    oriented_volume,
    surface_energy_gradient,
    total_area,
    total_surface_energy,
    triangle_areas,
    volume_centroid,
    volume_gradient,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@pytest.fixture
def tetrahedron() -> tuple[FloatArray, IntArray]:
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    outward_faces = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64
    )
    return positions, outward_faces


def _finite_difference_gradient(
    function: object, positions: FloatArray, step: float = 1.0e-4
) -> FloatArray:
    result = np.empty_like(positions)
    callable_function = function  # keep strict type checks out of numerical helper
    for vertex in range(len(positions)):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus_two = positions.copy()
            minus_two = positions.copy()
            plus[vertex, axis] += step
            minus[vertex, axis] -= step
            plus_two[vertex, axis] += 2.0 * step
            minus_two[vertex, axis] -= 2.0 * step
            result[vertex, axis] = (
                -callable_function(plus_two)  # type: ignore[operator]
                + 8.0 * callable_function(plus)  # type: ignore[operator]
                - 8.0 * callable_function(minus)  # type: ignore[operator]
                + callable_function(minus_two)  # type: ignore[operator]
            ) / (12.0 * step)
    return result


def _finite_difference_jacobian(
    function: object, positions: FloatArray, step: float = 1.0e-4
) -> FloatArray:
    result = np.empty((len(positions), 3, 3))
    callable_function = function
    for vertex in range(len(positions)):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus_two = positions.copy()
            minus_two = positions.copy()
            plus[vertex, axis] += step
            minus[vertex, axis] -= step
            plus_two[vertex, axis] += 2.0 * step
            minus_two[vertex, axis] -= 2.0 * step
            result[vertex, :, axis] = (
                -callable_function(plus_two)  # type: ignore[operator]
                + 8.0 * callable_function(plus)  # type: ignore[operator]
                - 8.0 * callable_function(minus)  # type: ignore[operator]
                + callable_function(minus_two)  # type: ignore[operator]
            ) / (12.0 * step)
    return result


def test_symbolic_tetrahedron_reference(
    tetrahedron: tuple[FloatArray, IntArray],
) -> None:
    positions, faces = tetrahedron
    expected_area = 1.5 + np.sqrt(3.0) / 2.0
    np.testing.assert_allclose(
        triangle_areas(positions, faces), [0.5, 0.5, 0.5, np.sqrt(3.0) / 2.0]
    )
    assert total_area(positions, faces) == pytest.approx(expected_area)
    assert oriented_volume(positions, faces) == pytest.approx(1.0 / 6.0)
    np.testing.assert_allclose(volume_centroid(positions, faces), [0.25] * 3)
    np.testing.assert_allclose(
        volume_gradient(positions, faces),
        np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        / 6.0,
    )
    np.testing.assert_allclose(
        centroid_jacobian(positions, faces),
        np.repeat((np.eye(3) / 4.0)[None, :, :], 4, axis=0),
    )


def test_area_derivative_matches_central_difference_on_tetrahedron(
    tetrahedron: tuple[FloatArray, IntArray],
) -> None:
    positions, faces = tetrahedron
    np.testing.assert_allclose(
        area_gradient(positions, faces),
        _finite_difference_gradient(lambda value: total_area(value, faces), positions),
        rtol=2.0e-8,
        atol=2.0e-9,
    )


def test_derivatives_on_nonsymmetric_closed_surface() -> None:
    positions, faces = create_icosphere(0, 1.0)
    positions = positions * np.array([1.17, 0.83, 1.31])
    positions += 0.07 * np.column_stack(
        (positions[:, 1] ** 2, positions[:, 2] * positions[:, 0], positions[:, 0] ** 2)
    )
    np.testing.assert_allclose(
        area_gradient(positions, faces),
        _finite_difference_gradient(lambda value: total_area(value, faces), positions),
        rtol=4.0e-8,
        atol=4.0e-9,
    )
    np.testing.assert_allclose(
        volume_gradient(positions, faces),
        _finite_difference_gradient(
            lambda value: oriented_volume(value, faces), positions
        ),
        rtol=4.0e-8,
        atol=4.0e-9,
    )
    np.testing.assert_allclose(
        centroid_jacobian(positions, faces),
        _finite_difference_jacobian(
            lambda value: volume_centroid(value, faces), positions
        ),
        rtol=5.0e-8,
        atol=5.0e-9,
    )
    np.testing.assert_allclose(
        surface_energy_gradient(positions, faces, 0.072, 2.0),
        _finite_difference_gradient(
            lambda value: total_surface_energy(value, faces, 0.072, 2.0), positions
        ),
        rtol=5.0e-8,
        atol=5.0e-9,
    )


def test_rigid_motion_invariance_and_gradient_covariance(
    tetrahedron: tuple[FloatArray, IntArray],
) -> None:
    positions, faces = tetrahedron
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([2.3, -1.7, 0.4])
    transformed = positions @ rotation.T + translation
    assert total_area(transformed, faces) == pytest.approx(total_area(positions, faces))
    assert oriented_volume(transformed, faces) == pytest.approx(
        oriented_volume(positions, faces)
    )
    np.testing.assert_allclose(
        volume_centroid(transformed, faces),
        volume_centroid(positions, faces) @ rotation.T + translation,
    )
    np.testing.assert_allclose(
        area_gradient(transformed, faces), area_gradient(positions, faces) @ rotation.T
    )
    np.testing.assert_allclose(
        volume_gradient(transformed, faces),
        volume_gradient(positions, faces) @ rotation.T,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        np.sum(area_gradient(positions, faces), axis=0), 0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        np.sum(volume_gradient(positions, faces), axis=0), 0.0, atol=1.0e-15
    )


def test_scaling_laws(tetrahedron: tuple[FloatArray, IntArray]) -> None:
    positions, faces = tetrahedron
    factor = 2.75
    scaled = positions * factor
    assert total_area(scaled, faces) == pytest.approx(
        factor**2 * total_area(positions, faces)
    )
    assert oriented_volume(scaled, faces) == pytest.approx(
        factor**3 * oriented_volume(positions, faces)
    )
    np.testing.assert_allclose(
        volume_centroid(scaled, faces), factor * volume_centroid(positions, faces)
    )
    np.testing.assert_allclose(
        area_gradient(scaled, faces), factor * area_gradient(positions, faces)
    )
    np.testing.assert_allclose(
        volume_gradient(scaled, faces), factor**2 * volume_gradient(positions, faces)
    )
    np.testing.assert_allclose(
        centroid_jacobian(scaled, faces),
        centroid_jacobian(positions, faces),
        atol=1.0e-15,
    )


def test_nondimensionalization_round_trips_and_volume_scale() -> None:
    radius_m = 0.03
    volume_m3 = 4.0 * np.pi * radius_m**3 / 3.0
    scales = EquilibriumScales.from_target_volume(volume_m3, 0.072, 2.0)
    assert scales.length_m == pytest.approx(volume_m3 ** (1.0 / 3.0))
    positions = np.array([[0.01, -0.02, 0.03]])
    np.testing.assert_allclose(
        scales.positions_to_si(scales.positions_to_dimensionless(positions)), positions
    )
    np.testing.assert_allclose(
        scales.centroid_to_si(scales.centroid_to_dimensionless(positions[0])),
        positions[0],
    )
    face_areas = np.array([0.001, 0.002])
    np.testing.assert_allclose(
        scales.face_areas_to_si(scales.face_areas_to_dimensionless(face_areas)),
        face_areas,
    )
    assert scales.area_to_si(scales.area_to_dimensionless(0.004)) == pytest.approx(
        0.004
    )
    assert scales.volume_to_si(
        scales.volume_to_dimensionless(volume_m3)
    ) == pytest.approx(volume_m3)
    assert scales.energy_to_si(scales.energy_to_dimensionless(1.2e-4)) == pytest.approx(
        1.2e-4
    )
    assert scales.pressure_to_si(
        scales.pressure_to_dimensionless(4.8)
    ) == pytest.approx(4.8)
    assert scales.curvature_to_si(
        scales.curvature_to_dimensionless(30.0)
    ) == pytest.approx(30.0)
    gradient = np.array([[0.3, -0.2, 0.1]])
    np.testing.assert_allclose(
        scales.area_gradient_to_si(scales.area_gradient_to_dimensionless(gradient)),
        gradient,
    )
    np.testing.assert_allclose(
        scales.volume_gradient_to_si(scales.volume_gradient_to_dimensionless(gradient)),
        gradient,
    )
    np.testing.assert_allclose(
        scales.energy_gradient_to_si(scales.energy_gradient_to_dimensionless(gradient)),
        gradient,
    )
    jacobian = np.eye(3)[None, :, :]
    np.testing.assert_allclose(
        scales.centroid_jacobian_to_si(
            scales.centroid_jacobian_to_dimensionless(jacobian)
        ),
        jacobian,
    )


def test_evaluation_is_immutable_unit_bearing_and_provenance_bearing(
    tetrahedron: tuple[FloatArray, IntArray],
) -> None:
    positions, faces = tetrahedron
    provenance = Provenance("exact evaluation", "tests.test_exact_surface", "1")
    result = evaluate_surface_functionals(positions, faces, 0.072, (provenance,), 2.0)
    fields = result.as_fields("laboratory_cartesian")
    assert fields["geometry.total_area"].descriptor.unit == "m^2"
    assert fields["energy.surface"].descriptor.unit == "J"
    assert fields["derivative.surface_energy_wrt_position"].descriptor.unit == "N"
    assert all(field.provenance == (provenance,) for field in fields.values())
    with pytest.raises(ValueError, match="read-only"):
        result.area_gradient_m[0, 0] = 0.0
    with pytest.raises(TypeError):
        fields["geometry.other"] = fields["geometry.total_area"]  # type: ignore[index]


def test_degenerate_and_zero_volume_inputs_are_rejected() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    with pytest.raises(ValueError, match="degenerate"):
        area_gradient(positions, faces)
    with pytest.raises(ValueError, match="closed"):
        volume_centroid(positions, faces)
