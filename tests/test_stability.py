from __future__ import annotations

import numpy as np
import pytest

from openphenomena.equilibrium import (
    ClosedSphereConfig,
    InitialShape,
    StabilitySettings,
    analyze_constrained_stability,
    build_closed_sphere_problem,
    constraint_tangent_basis,
    generate_initial_mesh,
)
from openphenomena.equilibrium.reference import solve_case


def test_constraint_tangent_basis_is_orthonormal_and_annihilates_jacobian() -> None:
    jacobian = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    basis, rank = constraint_tangent_basis(jacobian, 1.0e-12)
    assert rank == 2
    np.testing.assert_allclose(basis.T @ basis, np.eye(1), atol=1.0e-14)
    np.testing.assert_allclose(jacobian @ basis, 0.0, atol=1.0e-14)


def test_closed_sphere_projected_second_variation_is_reproducible() -> None:
    config = ClosedSphereConfig()
    initial, faces = generate_initial_mesh(0, InitialShape.PERTURBED_SPHERE, config)
    case = solve_case(0, InitialShape.PERTURBED_SPHERE, config=config)
    problem = build_closed_sphere_problem(initial, faces, config)
    settings = StabilitySettings(maximum_modes=8)
    first = analyze_constrained_stability(problem, case.result, settings)
    second = analyze_constrained_stability(problem, case.result, settings)
    np.testing.assert_array_equal(first.eigenvalues, second.eigenvalues)
    np.testing.assert_array_equal(first.modes_physical, second.modes_physical)
    assert first.constraint_rank == 4
    assert first.tangent_dimension == initial.size - 4
    assert first.hessian_symmetry_error < 1.0e-7
    assert np.all(np.diff(first.eigenvalues) >= 0.0)
    assert first.negative_mode_count >= 0
    assert first.null_mode_count >= 0
    assert first.modes_physical.shape == (8, initial.size)
    rms = np.sqrt(np.mean(first.modes_physical**2, axis=1))
    np.testing.assert_allclose(rms, config.scales.length_m, rtol=1.0e-12, atol=1.0e-12)


def test_low_spectrum_is_insensitive_to_reasonable_difference_step() -> None:
    config = ClosedSphereConfig()
    initial, faces = generate_initial_mesh(0, InitialShape.PERTURBED_SPHERE, config)
    case = solve_case(0, InitialShape.PERTURBED_SPHERE, config=config)
    problem = build_closed_sphere_problem(initial, faces, config)
    fine = analyze_constrained_stability(
        problem, case.result, StabilitySettings(relative_step=1.0e-5, maximum_modes=8)
    )
    coarse = analyze_constrained_stability(
        problem, case.result, StabilitySettings(relative_step=4.0e-5, maximum_modes=8)
    )
    np.testing.assert_allclose(
        fine.eigenvalues[3:], coarse.eigenvalues[3:], rtol=2.0e-7, atol=1.0e-9
    )
    assert fine.null_mode_count == coarse.null_mode_count == 3
    assert fine.stable_semidefinite and coarse.stable_semidefinite


def test_stability_settings_reject_invalid_controls() -> None:
    with pytest.raises(ValueError):
        StabilitySettings(relative_step=0.0)
