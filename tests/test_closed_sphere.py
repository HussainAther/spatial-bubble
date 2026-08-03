from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from openphenomena import RunStatus
from openphenomena.equilibrium import (
    ClosedSphereConfig,
    InitialShape,
    SphereAnalyticReference,
    build_closed_sphere_problem,
    generate_initial_mesh,
)
from openphenomena.equilibrium.closed_sphere import mesh_quality
from openphenomena.equilibrium.reference import (
    convergence_rows,
    run_closed_sphere_study,
    solve_case,
)
from openphenomena.storage import read_run_bundle
from openphenomena.surface import oriented_volume, volume_centroid


@pytest.mark.parametrize("shape", list(InitialShape))
def test_initial_meshes_are_deterministic_volume_normalized_and_admissible(
    shape: InitialShape,
) -> None:
    config = ClosedSphereConfig()
    first_positions, first_faces = generate_initial_mesh(1, shape, config)
    second_positions, second_faces = generate_initial_mesh(1, shape, config)
    np.testing.assert_array_equal(first_positions, second_positions)
    np.testing.assert_array_equal(first_faces, second_faces)
    assert oriented_volume(first_positions, first_faces) == pytest.approx(
        config.target_volume_m3, rel=2.0e-15
    )
    np.testing.assert_allclose(
        volume_centroid(first_positions, first_faces),
        config.target_centroid_array_m,
        atol=1.0e-17,
    )
    assert mesh_quality(first_positions, first_faces).admissible
    assert not first_positions.flags.writeable
    assert not first_faces.flags.writeable


def test_physical_problem_is_backend_neutral_and_uses_exact_contracts() -> None:
    code = (
        "import sys; import openphenomena.equilibrium.closed_sphere; "
        "assert not any(n == 'scipy' or n.startswith('scipy.') for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    config = ClosedSphereConfig()
    positions, faces = generate_initial_mesh(0, InitialShape.PERTURBED_SPHERE, config)
    problem = build_closed_sphere_problem(positions, faces, config)
    assert [item.constraint_id for item in problem.equality_constraints] == [
        "openphenomena.constraint.enclosed_volume",
        "openphenomena.constraint.volume_centroid_gauge",
    ]
    assert problem.objective.value(problem.variables.initial_values) > 0.0
    assert problem.objective.gradient(problem.variables.initial_values).shape == (
        positions.size,
    )
    assert problem.equality_constraints[0].jacobian(
        problem.variables.initial_values
    ).shape == (1, positions.size)
    assert problem.equality_constraints[1].jacobian(
        problem.variables.initial_values
    ).shape == (3, positions.size)


@pytest.mark.parametrize("shape", list(InitialShape))
def test_all_initializations_recover_same_level_one_equilibrium(
    shape: InitialShape,
) -> None:
    case = solve_case(1, shape)
    reference = SphereAnalyticReference.from_config(ClosedSphereConfig())
    assert case.result.backend_converged, (
        case.result.termination.category,
        case.result.termination.raw_message,
        case.result.iteration_count,
        case.result.lagrangian_kkt_inf_norm,
        case.result.equality_constraint_inf_norm,
    )
    assert case.acceptance.acceptable, case.acceptance.reasons
    assert case.result.lagrangian_kkt_inf_norm <= 3.1e-5
    assert case.metrics.volume_relative_residual <= 2.0e-9
    assert case.metrics.pressure_pa == pytest.approx(reference.pressure_pa, rel=0.02)
    assert case.metrics.energy_relative_error < 0.02
    assert case.metrics.mesh_quality.admissible
    assert case.metrics.relative_energy_decrease >= -1.0e-10
    assert all(item.passed for item in case.evidence)
    assert case.domain.fields["mechanics.pressure_jump"].values == pytest.approx(
        case.metrics.pressure_pa
    )


def test_pressure_is_negative_generic_volume_multiplier() -> None:
    case = solve_case(0, InitialShape.PERTURBED_SPHERE)
    volume_multiplier = next(
        item
        for item in case.result.multiplier_estimates
        if item.constraint_id == "openphenomena.constraint.enclosed_volume"
    )
    assert case.metrics.pressure_pa == pytest.approx(
        -float(volume_multiplier.physical_values[0])
    )
    assert volume_multiplier.sign_convention == (
        "L_physical = f_physical + lambda^T c_physical"
    )


def test_refinement_has_measured_error_and_multiplier_convergence() -> None:
    cases = tuple(
        solve_case(level, InitialShape.PERTURBED_SPHERE) for level in (0, 1, 2)
    )
    rows = convergence_rows(cases, ClosedSphereConfig())
    assert [row.refinement_level for row in rows] == [0, 1, 2]
    assert all(
        later.pressure_relative_error < earlier.pressure_relative_error
        and later.energy_relative_error < earlier.energy_relative_error
        and later.hausdorff_error_m < earlier.hausdorff_error_m
        for earlier, later in zip(rows[:-1], rows[1:], strict=True)
    )
    assert all(row.volume_relative_residual <= 2.0e-9 for row in rows)
    assert rows[0].pressure_observed_rate is None
    assert rows[1].pressure_observed_rate is not None
    assert rows[2].energy_observed_rate is not None


def test_study_bundle_round_trip_and_deterministic_scientific_results(
    tmp_path: Path,
) -> None:
    output = tmp_path / "closed-sphere"
    study, run = run_closed_sphere_study(output)
    assert run.status is RunStatus.COMPLETE
    restored_study, restored_run = read_run_bundle(output / "scientific")
    assert restored_study == study
    assert restored_run.status is RunStatus.COMPLETE
    assert len(restored_run.frames) == 6
    assert all(item.passed for item in restored_run.evidence)
    convergence = restored_run.metadata["convergence"]
    assert len(convergence) == 3
    assert asdict(ClosedSphereConfig()) == asdict(ClosedSphereConfig())
