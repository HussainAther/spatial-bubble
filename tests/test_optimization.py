from __future__ import annotations

import inspect
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openphenomena import Domain, Frame, Provenance, Run, RunStatus, Study
from openphenomena.optimization import (
    ConstrainedProblem,
    ConstrainedSolverBackend,
    EqualityConstraint,
    HessianMode,
    IterationDiagnostic,
    OptimizationVariables,
    ScalarObjective,
    SolverSettings,
    SolverTolerances,
    TerminationCategory,
    ThresholdAcceptancePolicy,
)
from openphenomena.optimization.scipy_trust_constr import ScipyTrustConstrAdapter
from openphenomena.storage import read_run_bundle, write_run_bundle
from openphenomena.surface import (
    area_gradient,
    oriented_volume,
    total_area,
    volume_gradient,
)


def _plain_metadata(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_metadata(item) for item in value]
    return value


def _assert_deterministic_diagnostics(first: object, second: object) -> None:
    assert _plain_metadata(first) == _plain_metadata(second)


def _provenance(problem_id: str) -> tuple[Provenance, ...]:
    return (
        Provenance(
            activity="synthetic constrained-problem verification",
            implementation="tests.test_optimization",
            implementation_version="1",
            source_ids=(problem_id,),
        ),
    )


def _settings(max_iterations: int = 200) -> SolverSettings:
    return SolverSettings(
        tolerances=SolverTolerances(
            optimality=1.0e-11,
            step=1.0e-13,
            constraint=1.0e-10,
            barrier=1.0e-13,
        ),
        max_iterations=max_iterations,
    )


def _quadratic_problem(
    *,
    initial: np.ndarray[Any, Any] | None = None,
    variable_scales: np.ndarray[Any, Any] | None = None,
    objective_scale: float = 1.0,
    constraint_scale: float = 1.0,
) -> ConstrainedProblem:
    problem_id = "verification.quadratic_one_equality"
    return ConstrainedProblem(
        problem_id=problem_id,
        variables=OptimizationVariables(
            names=("x", "y"),
            initial_values=np.array([2.5, -1.0]) if initial is None else initial,
            scale_to_physical=(
                np.ones(2) if variable_scales is None else variable_scales
            ),
            units=("1", "1"),
        ),
        objective=ScalarObjective(
            objective_id="verification.quadratic_objective",
            value=lambda x: 0.5 * float((x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2),
            gradient=lambda x: np.array([x[0] - 1.0, x[1] - 2.0]),
            scale_to_physical=objective_scale,
            unit="1",
            hessian_mode=HessianMode.EXACT,
            hessian=lambda _x: np.eye(2),
        ),
        equality_constraints=(
            EqualityConstraint(
                constraint_id="verification.sum_equals_one",
                size=1,
                value=lambda x: np.array([x[0] + x[1] - 1.0]),
                jacobian=lambda _x: np.array([[1.0, 1.0]]),
                scale_to_physical=np.array([constraint_scale]),
                units=("1",),
                hessian_mode=HessianMode.EXACT,
                hessian=lambda _x, _weights: np.zeros((2, 2)),
            ),
        ),
        provenance=_provenance(problem_id),
    )


def test_backend_protocol_and_deterministic_options() -> None:
    adapter = ScipyTrustConstrAdapter()
    assert isinstance(adapter, ConstrainedSolverBackend)
    options = adapter.scipy_options(_settings(17))
    assert list(options) == [
        "barrier_tol",
        "disp",
        "factorization_method",
        "finite_diff_rel_step",
        "gtol",
        "initial_barrier_parameter",
        "initial_barrier_tolerance",
        "initial_constr_penalty",
        "initial_tr_radius",
        "maxiter",
        "sparse_jacobian",
        "verbose",
        "xtol",
    ]
    assert options["maxiter"] == 17
    assert options["disp"] is False


def test_backend_neutral_contract_import_does_not_load_scipy() -> None:
    code = (
        "import sys; import openphenomena.optimization; "
        "assert not any(name == 'scipy' or name.startswith('scipy.') "
        "for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_constrained_quadratic_solution_objective_and_multiplier() -> None:
    policy = ThresholdAcceptancePolicy(
        policy_id="verification.synthetic_acceptance",
        kkt_inf_norm=1.0e-9,
        constraint_inf_norm=1.0e-9,
        step_norm=3.0,
    )
    result = ScipyTrustConstrAdapter().solve(
        _quadratic_problem(), _settings(), acceptance_policy=policy
    )
    assert result.termination.category is TerminationCategory.CONVERGED_OPTIMALITY
    np.testing.assert_allclose(result.solution_physical, [0.0, 1.0], atol=1.0e-9)
    assert result.objective_physical == pytest.approx(1.0, abs=1.0e-10)
    assert result.equality_constraint_inf_norm < 1.0e-10
    assert result.lagrangian_kkt_inf_norm < 1.0e-9
    np.testing.assert_allclose(
        result.multiplier_estimates[0].physical_values, [1.0], atol=1.0e-9
    )
    assert result.multiplier_estimates[0].sign_convention.startswith("L_physical = f")
    assert result.scientifically_acceptable
    assert result.objective_gradient_inf_norm == pytest.approx(1.0, abs=1.0e-9)
    evidence = result.acceptance_evidence(
        "verification.quadratic_acceptance", ("solver-result.json",)
    )
    assert evidence.passed


def test_multiplier_restoration_accounts_for_objective_and_constraint_scales() -> None:
    problem = _quadratic_problem(
        variable_scales=np.array([3.0, 4.0]),
        objective_scale=10.0,
        constraint_scale=2.0,
    )
    result = ScipyTrustConstrAdapter().solve(problem, _settings())
    np.testing.assert_allclose(
        result.multiplier_estimates[0].physical_values, [1.0], atol=1.0e-9
    )
    np.testing.assert_allclose(
        result.multiplier_estimates[0].dimensionless_values, [0.2], atol=1.0e-9
    )


def test_multiple_linear_equalities_and_multiplier_order() -> None:
    problem_id = "verification.multiple_linear_equalities"
    problem = ConstrainedProblem(
        problem_id=problem_id,
        variables=OptimizationVariables(
            names=("x", "y", "z"),
            initial_values=np.zeros(3),
            scale_to_physical=np.ones(3),
            units=("1", "1", "1"),
        ),
        objective=ScalarObjective(
            objective_id="verification.three_variable_quadratic",
            value=lambda x: (
                0.5 * float(np.dot(x - [1.0, 2.0, 3.0], x - [1.0, 2.0, 3.0]))
            ),
            gradient=lambda x: x - np.array([1.0, 2.0, 3.0]),
            scale_to_physical=1.0,
            unit="1",
            hessian_mode=HessianMode.EXACT,
            hessian=lambda _x: np.eye(3),
        ),
        equality_constraints=(
            EqualityConstraint(
                "verification.first_linear_constraint",
                1,
                lambda x: np.array([x[0] + x[1]]),
                lambda _x: np.array([[1.0, 1.0, 0.0]]),
                np.ones(1),
                ("1",),
                HessianMode.EXACT,
                lambda _x, _w: np.zeros((3, 3)),
            ),
            EqualityConstraint(
                "verification.second_linear_constraint",
                1,
                lambda x: np.array([x[1] + x[2] - 1.0]),
                lambda _x: np.array([[0.0, 1.0, 1.0]]),
                np.ones(1),
                ("1",),
                HessianMode.EXACT,
                lambda _x, _w: np.zeros((3, 3)),
            ),
        ),
        provenance=_provenance(problem_id),
    )
    result = ScipyTrustConstrAdapter().solve(problem, _settings())
    np.testing.assert_allclose(
        result.solution_physical, [1 / 3, -1 / 3, 4 / 3], atol=1.0e-9
    )
    np.testing.assert_allclose(
        [item.physical_values[0] for item in result.multiplier_estimates],
        [2 / 3, 5 / 3],
        atol=1.0e-9,
    )
    assert result.lagrangian_kkt_inf_norm < 1.0e-9
    repeated = ScipyTrustConstrAdapter().solve(problem, _settings())
    _assert_deterministic_diagnostics(
        result.iteration_history, repeated.iteration_history
    )


def test_nonlinear_equality_known_solution_and_multiplier() -> None:
    problem_id = "verification.nonlinear_circle"
    problem = ConstrainedProblem(
        problem_id=problem_id,
        variables=OptimizationVariables(
            ("x", "y"), np.array([0.8, 0.6]), np.ones(2), ("1", "1")
        ),
        objective=ScalarObjective(
            "verification.closest_point",
            lambda x: 0.5 * float((x[0] - 2.0) ** 2 + x[1] ** 2),
            lambda x: np.array([x[0] - 2.0, x[1]]),
            1.0,
            "1",
            HessianMode.EXACT,
            lambda _x: np.eye(2),
        ),
        equality_constraints=(
            EqualityConstraint(
                "verification.unit_circle",
                1,
                lambda x: np.array([np.dot(x, x) - 1.0]),
                lambda x: np.array([[2.0 * x[0], 2.0 * x[1]]]),
                np.ones(1),
                ("1",),
                HessianMode.EXACT,
                lambda _x, weights: 2.0 * weights[0] * np.eye(2),
            ),
        ),
        provenance=_provenance(problem_id),
    )
    result = ScipyTrustConstrAdapter().solve(problem, _settings())
    np.testing.assert_allclose(result.solution_physical, [1.0, 0.0], atol=2.0e-8)
    assert result.objective_physical == pytest.approx(0.5, abs=2.0e-9)
    np.testing.assert_allclose(
        result.multiplier_estimates[0].physical_values, [0.5], atol=2.0e-8
    )
    assert result.lagrangian_kkt_inf_norm < 1.0e-8
    repeated = ScipyTrustConstrAdapter().solve(problem, _settings())
    _assert_deterministic_diagnostics(
        result.iteration_history, repeated.iteration_history
    )


def test_hessian_vector_product_mode_is_explicit_and_exercised() -> None:
    base = _quadratic_problem()
    objective = replace(
        base.objective,
        hessian_mode=HessianMode.HESSIAN_VECTOR_PRODUCT,
        hessian=None,
        hessian_vector_product=lambda _x, vector: np.array(vector, copy=True),
    )
    result = ScipyTrustConstrAdapter().solve(
        replace(base, objective=objective), _settings()
    )
    np.testing.assert_allclose(result.solution_physical, [0.0, 1.0], atol=1.0e-9)
    assert result.evaluations.objective_hessian > 0
    assert "objective:hessian_vector_product" in result.backend.hessian_strategy


def test_deliberately_infeasible_problem_is_not_scientifically_accepted() -> None:
    problem_id = "verification.infeasible_nonlinear_equality"
    problem = ConstrainedProblem(
        problem_id=problem_id,
        variables=OptimizationVariables(("x",), np.array([0.5]), np.ones(1), ("1",)),
        objective=ScalarObjective(
            "verification.scalar_quadratic",
            lambda x: 0.5 * float(x[0] ** 2),
            lambda x: np.array([x[0]]),
            1.0,
            "1",
            HessianMode.EXACT,
            lambda _x: np.ones((1, 1)),
        ),
        equality_constraints=(
            EqualityConstraint(
                "verification.impossible_positive_constraint",
                1,
                lambda x: np.array([x[0] ** 2 + 1.0]),
                lambda x: np.array([[2.0 * x[0]]]),
                np.ones(1),
                ("1",),
                HessianMode.EXACT,
                lambda _x, weights: np.array([[2.0 * weights[0]]]),
            ),
        ),
        provenance=_provenance(problem_id),
    )
    policy = ThresholdAcceptancePolicy(
        "verification.reject_infeasible", 1.0e-8, 1.0e-8, 1.0e-5
    )
    result = ScipyTrustConstrAdapter().solve(
        problem, _settings(40), acceptance_policy=policy
    )
    assert result.termination.category in {
        TerminationCategory.INFEASIBLE,
        TerminationCategory.ITERATION_LIMIT,
    }
    assert result.equality_constraint_inf_norm >= 1.0
    assert not result.scientifically_acceptable


def test_rank_deficient_constraint_jacobian_records_warning_or_failure() -> None:
    problem_id = "verification.rank_deficient_constraints"
    problem = ConstrainedProblem(
        problem_id=problem_id,
        variables=OptimizationVariables(
            ("x", "y"), np.array([2.0, -0.2]), np.ones(2), ("1", "1")
        ),
        objective=ScalarObjective(
            "verification.rank_deficient_objective",
            lambda x: 0.5 * float(np.dot(x - 1.0, x - 1.0)),
            lambda x: x - 1.0,
            1.0,
            "1",
            HessianMode.EXACT,
            lambda _x: np.eye(2),
        ),
        equality_constraints=(
            EqualityConstraint(
                "verification.redundant_equalities",
                2,
                lambda x: np.array([x[0] + x[1] - 1.0, 2.0 * x[0] + 2.0 * x[1] - 2.0]),
                lambda _x: np.array([[1.0, 1.0], [2.0, 2.0]]),
                np.ones(2),
                ("1", "1"),
                HessianMode.EXACT,
                lambda _x, _weights: np.zeros((2, 2)),
            ),
        ),
        provenance=_provenance(problem_id),
    )
    result = ScipyTrustConstrAdapter().solve(problem, _settings(80))
    assert result.numerical_warnings or not result.backend_terminated_successfully
    assert result.equality_constraint_inf_norm < 1.0e-7


def _large_scale_problem(
    scales: np.ndarray[Any, Any], objective_scale: float, constraint_scale: float
) -> ConstrainedProblem:
    problem_id = "verification.large_scale_quadratic"
    target = np.array([1.0e10, 2.0e10])
    return ConstrainedProblem(
        problem_id=problem_id,
        variables=OptimizationVariables(
            ("x", "y"), np.array([4.0e10, -1.0e10]), scales, ("m", "m")
        ),
        objective=ScalarObjective(
            "verification.large_scale_objective",
            lambda x: 0.5 * float(np.dot(x - target, x - target)),
            lambda x: x - target,
            objective_scale,
            "m^2",
            HessianMode.EXACT,
            lambda _x: np.eye(2),
        ),
        equality_constraints=(
            EqualityConstraint(
                "verification.large_scale_sum",
                1,
                lambda x: np.array([x[0] + x[1] - 3.0e10]),
                lambda _x: np.array([[1.0, 1.0]]),
                np.array([constraint_scale]),
                ("m",),
                HessianMode.EXACT,
                lambda _x, _weights: np.zeros((2, 2)),
            ),
        ),
        provenance=_provenance(problem_id),
    )


def test_explicit_nondimensionalization_improves_bad_scaling() -> None:
    adapter = ScipyTrustConstrAdapter()
    unscaled = adapter.solve(_large_scale_problem(np.ones(2), 1.0, 1.0), _settings(60))
    scaled = adapter.solve(
        _large_scale_problem(np.full(2, 1.0e10), 1.0e20, 1.0e10),
        _settings(60),
    )
    np.testing.assert_allclose(scaled.solution_physical, [1.0e10, 2.0e10], rtol=1.0e-10)
    assert scaled.lagrangian_kkt_inf_norm < 1.0e-9
    assert scaled.equality_constraint_inf_norm < 1.0e-9
    assert (
        not unscaled.backend_terminated_successfully
        or unscaled.iteration_count >= scaled.iteration_count
        or unscaled.lagrangian_kkt_inf_norm > scaled.lagrangian_kkt_inf_norm
    )


def test_iteration_limit_and_callback_stop_are_distinct() -> None:
    adapter = ScipyTrustConstrAdapter()
    limited = adapter.solve(_quadratic_problem(), _settings(1))
    assert limited.termination.category is TerminationCategory.ITERATION_LIMIT
    stopped = adapter.solve(
        _quadratic_problem(), _settings(), callback=lambda _diagnostic: True
    )
    assert stopped.termination.category is TerminationCategory.CALLBACK_STOP


def test_callback_history_and_diagnostics_are_deterministic() -> None:
    callback_records = []

    def callback(diagnostic: IterationDiagnostic) -> bool:
        callback_records.append(diagnostic)
        return False

    adapter = ScipyTrustConstrAdapter()
    first = adapter.solve(_quadratic_problem(), _settings(), callback=callback)
    second = adapter.solve(_quadratic_problem(), _settings())
    assert first.iteration_history == second.iteration_history
    assert first.evaluations == second.evaluations
    assert first.termination == second.termination
    assert tuple(callback_records) == first.iteration_history


def test_admissibility_and_backend_success_are_independent() -> None:
    problem = replace(
        _quadratic_problem(),
        admissibility=lambda _x: (False, ("synthetic admissibility rejection",)),
    )
    policy = ThresholdAcceptancePolicy(
        "verification.admissibility_policy", 1.0e-8, 1.0e-8, 3.0
    )
    result = ScipyTrustConstrAdapter().solve(
        problem, _settings(), acceptance_policy=policy
    )
    assert result.backend_terminated_successfully
    assert not result.admissible
    assert not result.scientifically_acceptable


def test_analytic_derivatives_are_called_and_wrong_gradient_test_is_external() -> None:
    calls = {"gradient": 0, "jacobian": 0}
    base = _quadratic_problem()

    def gradient(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        calls["gradient"] += 1
        return np.array([x[0] - 1.0, x[1] - 2.0])

    def jacobian(_x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        calls["jacobian"] += 1
        return np.array([[1.0, 1.0]])

    objective = replace(base.objective, gradient=gradient)
    constraint = replace(base.equality_constraints[0], jacobian=jacobian)
    result = ScipyTrustConstrAdapter().solve(
        replace(base, objective=objective, equality_constraints=(constraint,)),
        _settings(),
    )
    assert result.backend_terminated_successfully
    assert calls["gradient"] > 0 and calls["jacobian"] > 0
    assert result.evaluations.objective_gradient > 0
    assert result.evaluations.constraint_jacobians > 0

    # Testing-only finite difference detects a deliberately wrong derivative.
    point = np.array([0.3, -0.4])
    step = 1.0e-6
    finite_difference = np.empty(2)
    for axis in range(2):
        plus, minus = point.copy(), point.copy()
        plus[axis] += step
        minus[axis] -= step
        finite_difference[axis] = (
            base.objective.value(plus) - base.objective.value(minus)
        ) / (2.0 * step)
    wrong_gradient = np.zeros(2)
    assert np.max(np.abs(finite_difference - wrong_gradient)) > 0.5


def test_invalid_analytic_gradient_is_normalized_as_backend_failure() -> None:
    base = _quadratic_problem()
    bad_objective = replace(base.objective, gradient=lambda _x: np.zeros(3))
    result = ScipyTrustConstrAdapter().solve(
        replace(base, objective=bad_objective), _settings()
    )
    assert result.termination.category is TerminationCategory.BACKEND_FAILURE
    assert any("objective gradient" in item for item in result.numerical_warnings)


def test_solver_result_serializes_through_existing_run_bundle(tmp_path: Path) -> None:
    result = ScipyTrustConstrAdapter().solve(_quadratic_problem(), _settings())
    domain = Domain(
        domain_id="verification.empty_domain",
        kind="oriented_triangular_surface",
        coordinate_frame="dimensionless_cartesian",
        positions_m=np.zeros((3, 3)),
        faces=np.array([[0, 1, 2]]),
        fields={},
    )
    study = Study(
        "verification.solver_serialization",
        "Solver serialization",
        {},
        {},
        "test",
        "test",
        {},
    )
    run = Run(
        "verification.solver_run",
        study.study_id,
        RunStatus.COMPLETE,
        (),
        (Frame("verification.frame", 0.0, 0, (domain,)),),
        (),
        metadata={"solver_result": result.as_metadata()},
    )
    write_run_bundle(study, run, tmp_path)
    restored = read_run_bundle(tmp_path)[1]
    assert _plain_metadata(restored.metadata["solver_result"]) == _plain_metadata(
        result.as_metadata()
    )
    assert (
        restored.metadata["solver_result"]["backend"]["capability_id"]
        == ScipyTrustConstrAdapter().capability_id
    )


def test_geometry_kernel_interface_initialization_only() -> None:
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    target_volume = oriented_volume(positions, faces)
    problem_id = "integration.geometry_kernel_initialization"
    problem = ConstrainedProblem(
        problem_id=problem_id,
        variables=OptimizationVariables(
            tuple(f"x_{index}" for index in range(12)),
            positions.ravel(),
            np.ones(12),
            ("m",) * 12,
        ),
        objective=ScalarObjective(
            "integration.discrete_area",
            lambda x: total_area(x.reshape(-1, 3), faces),
            lambda x: area_gradient(x.reshape(-1, 3), faces).ravel(),
            total_area(positions, faces),
            "m^2",
            HessianMode.BFGS,
        ),
        equality_constraints=(
            EqualityConstraint(
                "integration.discrete_volume",
                1,
                lambda x: np.array(
                    [oriented_volume(x.reshape(-1, 3), faces) - target_volume]
                ),
                lambda x: volume_gradient(x.reshape(-1, 3), faces).reshape(1, -1),
                np.array([target_volume]),
                ("m^3",),
                HessianMode.BFGS,
            ),
        ),
        provenance=_provenance(problem_id),
    )
    result = ScipyTrustConstrAdapter().solve(problem, _settings(1))
    assert result.termination.category is TerminationCategory.ITERATION_LIMIT
    assert result.evaluations.objective_gradient > 0
    assert result.evaluations.constraint_jacobians > 0
    assert "objective:bfgs" in result.backend.hessian_strategy


def test_geometry_kernel_has_no_optimizer_or_scipy_imports() -> None:
    import openphenomena.surface.exact as exact_geometry

    source = inspect.getsource(exact_geometry)
    assert "scipy" not in source.lower()
    assert "optimization" not in source.lower()
