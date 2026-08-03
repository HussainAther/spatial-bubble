"""Second-variation stability analysis for constrained equilibria.

The implementation evaluates a finite-difference Hessian of the dimensionless
Lagrangian gradient, projects it onto the tangent space of the equality
constraints, and diagonalizes the projected operator.  The method is an
engineering approximation: it is backend independent and evidence bearing, but
its accuracy depends on the finite-difference step, mesh, and eigenspace
separation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openphenomena.data import FloatArray
from openphenomena.optimization import ConstrainedProblem, SolverResult


@dataclass(frozen=True, slots=True)
class StabilitySettings:
    """Numerical controls for projected second-variation analysis."""

    relative_step: float = 2.0e-5
    rank_relative_tolerance: float = 1.0e-10
    zero_eigenvalue_tolerance: float = 2.0e-5
    negative_eigenvalue_tolerance: float = 2.0e-5
    maximum_modes: int = 12

    def __post_init__(self) -> None:
        if self.relative_step <= 0.0:
            raise ValueError("relative_step must be positive")
        if self.rank_relative_tolerance <= 0.0:
            raise ValueError("rank_relative_tolerance must be positive")
        if self.zero_eigenvalue_tolerance <= 0.0:
            raise ValueError("zero_eigenvalue_tolerance must be positive")
        if self.negative_eigenvalue_tolerance <= 0.0:
            raise ValueError("negative_eigenvalue_tolerance must be positive")
        if self.maximum_modes <= 0:
            raise ValueError("maximum_modes must be positive")


@dataclass(frozen=True, slots=True)
class StabilityResult:
    """Immutable constrained spectrum and mode shapes."""

    eigenvalues: FloatArray
    modes_dimensionless: FloatArray
    modes_physical: FloatArray
    constraint_rank: int
    tangent_dimension: int
    negative_mode_count: int
    null_mode_count: int
    smallest_eigenvalue: float
    smallest_nonnull_eigenvalue: float | None
    stable_semidefinite: bool
    hessian_symmetry_error: float
    finite_difference_step: float

    def __post_init__(self) -> None:
        eigenvalues = _readonly(self.eigenvalues)
        modes_dimensionless = _readonly(self.modes_dimensionless)
        modes_physical = _readonly(self.modes_physical)
        if eigenvalues.ndim != 1:
            raise ValueError("eigenvalues must be one-dimensional")
        if modes_dimensionless.shape[0] != len(eigenvalues):
            raise ValueError("mode count must match eigenvalue count")
        if modes_physical.shape != modes_dimensionless.shape:
            raise ValueError("physical and dimensionless mode arrays must match")
        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "modes_dimensionless", modes_dimensionless)
        object.__setattr__(self, "modes_physical", modes_physical)


def analyze_constrained_stability(
    problem: ConstrainedProblem,
    result: SolverResult,
    settings: StabilitySettings | None = None,
) -> StabilityResult:
    """Compute the projected finite-difference Lagrangian Hessian spectrum."""

    resolved = settings or StabilitySettings()
    x = np.asarray(result.solution_dimensionless, dtype=np.float64)
    multipliers = _multipliers_by_constraint(problem, result)
    jacobian = stacked_constraint_jacobian_dimensionless(problem, x)
    tangent, rank = constraint_tangent_basis(
        jacobian, resolved.rank_relative_tolerance
    )
    hessian = finite_difference_lagrangian_hessian(
        problem, x, multipliers, resolved.relative_step
    )
    symmetry_error = float(np.max(np.abs(hessian - hessian.T)))
    hessian = 0.5 * (hessian + hessian.T)
    projected = tangent.T @ hessian @ tangent
    projected = 0.5 * (projected + projected.T)
    eigenvalues, vectors = np.linalg.eigh(projected)
    count = min(resolved.maximum_modes, len(eigenvalues))
    eigenvalues = eigenvalues[:count]
    vectors = vectors[:, :count]
    modes_dimensionless = (tangent @ vectors).T
    modes_physical = modes_dimensionless * problem.variables.scale_to_physical[None, :]
    modes_physical = _normalize_modes(
        modes_physical, float(np.median(problem.variables.scale_to_physical))
    )
    modes_dimensionless = modes_physical / problem.variables.scale_to_physical[None, :]
    negative_count = int(
        np.count_nonzero(eigenvalues < -resolved.negative_eigenvalue_tolerance)
    )
    null_count = int(
        np.count_nonzero(np.abs(eigenvalues) <= resolved.zero_eigenvalue_tolerance)
    )
    nonnull = eigenvalues[np.abs(eigenvalues) > resolved.zero_eigenvalue_tolerance]
    return StabilityResult(
        eigenvalues=eigenvalues,
        modes_dimensionless=modes_dimensionless,
        modes_physical=modes_physical,
        constraint_rank=rank,
        tangent_dimension=tangent.shape[1],
        negative_mode_count=negative_count,
        null_mode_count=null_count,
        smallest_eigenvalue=float(eigenvalues[0]),
        smallest_nonnull_eigenvalue=float(nonnull[0]) if len(nonnull) else None,
        stable_semidefinite=negative_count == 0,
        hessian_symmetry_error=symmetry_error,
        finite_difference_step=resolved.relative_step,
    )


def finite_difference_lagrangian_hessian(
    problem: ConstrainedProblem,
    dimensionless_x: FloatArray,
    multipliers: tuple[FloatArray, ...],
    relative_step: float,
) -> FloatArray:
    """Differentiate the analytic dimensionless Lagrangian gradient centrally."""

    x = np.asarray(dimensionless_x, dtype=np.float64)
    hessian = np.empty((len(x), len(x)), dtype=np.float64)
    for column in range(len(x)):
        step = relative_step * max(1.0, abs(float(x[column])))
        plus = np.array(x, copy=True)
        minus = np.array(x, copy=True)
        plus[column] += step
        minus[column] -= step
        hessian[:, column] = (
            lagrangian_gradient_dimensionless(problem, plus, multipliers)
            - lagrangian_gradient_dimensionless(problem, minus, multipliers)
        ) / (2.0 * step)
    return np.asarray(hessian, dtype=np.float64)


def lagrangian_gradient_dimensionless(
    problem: ConstrainedProblem,
    dimensionless_x: FloatArray,
    multipliers: tuple[FloatArray, ...],
) -> FloatArray:
    """Evaluate ``grad f + J.T lambda`` in dimensionless coordinates."""

    x = np.asarray(dimensionless_x, dtype=np.float64)
    physical_x = x * problem.variables.scale_to_physical
    physical_gradient = np.asarray(problem.objective.gradient(physical_x))
    gradient = (
        physical_gradient
        * problem.variables.scale_to_physical
        / problem.objective.scale_to_physical
    )
    for constraint, multiplier in zip(
        problem.equality_constraints, multipliers, strict=True
    ):
        physical_jacobian = np.asarray(constraint.jacobian(physical_x))
        jacobian = (
            physical_jacobian
            * problem.variables.scale_to_physical[None, :]
            / constraint.scale_to_physical[:, None]
        )
        gradient = gradient + jacobian.T @ multiplier
    return np.asarray(gradient, dtype=np.float64)


def stacked_constraint_jacobian_dimensionless(
    problem: ConstrainedProblem, dimensionless_x: FloatArray
) -> FloatArray:
    """Stack all equality-constraint Jacobians in dimensionless coordinates."""

    x = np.asarray(dimensionless_x, dtype=np.float64)
    physical_x = x * problem.variables.scale_to_physical
    blocks = []
    for constraint in problem.equality_constraints:
        physical = np.asarray(constraint.jacobian(physical_x), dtype=np.float64)
        blocks.append(
            physical
            * problem.variables.scale_to_physical[None, :]
            / constraint.scale_to_physical[:, None]
        )
    return np.concatenate(blocks, axis=0)


def constraint_tangent_basis(
    constraint_jacobian: FloatArray, relative_tolerance: float
) -> tuple[FloatArray, int]:
    """Return an orthonormal basis for the Jacobian null space."""

    jacobian = np.asarray(constraint_jacobian, dtype=np.float64)
    _, singular_values, vh = np.linalg.svd(jacobian, full_matrices=True)
    scale = float(singular_values[0]) if len(singular_values) else 1.0
    rank = int(np.count_nonzero(singular_values > relative_tolerance * scale))
    basis = vh[rank:].T
    return np.asarray(basis, dtype=np.float64), rank


def _multipliers_by_constraint(
    problem: ConstrainedProblem, result: SolverResult
) -> tuple[FloatArray, ...]:
    estimates = {item.constraint_id: item for item in result.multiplier_estimates}
    values = []
    for constraint in problem.equality_constraints:
        if constraint.constraint_id not in estimates:
            raise ValueError(f"missing multiplier for {constraint.constraint_id}")
        values.append(
            np.asarray(
                estimates[constraint.constraint_id].dimensionless_values,
                dtype=np.float64,
            )
        )
    return tuple(values)


def _normalize_modes(modes: FloatArray, target_rms: float) -> FloatArray:
    normalized = np.array(modes, dtype=np.float64, copy=True)
    for index, mode in enumerate(normalized):
        rms = float(np.sqrt(np.mean(mode**2)))
        if rms > 0.0:
            normalized[index] = mode * (target_rms / rms)
    return normalized


def _readonly(values: FloatArray) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result
