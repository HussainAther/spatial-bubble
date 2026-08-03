"""Isolated SciPy ``trust-constr`` adapter for moderate serial problems."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import scipy  # type: ignore[import-untyped]
from scipy.optimize import (  # type: ignore[import-untyped]
    BFGS,
    NonlinearConstraint,
    minimize,
)

from openphenomena.data import FloatArray, Provenance
from openphenomena.optimization.contracts import (
    AcceptancePolicy,
    BackendInformation,
    BackendTermination,
    ConstrainedProblem,
    ConstraintResidual,
    DiagnosticCallback,
    EqualityConstraint,
    EvaluationCounts,
    HessianMode,
    IterationDiagnostic,
    MultiplierEstimate,
    SolverResult,
    SolverSettings,
    TerminationCategory,
)


@dataclass(slots=True)
class _EvaluationCounter:
    objective: int = 0
    objective_gradient: int = 0
    objective_hessian: int = 0
    constraints: int = 0
    constraint_jacobians: int = 0
    constraint_hessians: int = 0

    def snapshot(self) -> EvaluationCounts:
        return EvaluationCounts(
            objective=self.objective,
            objective_gradient=self.objective_gradient,
            objective_hessian=self.objective_hessian,
            constraints=self.constraints,
            constraint_jacobians=self.constraint_jacobians,
            constraint_hessians=self.constraint_hessians,
        )


@dataclass(slots=True)
class _CallbackState:
    previous_x: FloatArray
    previous_objective: float
    last_step_norm: float = 0.0
    last_objective_change: float = 0.0


class ScipyTrustConstrAdapter:
    """Translate backend-neutral contracts to SciPy without leaking its types."""

    _CAPABILITY_ID = "openphenomena.equilibrium.solver.scipy_trust_constr.v1"

    @property
    def capability_id(self) -> str:
        return self._CAPABILITY_ID

    def solve(
        self,
        problem: ConstrainedProblem,
        settings: SolverSettings,
        callback: DiagnosticCallback | None = None,
        acceptance_policy: AcceptancePolicy | None = None,
    ) -> SolverResult:
        counter = _EvaluationCounter()
        variable_scales = problem.variables.scale_to_physical
        initial = problem.variables.initial_dimensionless
        history: list[IterationDiagnostic] = []
        callback_state = _CallbackState(
            previous_x=np.array(initial, copy=True),
            previous_objective=self._objective_dimensionless(problem, initial),
        )

        objective = self._objective_callable(problem, counter)
        gradient = self._gradient_callable(problem, counter)
        hessian_arguments = self._objective_hessian_arguments(problem, counter)
        constraints = tuple(
            self._constraint_adapter(item, variable_scales, counter)
            for item in problem.equality_constraints
        )
        options = self.scipy_options(settings)

        def scipy_callback(x: FloatArray, state: Any) -> bool:
            diagnostic = self._iteration_diagnostic(
                problem, np.asarray(x, dtype=np.float64), state, callback_state
            )
            if diagnostic.iteration % settings.history_stride == 0:
                history.append(diagnostic)
            callback_state.previous_x = np.array(x, dtype=np.float64, copy=True)
            callback_state.previous_objective = diagnostic.objective_dimensionless
            callback_state.last_step_norm = diagnostic.step_norm
            callback_state.last_objective_change = diagnostic.objective_change
            return bool(callback is not None and callback(diagnostic))

        caught_messages: list[str] = []
        raw_result: Any | None = None
        backend_exception: Exception | None = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                raw_result = minimize(
                    objective,
                    np.array(initial, copy=True),
                    method="trust-constr",
                    jac=gradient,
                    constraints=constraints,
                    callback=scipy_callback,
                    options=options,
                    **hessian_arguments,
                )
            except Exception as exc:  # normalized into a backend-neutral result
                backend_exception = exc
            caught_messages.extend(
                f"{item.category.__name__}: {item.message}" for item in caught
            )

        if raw_result is None:
            assert backend_exception is not None
            caught_messages.append(
                f"{type(backend_exception).__name__}: {backend_exception}"
            )
            result = self._failure_result(
                problem,
                settings,
                counter,
                history,
                tuple(caught_messages),
                callback_state,
            )
        else:
            result = self._normalized_result(
                problem,
                settings,
                counter,
                history,
                tuple(caught_messages),
                callback_state,
                raw_result,
            )
        if acceptance_policy is not None:
            result = replace(
                result,
                scientific_acceptance=acceptance_policy.evaluate(result),
            )
        return result

    @staticmethod
    def scipy_options(settings: SolverSettings) -> dict[str, object]:
        """Return the complete deterministic option map passed to SciPy."""

        options: dict[str, object] = {
            "barrier_tol": settings.tolerances.barrier,
            "disp": False,
            "factorization_method": None,
            "finite_diff_rel_step": None,
            "gtol": settings.tolerances.optimality,
            "initial_barrier_parameter": 0.1,
            "initial_barrier_tolerance": 0.1,
            "initial_constr_penalty": settings.initial_constraint_penalty,
            "initial_tr_radius": settings.initial_trust_radius,
            "maxiter": settings.max_iterations,
            "sparse_jacobian": False,
            "verbose": 0,
            "xtol": settings.tolerances.step,
        }
        return dict(sorted(options.items()))

    @staticmethod
    def _objective_dimensionless(
        problem: ConstrainedProblem, dimensionless_x: FloatArray
    ) -> float:
        physical_x = dimensionless_x * problem.variables.scale_to_physical
        return float(problem.objective.value(physical_x)) / (
            problem.objective.scale_to_physical
        )

    def _objective_callable(
        self, problem: ConstrainedProblem, counter: _EvaluationCounter
    ) -> Any:
        def objective(dimensionless_x: FloatArray) -> float:
            counter.objective += 1
            return self._objective_dimensionless(problem, dimensionless_x)

        return objective

    @staticmethod
    def _gradient_dimensionless(
        problem: ConstrainedProblem, dimensionless_x: FloatArray
    ) -> FloatArray:
        physical_x = dimensionless_x * problem.variables.scale_to_physical
        physical_gradient = np.asarray(
            problem.objective.gradient(physical_x), dtype=np.float64
        )
        expected = problem.variables.initial_values.shape
        if physical_gradient.shape != expected or not np.all(
            np.isfinite(physical_gradient)
        ):
            raise ValueError(f"objective gradient must have shape {expected}")
        return (
            physical_gradient
            * problem.variables.scale_to_physical
            / problem.objective.scale_to_physical
        )

    def _gradient_callable(
        self, problem: ConstrainedProblem, counter: _EvaluationCounter
    ) -> Any:
        def gradient(dimensionless_x: FloatArray) -> FloatArray:
            counter.objective_gradient += 1
            return self._gradient_dimensionless(problem, dimensionless_x)

        return gradient

    def _objective_hessian_arguments(
        self, problem: ConstrainedProblem, counter: _EvaluationCounter
    ) -> dict[str, object]:
        objective = problem.objective
        scales = problem.variables.scale_to_physical
        objective_scale = objective.scale_to_physical
        if objective.hessian_mode is HessianMode.BFGS:
            return {"hess": BFGS(exception_strategy="skip_update", init_scale="auto")}
        if objective.hessian_mode is HessianMode.EXACT:
            assert objective.hessian is not None
            exact_hessian_function = objective.hessian

            def hessian(dimensionless_x: FloatArray) -> FloatArray:
                counter.objective_hessian += 1
                physical_x = dimensionless_x * scales
                physical_hessian = np.asarray(
                    exact_hessian_function(physical_x), dtype=np.float64
                )
                _validate_square_matrix(
                    physical_hessian, len(scales), "objective Hessian"
                )
                return (
                    scales[:, None]
                    * physical_hessian
                    * scales[None, :]
                    / objective_scale
                )

            return {"hess": hessian}
        assert objective.hessian_vector_product is not None
        exact_hessian_vector_product = objective.hessian_vector_product

        def hessian_vector_product(
            dimensionless_x: FloatArray, dimensionless_vector: FloatArray
        ) -> FloatArray:
            counter.objective_hessian += 1
            physical_x = dimensionless_x * scales
            physical_vector = dimensionless_vector * scales
            physical_product = np.asarray(
                exact_hessian_vector_product(physical_x, physical_vector),
                dtype=np.float64,
            )
            if physical_product.shape != scales.shape:
                raise ValueError("objective Hessian-vector product has wrong shape")
            return scales * physical_product / objective_scale

        return {"hessp": hessian_vector_product}

    def _constraint_adapter(
        self,
        constraint: EqualityConstraint,
        variable_scales: FloatArray,
        counter: _EvaluationCounter,
    ) -> NonlinearConstraint:
        def value(dimensionless_x: FloatArray) -> FloatArray:
            counter.constraints += 1
            return self._constraint_dimensionless(
                constraint, variable_scales, dimensionless_x
            )

        def jacobian(dimensionless_x: FloatArray) -> FloatArray:
            counter.constraint_jacobians += 1
            return self._constraint_jacobian_dimensionless(
                constraint, variable_scales, dimensionless_x
            )

        if constraint.hessian_mode is HessianMode.BFGS:
            hessian: object = BFGS(exception_strategy="skip_update", init_scale="auto")
        else:
            assert constraint.hessian is not None
            exact_constraint_hessian = constraint.hessian

            def exact_hessian(
                dimensionless_x: FloatArray, dimensionless_multipliers: FloatArray
            ) -> FloatArray:
                counter.constraint_hessians += 1
                physical_x = dimensionless_x * variable_scales
                physical_weights = (
                    np.asarray(dimensionless_multipliers, dtype=np.float64)
                    / constraint.scale_to_physical
                )
                physical_hessian = np.asarray(
                    exact_constraint_hessian(physical_x, physical_weights),
                    dtype=np.float64,
                )
                _validate_square_matrix(
                    physical_hessian,
                    len(variable_scales),
                    f"constraint Hessian {constraint.constraint_id}",
                )
                return (
                    variable_scales[:, None]
                    * physical_hessian
                    * variable_scales[None, :]
                )

            hessian = exact_hessian
        zeros = np.zeros(constraint.size, dtype=np.float64)
        return NonlinearConstraint(
            value,
            zeros,
            zeros,
            jac=jacobian,
            hess=hessian,
            keep_feasible=False,
        )

    @staticmethod
    def _constraint_dimensionless(
        constraint: EqualityConstraint,
        variable_scales: FloatArray,
        dimensionless_x: FloatArray,
    ) -> FloatArray:
        physical_x = dimensionless_x * variable_scales
        physical_value = np.asarray(constraint.value(physical_x), dtype=np.float64)
        if physical_value.shape != (constraint.size,) or not np.all(
            np.isfinite(physical_value)
        ):
            raise ValueError(
                f"constraint {constraint.constraint_id} must return shape "
                f"{(constraint.size,)}"
            )
        return physical_value / constraint.scale_to_physical

    @staticmethod
    def _constraint_jacobian_dimensionless(
        constraint: EqualityConstraint,
        variable_scales: FloatArray,
        dimensionless_x: FloatArray,
    ) -> FloatArray:
        physical_x = dimensionless_x * variable_scales
        physical_jacobian = np.asarray(
            constraint.jacobian(physical_x), dtype=np.float64
        )
        expected = (constraint.size, len(variable_scales))
        if physical_jacobian.shape != expected or not np.all(
            np.isfinite(physical_jacobian)
        ):
            raise ValueError(
                f"constraint {constraint.constraint_id} Jacobian must have shape "
                f"{expected}"
            )
        return (
            physical_jacobian
            * variable_scales[None, :]
            / constraint.scale_to_physical[:, None]
        )

    def _iteration_diagnostic(
        self,
        problem: ConstrainedProblem,
        dimensionless_x: FloatArray,
        state: Any,
        previous: _CallbackState,
    ) -> IterationDiagnostic:
        objective = self._objective_dimensionless(problem, dimensionless_x)
        gradient = self._gradient_dimensionless(problem, dimensionless_x)
        multipliers = self._extract_dimensionless_multipliers(problem, state)
        kkt = self._kkt_vector(problem, dimensionless_x, gradient, multipliers)
        residual_norm = self._constraint_inf_norm(problem, dimensionless_x)
        return IterationDiagnostic(
            iteration=int(getattr(state, "nit", getattr(state, "niter", 0))),
            objective_dimensionless=objective,
            objective_change=objective - previous.previous_objective,
            objective_gradient_inf_norm=_inf_norm(gradient),
            lagrangian_kkt_inf_norm=_inf_norm(kkt),
            equality_constraint_inf_norm=residual_norm,
            step_norm=float(np.linalg.norm(dimensionless_x - previous.previous_x)),
            trust_radius=_optional_float(getattr(state, "tr_radius", None)),
        )

    def _normalized_result(
        self,
        problem: ConstrainedProblem,
        settings: SolverSettings,
        counter: _EvaluationCounter,
        history: list[IterationDiagnostic],
        caught_messages: tuple[str, ...],
        callback_state: _CallbackState,
        raw_result: Any,
    ) -> SolverResult:
        dimensionless_solution = np.asarray(raw_result.x, dtype=np.float64)
        physical_solution = dimensionless_solution * problem.variables.scale_to_physical
        gradient = self._gradient_dimensionless(problem, dimensionless_solution)
        multipliers_dimensionless = self._extract_dimensionless_multipliers(
            problem, raw_result
        )
        kkt = self._kkt_vector(
            problem, dimensionless_solution, gradient, multipliers_dimensionless
        )
        residuals = self._constraint_residuals(problem, dimensionless_solution)
        multiplier_estimates = self._multiplier_estimates(
            problem, multipliers_dimensionless
        )
        residual_norm = max((item.inf_norm for item in residuals), default=0.0)
        raw_status = _optional_int(getattr(raw_result, "status", None))
        raw_message = str(getattr(raw_result, "message", ""))
        backend_success = bool(getattr(raw_result, "success", False))
        termination = BackendTermination(
            category=_termination_category(
                raw_status,
                backend_success,
                residual_norm,
                settings.tolerances.constraint,
                raw_message,
            ),
            backend_success=backend_success,
            raw_status=raw_status,
            raw_message=raw_message,
        )
        admissible, admissibility_messages = self._admissibility(
            problem, physical_solution
        )
        provenance = self._result_provenance(problem, settings)
        return SolverResult(
            problem_id=problem.problem_id,
            specification=problem.specification(settings),
            backend=self._backend_information(problem, settings),
            termination=termination,
            solution_physical=physical_solution,
            solution_dimensionless=dimensionless_solution,
            objective_physical=float(problem.objective.value(physical_solution)),
            objective_dimensionless=self._objective_dimensionless(
                problem, dimensionless_solution
            ),
            objective_gradient_inf_norm=_inf_norm(gradient),
            lagrangian_kkt_inf_norm=_inf_norm(kkt),
            equality_constraint_inf_norm=residual_norm,
            step_norm=callback_state.last_step_norm,
            objective_change=callback_state.last_objective_change,
            iteration_count=int(
                getattr(raw_result, "nit", getattr(raw_result, "niter", 0))
            ),
            evaluations=counter.snapshot(),
            constraint_residuals=residuals,
            multiplier_estimates=multiplier_estimates,
            iteration_history=tuple(history),
            numerical_warnings=_unique_messages(caught_messages),
            admissible=admissible,
            admissibility_messages=admissibility_messages,
            provenance=provenance,
        )

    def _failure_result(
        self,
        problem: ConstrainedProblem,
        settings: SolverSettings,
        counter: _EvaluationCounter,
        history: list[IterationDiagnostic],
        caught_messages: tuple[str, ...],
        callback_state: _CallbackState,
    ) -> SolverResult:
        dimensionless = problem.variables.initial_dimensionless
        physical = problem.variables.initial_values
        residuals = self._constraint_residuals(problem, dimensionless)
        residual_norm = max((item.inf_norm for item in residuals), default=0.0)
        admissible, admissibility_messages = self._admissibility(problem, physical)
        raw_message = caught_messages[-1] if caught_messages else "backend exception"
        return SolverResult(
            problem_id=problem.problem_id,
            specification=problem.specification(settings),
            backend=self._backend_information(problem, settings),
            termination=BackendTermination(
                category=TerminationCategory.BACKEND_FAILURE,
                backend_success=False,
                raw_status=None,
                raw_message=raw_message,
            ),
            solution_physical=physical,
            solution_dimensionless=dimensionless,
            objective_physical=float(problem.objective.value(physical)),
            objective_dimensionless=self._objective_dimensionless(
                problem, dimensionless
            ),
            objective_gradient_inf_norm=np.finfo(np.float64).max,
            lagrangian_kkt_inf_norm=np.finfo(np.float64).max,
            equality_constraint_inf_norm=residual_norm,
            step_norm=callback_state.last_step_norm,
            objective_change=callback_state.last_objective_change,
            iteration_count=len(history),
            evaluations=counter.snapshot(),
            constraint_residuals=residuals,
            multiplier_estimates=(),
            iteration_history=tuple(history),
            numerical_warnings=_unique_messages(caught_messages),
            admissible=admissible,
            admissibility_messages=admissibility_messages,
            provenance=self._result_provenance(problem, settings),
        )

    def _constraint_residuals(
        self, problem: ConstrainedProblem, dimensionless_x: FloatArray
    ) -> tuple[ConstraintResidual, ...]:
        physical_x = dimensionless_x * problem.variables.scale_to_physical
        result: list[ConstraintResidual] = []
        for constraint in problem.equality_constraints:
            physical = np.asarray(constraint.value(physical_x), dtype=np.float64)
            dimensionless = physical / constraint.scale_to_physical
            result.append(
                ConstraintResidual(
                    constraint_id=constraint.constraint_id,
                    physical_values=physical,
                    dimensionless_values=dimensionless,
                    inf_norm=_inf_norm(dimensionless),
                )
            )
        return tuple(result)

    def _multiplier_estimates(
        self,
        problem: ConstrainedProblem,
        multipliers_dimensionless: tuple[FloatArray, ...],
    ) -> tuple[MultiplierEstimate, ...]:
        result: list[MultiplierEstimate] = []
        objective_scale = problem.objective.scale_to_physical
        for constraint, dimensionless in zip(
            problem.equality_constraints, multipliers_dimensionless, strict=True
        ):
            result.append(
                MultiplierEstimate(
                    constraint_id=constraint.constraint_id,
                    physical_values=(
                        objective_scale * dimensionless / constraint.scale_to_physical
                    ),
                    dimensionless_values=dimensionless,
                )
            )
        return tuple(result)

    def _extract_dimensionless_multipliers(
        self, problem: ConstrainedProblem, state: Any
    ) -> tuple[FloatArray, ...]:
        raw = getattr(state, "v", None)
        if raw is None:
            return tuple(
                np.zeros(item.size, dtype=np.float64)
                for item in problem.equality_constraints
            )
        values = tuple(np.asarray(item, dtype=np.float64).reshape(-1) for item in raw)
        if len(values) < len(problem.equality_constraints):
            return tuple(
                np.zeros(item.size, dtype=np.float64)
                for item in problem.equality_constraints
            )
        return values[: len(problem.equality_constraints)]

    def _kkt_vector(
        self,
        problem: ConstrainedProblem,
        dimensionless_x: FloatArray,
        dimensionless_gradient: FloatArray,
        multipliers: tuple[FloatArray, ...],
    ) -> FloatArray:
        result = np.array(dimensionless_gradient, copy=True)
        for constraint, multiplier in zip(
            problem.equality_constraints, multipliers, strict=True
        ):
            jacobian = self._constraint_jacobian_dimensionless(
                constraint,
                problem.variables.scale_to_physical,
                dimensionless_x,
            )
            result += jacobian.T @ multiplier
        return result

    def _constraint_inf_norm(
        self, problem: ConstrainedProblem, dimensionless_x: FloatArray
    ) -> float:
        return max(
            (
                _inf_norm(
                    self._constraint_dimensionless(
                        item,
                        problem.variables.scale_to_physical,
                        dimensionless_x,
                    )
                )
                for item in problem.equality_constraints
            ),
            default=0.0,
        )

    def _backend_information(
        self, problem: ConstrainedProblem, settings: SolverSettings
    ) -> BackendInformation:
        hessian_parts = [f"objective:{problem.objective.hessian_mode.value}"]
        hessian_parts.extend(
            f"{item.constraint_id}:{item.hessian_mode.value}"
            for item in problem.equality_constraints
        )
        return BackendInformation(
            capability_id=self.capability_id,
            backend_name="scipy.optimize.minimize(method='trust-constr')",
            backend_version=scipy.__version__,
            settings=self.scipy_options(settings),
            hessian_strategy=";".join(hessian_parts),
        )

    def _result_provenance(
        self, problem: ConstrainedProblem, settings: SolverSettings
    ) -> tuple[Provenance, ...]:
        adapter = Provenance(
            activity="equality-constrained numerical optimization",
            implementation=(
                "openphenomena.optimization.scipy_trust_constr.ScipyTrustConstrAdapter"
            ),
            implementation_version="1",
            source_ids=(problem.problem_id,),
            parameters={
                "backend": "scipy.optimize.trust-constr",
                "hessian_strategy": self._backend_information(
                    problem, settings
                ).hessian_strategy,
                "scipy_version": scipy.__version__,
                "settings": dict(settings.as_mapping()),
            },
            citations=(
                "https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html",
            ),
        )
        return (*problem.provenance, adapter)

    @staticmethod
    def _admissibility(
        problem: ConstrainedProblem, physical_x: FloatArray
    ) -> tuple[bool, tuple[str, ...]]:
        if problem.admissibility is None:
            return True, ("no additional admissibility predicate supplied",)
        admissible, messages = problem.admissibility(physical_x)
        return bool(admissible), tuple(messages)


def _validate_square_matrix(values: FloatArray, size: int, name: str) -> None:
    if values.shape != (size, size) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite {(size, size)} matrix")


def _inf_norm(values: FloatArray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _unique_messages(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in values if item))


def _termination_category(
    status: int | None,
    success: bool,
    constraint_inf_norm: float,
    constraint_tolerance: float,
    message: str,
) -> TerminationCategory:
    if status == 0:
        return TerminationCategory.ITERATION_LIMIT
    if constraint_inf_norm > constraint_tolerance and status in (0, 2, 4):
        return TerminationCategory.INFEASIBLE
    if status == 1 and (success or constraint_inf_norm <= constraint_tolerance):
        return TerminationCategory.CONVERGED_OPTIMALITY
    if status == 2 and (success or constraint_inf_norm <= constraint_tolerance):
        return TerminationCategory.CONVERGED_STEP
    if status == 3:
        return TerminationCategory.CALLBACK_STOP
    if status == 4:
        return TerminationCategory.INFEASIBLE
    lowered = message.lower()
    if "nan" in lowered or "singular" in lowered or "factorization" in lowered:
        return TerminationCategory.NUMERICAL_FAILURE
    if not success:
        return TerminationCategory.BACKEND_FAILURE
    return TerminationCategory.UNKNOWN
