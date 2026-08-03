"""Backend-neutral contracts for equality-constrained numerical optimization."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np

from openphenomena.data import (
    EvidenceRecord,
    Fidelity,
    FloatArray,
    Provenance,
    ValidationStatus,
)

ScalarFunction = Callable[[FloatArray], float]
GradientFunction = Callable[[FloatArray], FloatArray]
VectorFunction = Callable[[FloatArray], FloatArray]
JacobianFunction = Callable[[FloatArray], FloatArray]
HessianFunction = Callable[[FloatArray], FloatArray]
HessianVectorProduct = Callable[[FloatArray, FloatArray], FloatArray]
ConstraintHessianFunction = Callable[[FloatArray, FloatArray], FloatArray]
AdmissibilityFunction = Callable[[FloatArray], tuple[bool, tuple[str, ...]]]


class HessianMode(StrEnum):
    """Explicit second-derivative strategy requested from a backend."""

    EXACT = "exact"
    HESSIAN_VECTOR_PRODUCT = "hessian_vector_product"
    BFGS = "bfgs"


class TerminationCategory(StrEnum):
    """Backend-independent termination categories."""

    CONVERGED_OPTIMALITY = "converged_optimality"
    CONVERGED_STEP = "converged_step"
    ITERATION_LIMIT = "iteration_limit"
    CALLBACK_STOP = "callback_stop"
    INFEASIBLE = "infeasible"
    NUMERICAL_FAILURE = "numerical_failure"
    BACKEND_FAILURE = "backend_failure"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class OptimizationVariables:
    """Physical initial variables and diagonal scales used by a backend."""

    names: tuple[str, ...]
    initial_values: FloatArray
    scale_to_physical: FloatArray
    units: tuple[str, ...]

    def __post_init__(self) -> None:
        initial = _readonly_vector(self.initial_values, "initial values")
        scales = _readonly_vector(self.scale_to_physical, "variable scales")
        if initial.shape != scales.shape:
            raise ValueError("initial values and variable scales must have equal shape")
        if len(self.names) != len(initial) or len(self.units) != len(initial):
            raise ValueError("variable names and units must match variable count")
        if len(set(self.names)) != len(self.names) or any(
            not name for name in self.names
        ):
            raise ValueError("variable names must be nonempty and unique")
        if any(not unit for unit in self.units):
            raise ValueError("variable units cannot be empty")
        if np.any(scales <= 0.0):
            raise ValueError("variable scales must be positive")
        object.__setattr__(self, "initial_values", initial)
        object.__setattr__(self, "scale_to_physical", scales)

    @property
    def initial_dimensionless(self) -> FloatArray:
        return _readonly_vector(
            self.initial_values / self.scale_to_physical,
            "dimensionless initial values",
        )


@dataclass(frozen=True, slots=True)
class ScalarObjective:
    """Physical scalar objective with analytic first derivative."""

    objective_id: str
    value: ScalarFunction
    gradient: GradientFunction
    scale_to_physical: float
    unit: str
    hessian_mode: HessianMode = HessianMode.BFGS
    hessian: HessianFunction | None = None
    hessian_vector_product: HessianVectorProduct | None = None

    def __post_init__(self) -> None:
        _validate_namespaced(self.objective_id, "objective ID")
        _validate_positive_finite(self.scale_to_physical, "objective scale")
        if not self.unit:
            raise ValueError("objective unit cannot be empty")
        if self.hessian_mode is HessianMode.EXACT:
            if self.hessian is None or self.hessian_vector_product is not None:
                raise ValueError("exact Hessian mode requires only a Hessian callable")
        elif self.hessian_mode is HessianMode.HESSIAN_VECTOR_PRODUCT:
            if self.hessian_vector_product is None or self.hessian is not None:
                raise ValueError("Hessian-vector mode requires only an HVP callable")
        elif self.hessian is not None or self.hessian_vector_product is not None:
            raise ValueError("BFGS mode cannot include exact second derivatives")


@dataclass(frozen=True, slots=True)
class EqualityConstraint:
    """Physical equality residual with analytic Jacobian."""

    constraint_id: str
    size: int
    value: VectorFunction
    jacobian: JacobianFunction
    scale_to_physical: FloatArray
    units: tuple[str, ...]
    hessian_mode: HessianMode = HessianMode.BFGS
    hessian: ConstraintHessianFunction | None = None

    def __post_init__(self) -> None:
        _validate_namespaced(self.constraint_id, "constraint ID")
        if self.size <= 0:
            raise ValueError("equality constraint size must be positive")
        scales = _readonly_vector(self.scale_to_physical, "constraint scales")
        if scales.shape != (self.size,) or len(self.units) != self.size:
            raise ValueError("constraint scales and units must match declared size")
        if np.any(scales <= 0.0) or any(not unit for unit in self.units):
            raise ValueError("constraint scales must be positive and units nonempty")
        if self.hessian_mode is HessianMode.EXACT and self.hessian is None:
            raise ValueError("exact constraint Hessian mode requires a callable")
        if self.hessian_mode is not HessianMode.EXACT and self.hessian is not None:
            raise ValueError("only exact constraint Hessian mode accepts a callable")
        if self.hessian_mode is HessianMode.HESSIAN_VECTOR_PRODUCT:
            raise ValueError("constraint Hessian-vector mode is not in this protocol")
        object.__setattr__(self, "scale_to_physical", scales)


@dataclass(frozen=True, slots=True)
class SolverTolerances:
    """Dimensionless backend and independent scientific tolerances."""

    optimality: float = 1.0e-9
    step: float = 1.0e-12
    constraint: float = 1.0e-9
    barrier: float = 1.0e-12

    def __post_init__(self) -> None:
        for name in ("optimality", "step", "constraint", "barrier"):
            _validate_positive_finite(getattr(self, name), f"{name} tolerance")


@dataclass(frozen=True, slots=True)
class SolverSettings:
    """Backend-neutral deterministic solver configuration."""

    tolerances: SolverTolerances = field(default_factory=SolverTolerances)
    max_iterations: int = 500
    history_stride: int = 1
    initial_trust_radius: float = 1.0
    initial_constraint_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.max_iterations <= 0 or self.history_stride <= 0:
            raise ValueError("iteration limit and history stride must be positive")
        _validate_positive_finite(self.initial_trust_radius, "initial trust radius")
        _validate_positive_finite(
            self.initial_constraint_penalty, "initial constraint penalty"
        )

    def as_mapping(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "barrier_tolerance": self.tolerances.barrier,
                "constraint_tolerance": self.tolerances.constraint,
                "history_stride": self.history_stride,
                "initial_constraint_penalty": self.initial_constraint_penalty,
                "initial_trust_radius": self.initial_trust_radius,
                "max_iterations": self.max_iterations,
                "optimality_tolerance": self.tolerances.optimality,
                "step_tolerance": self.tolerances.step,
            }
        )


@dataclass(frozen=True, slots=True)
class ConstrainedProblem:
    """Backend-neutral equality-constrained physical problem."""

    problem_id: str
    variables: OptimizationVariables
    objective: ScalarObjective
    equality_constraints: tuple[EqualityConstraint, ...]
    provenance: tuple[Provenance, ...]
    admissibility: AdmissibilityFunction | None = None

    def __post_init__(self) -> None:
        _validate_namespaced(self.problem_id, "problem ID")
        if not self.equality_constraints:
            raise ValueError("at least one equality constraint is required")
        identifiers = tuple(item.constraint_id for item in self.equality_constraints)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("constraint IDs must be unique")
        if not self.provenance:
            raise ValueError("optimization problem requires provenance")
        object.__setattr__(self, "provenance", tuple(self.provenance))

    def specification(self, settings: SolverSettings) -> SolverSpecification:
        return SolverSpecification(
            problem_id=self.problem_id,
            objective_id=self.objective.objective_id,
            objective_scale_to_physical=self.objective.scale_to_physical,
            objective_unit=self.objective.unit,
            variable_names=self.variables.names,
            initial_values=self.variables.initial_values,
            variable_scales_to_physical=self.variables.scale_to_physical,
            variable_units=self.variables.units,
            constraints=tuple(
                ConstraintSpecification(
                    constraint_id=item.constraint_id,
                    size=item.size,
                    scales_to_physical=item.scale_to_physical,
                    units=item.units,
                )
                for item in self.equality_constraints
            ),
            settings=settings,
        )


@dataclass(frozen=True, slots=True)
class ConstraintSpecification:
    constraint_id: str
    size: int
    scales_to_physical: FloatArray
    units: tuple[str, ...]

    def __post_init__(self) -> None:
        scales = _readonly_vector(self.scales_to_physical, "constraint scales")
        if scales.shape != (self.size,) or len(self.units) != self.size:
            raise ValueError("serialized constraint specification has invalid size")
        object.__setattr__(self, "scales_to_physical", scales)


@dataclass(frozen=True, slots=True)
class SolverSpecification:
    """Serializable, callable-free snapshot of a constrained problem."""

    problem_id: str
    objective_id: str
    objective_scale_to_physical: float
    objective_unit: str
    variable_names: tuple[str, ...]
    initial_values: FloatArray
    variable_scales_to_physical: FloatArray
    variable_units: tuple[str, ...]
    constraints: tuple[ConstraintSpecification, ...]
    settings: SolverSettings

    def __post_init__(self) -> None:
        initial = _readonly_vector(self.initial_values, "initial values")
        scales = _readonly_vector(self.variable_scales_to_physical, "variable scales")
        if initial.shape != scales.shape or len(self.variable_names) != len(initial):
            raise ValueError("serialized variable specification has invalid size")
        object.__setattr__(self, "initial_values", initial)
        object.__setattr__(self, "variable_scales_to_physical", scales)

    def as_mapping(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "constraints": [
                    {
                        "constraint_id": item.constraint_id,
                        "scales_to_physical": item.scales_to_physical.tolist(),
                        "size": item.size,
                        "units": list(item.units),
                    }
                    for item in self.constraints
                ],
                "initial_values": self.initial_values.tolist(),
                "objective_id": self.objective_id,
                "objective_scale_to_physical": self.objective_scale_to_physical,
                "objective_unit": self.objective_unit,
                "problem_id": self.problem_id,
                "settings": _plain(self.settings.as_mapping()),
                "variable_names": list(self.variable_names),
                "variable_scales_to_physical": (
                    self.variable_scales_to_physical.tolist()
                ),
                "variable_units": list(self.variable_units),
            }
        )


@dataclass(frozen=True, slots=True)
class IterationDiagnostic:
    """Deterministic reduced diagnostic for one accepted backend iterate."""

    iteration: int
    objective_dimensionless: float
    objective_change: float
    objective_gradient_inf_norm: float
    lagrangian_kkt_inf_norm: float
    equality_constraint_inf_norm: float
    step_norm: float
    trust_radius: float | None = None

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError("iteration must be nonnegative")
        for name in (
            "objective_dimensionless",
            "objective_change",
            "objective_gradient_inf_norm",
            "lagrangian_kkt_inf_norm",
            "equality_constraint_inf_norm",
            "step_norm",
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.trust_radius is not None and not np.isfinite(self.trust_radius):
            raise ValueError("trust radius must be finite when present")


@runtime_checkable
class DiagnosticCallback(Protocol):
    """Backend-neutral callback; true requests controlled termination."""

    def __call__(self, diagnostic: IterationDiagnostic) -> bool: ...


@dataclass(frozen=True, slots=True)
class BackendTermination:
    category: TerminationCategory
    backend_success: bool
    raw_status: int | None
    raw_message: str


@dataclass(frozen=True, slots=True)
class EvaluationCounts:
    objective: int
    objective_gradient: int
    objective_hessian: int
    constraints: int
    constraint_jacobians: int
    constraint_hessians: int

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.as_tuple()):
            raise ValueError("evaluation counts must be nonnegative")

    def as_tuple(self) -> tuple[int, ...]:
        return (
            self.objective,
            self.objective_gradient,
            self.objective_hessian,
            self.constraints,
            self.constraint_jacobians,
            self.constraint_hessians,
        )


@dataclass(frozen=True, slots=True)
class ConstraintResidual:
    constraint_id: str
    physical_values: FloatArray
    dimensionless_values: FloatArray
    inf_norm: float

    def __post_init__(self) -> None:
        physical = _readonly_vector(self.physical_values, "physical residual")
        dimensionless = _readonly_vector(
            self.dimensionless_values, "dimensionless residual"
        )
        if physical.shape != dimensionless.shape:
            raise ValueError("physical and dimensionless residual shapes must match")
        if not np.isfinite(self.inf_norm) or self.inf_norm < 0.0:
            raise ValueError("constraint residual norm must be finite and nonnegative")
        object.__setattr__(self, "physical_values", physical)
        object.__setattr__(self, "dimensionless_values", dimensionless)


@dataclass(frozen=True, slots=True)
class MultiplierEstimate:
    """Multiplier for ``L=f+lambda^T c`` in physical and scaled systems."""

    constraint_id: str
    physical_values: FloatArray
    dimensionless_values: FloatArray
    sign_convention: str = "L_physical = f_physical + lambda^T c_physical"

    def __post_init__(self) -> None:
        physical = _readonly_vector(self.physical_values, "physical multipliers")
        dimensionless = _readonly_vector(
            self.dimensionless_values, "dimensionless multipliers"
        )
        if physical.shape != dimensionless.shape:
            raise ValueError("physical and dimensionless multiplier shapes must match")
        if not self.sign_convention:
            raise ValueError("multiplier sign convention is required")
        object.__setattr__(self, "physical_values", physical)
        object.__setattr__(self, "dimensionless_values", dimensionless)


@dataclass(frozen=True, slots=True)
class BackendInformation:
    capability_id: str
    backend_name: str
    backend_version: str
    settings: Mapping[str, object]
    hessian_strategy: str

    def __post_init__(self) -> None:
        _validate_namespaced(self.capability_id, "backend capability ID")
        if (
            not self.backend_name
            or not self.backend_version
            or not self.hessian_strategy
        ):
            raise ValueError(
                "complete backend identity and Hessian strategy are required"
            )
        object.__setattr__(self, "settings", _freeze_mapping(self.settings))


@dataclass(frozen=True, slots=True)
class ScientificAcceptance:
    policy_id: str
    acceptable: bool
    reasons: tuple[str, ...]
    thresholds: Mapping[str, float]

    def __post_init__(self) -> None:
        _validate_namespaced(self.policy_id, "acceptance policy ID")
        if not self.reasons:
            raise ValueError("scientific acceptance requires at least one reason")
        if any(
            not np.isfinite(value) or value < 0.0 for value in self.thresholds.values()
        ):
            raise ValueError("scientific acceptance thresholds must be nonnegative")
        object.__setattr__(self, "thresholds", _freeze_mapping(self.thresholds))


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Immutable backend-neutral solution candidate and numerical diagnostics."""

    problem_id: str
    specification: SolverSpecification
    backend: BackendInformation
    termination: BackendTermination
    solution_physical: FloatArray
    solution_dimensionless: FloatArray
    objective_physical: float
    objective_dimensionless: float
    objective_gradient_inf_norm: float
    lagrangian_kkt_inf_norm: float
    equality_constraint_inf_norm: float
    step_norm: float
    objective_change: float
    iteration_count: int
    evaluations: EvaluationCounts
    constraint_residuals: tuple[ConstraintResidual, ...]
    multiplier_estimates: tuple[MultiplierEstimate, ...]
    iteration_history: tuple[IterationDiagnostic, ...]
    numerical_warnings: tuple[str, ...]
    admissible: bool
    admissibility_messages: tuple[str, ...]
    provenance: tuple[Provenance, ...]
    fidelity: Fidelity = Fidelity.ENGINEERING_APPROXIMATION
    validation_status: ValidationStatus = ValidationStatus.VERIFIED
    scientific_acceptance: ScientificAcceptance | None = None

    def __post_init__(self) -> None:
        _validate_namespaced(self.problem_id, "problem ID")
        if self.specification.problem_id != self.problem_id:
            raise ValueError("result and specification problem IDs must match")
        physical = _readonly_vector(self.solution_physical, "physical solution")
        dimensionless = _readonly_vector(
            self.solution_dimensionless, "dimensionless solution"
        )
        if physical.shape != dimensionless.shape:
            raise ValueError("physical and dimensionless solutions must match shape")
        for name in (
            "objective_physical",
            "objective_dimensionless",
            "objective_gradient_inf_norm",
            "lagrangian_kkt_inf_norm",
            "equality_constraint_inf_norm",
            "step_norm",
            "objective_change",
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.iteration_count < 0 or not self.provenance:
            raise ValueError("result requires nonnegative iterations and provenance")
        object.__setattr__(self, "solution_physical", physical)
        object.__setattr__(self, "solution_dimensionless", dimensionless)
        object.__setattr__(self, "provenance", tuple(self.provenance))

    @property
    def backend_terminated_successfully(self) -> bool:
        """Return the backend library's raw success flag."""

        return self.termination.backend_success

    @property
    def backend_converged(self) -> bool:
        """Return whether normalized termination represents convergence.

        Some SciPy releases report a false raw ``success`` flag for an ``xtol``
        or ``gtol`` termination even when independently recomputed equality
        residuals satisfy the configured tolerance. The normalized category
        preserves that distinction while never treating iteration limits or
        failures as convergence.
        """

        return self.termination.category in {
            TerminationCategory.CONVERGED_OPTIMALITY,
            TerminationCategory.CONVERGED_STEP,
        }

    @property
    def scientifically_acceptable(self) -> bool:
        return bool(
            self.scientific_acceptance is not None
            and self.scientific_acceptance.acceptable
        )

    def as_metadata(self) -> Mapping[str, object]:
        """Return deterministic JSON-compatible metadata for a scientific Run."""

        return MappingProxyType(
            {
                "admissibility_messages": list(self.admissibility_messages),
                "admissible": self.admissible,
                "backend": {
                    "backend_name": self.backend.backend_name,
                    "backend_version": self.backend.backend_version,
                    "capability_id": self.backend.capability_id,
                    "hessian_strategy": self.backend.hessian_strategy,
                    "settings": _plain(self.backend.settings),
                },
                "constraint_residuals": [
                    {
                        "constraint_id": item.constraint_id,
                        "dimensionless_values": item.dimensionless_values.tolist(),
                        "inf_norm": item.inf_norm,
                        "physical_values": item.physical_values.tolist(),
                    }
                    for item in self.constraint_residuals
                ],
                "equality_constraint_inf_norm": self.equality_constraint_inf_norm,
                "evaluations": {
                    "constraint_hessians": self.evaluations.constraint_hessians,
                    "constraint_jacobians": self.evaluations.constraint_jacobians,
                    "constraints": self.evaluations.constraints,
                    "objective": self.evaluations.objective,
                    "objective_gradient": self.evaluations.objective_gradient,
                    "objective_hessian": self.evaluations.objective_hessian,
                },
                "fidelity": self.fidelity.value,
                "iteration_count": self.iteration_count,
                "iteration_history": [
                    {
                        "equality_constraint_inf_norm": (
                            item.equality_constraint_inf_norm
                        ),
                        "iteration": item.iteration,
                        "lagrangian_kkt_inf_norm": item.lagrangian_kkt_inf_norm,
                        "objective_change": item.objective_change,
                        "objective_dimensionless": item.objective_dimensionless,
                        "objective_gradient_inf_norm": item.objective_gradient_inf_norm,
                        "step_norm": item.step_norm,
                        "trust_radius": item.trust_radius,
                    }
                    for item in self.iteration_history
                ],
                "lagrangian_kkt_inf_norm": self.lagrangian_kkt_inf_norm,
                "multiplier_estimates": [
                    {
                        "constraint_id": item.constraint_id,
                        "dimensionless_values": item.dimensionless_values.tolist(),
                        "physical_values": item.physical_values.tolist(),
                        "sign_convention": item.sign_convention,
                    }
                    for item in self.multiplier_estimates
                ],
                "numerical_warnings": list(self.numerical_warnings),
                "objective_change": self.objective_change,
                "objective_dimensionless": self.objective_dimensionless,
                "objective_gradient_inf_norm": self.objective_gradient_inf_norm,
                "objective_physical": self.objective_physical,
                "problem_id": self.problem_id,
                "provenance": [_provenance_mapping(item) for item in self.provenance],
                "scientific_acceptance": _acceptance_mapping(
                    self.scientific_acceptance
                ),
                "specification": _plain(self.specification.as_mapping()),
                "solution_dimensionless": self.solution_dimensionless.tolist(),
                "solution_physical": self.solution_physical.tolist(),
                "step_norm": self.step_norm,
                "termination": {
                    "backend_success": self.termination.backend_success,
                    "category": self.termination.category.value,
                    "raw_message": self.termination.raw_message,
                    "raw_status": self.termination.raw_status,
                },
                "validation_status": self.validation_status.value,
            }
        )

    def acceptance_evidence(
        self, evidence_id: str, artifact_references: tuple[str, ...]
    ) -> EvidenceRecord:
        """Build evidence for a caller-supplied scientific acceptance decision."""

        if self.scientific_acceptance is None:
            raise ValueError("scientific acceptance policy has not been evaluated")
        thresholds = self.scientific_acceptance.thresholds
        candidates = (
            (self.lagrangian_kkt_inf_norm, thresholds.get("kkt_inf_norm")),
            (
                self.equality_constraint_inf_norm,
                thresholds.get("constraint_inf_norm"),
            ),
            (self.step_norm, thresholds.get("step_norm")),
        )
        normalized_errors = tuple(
            value / threshold
            for value, threshold in candidates
            if threshold is not None and threshold > 0.0
        )
        measured_error = max(normalized_errors, default=0.0)
        if not self.scientific_acceptance.acceptable:
            measured_error = max(measured_error, 2.0)
        return EvidenceRecord(
            evidence_id=evidence_id,
            evidence_type="numerical_verification",
            quantity_of_interest="scientific constrained-solver acceptance",
            conditions={
                "backend_capability_id": self.backend.capability_id,
                "policy_id": self.scientific_acceptance.policy_id,
                "termination_category": self.termination.category.value,
            },
            tolerance=1.0,
            measured_error=measured_error,
            passed=self.scientific_acceptance.acceptable,
            implementation="openphenomena.optimization.contracts.SolverResult",
            implementation_version="1",
            artifact_references=artifact_references,
            notes="; ".join(self.scientific_acceptance.reasons),
        )


@runtime_checkable
class AcceptancePolicy(Protocol):
    """Caller-owned scientific acceptance policy."""

    def evaluate(self, result: SolverResult) -> ScientificAcceptance: ...


@runtime_checkable
class ConstrainedSolverBackend(Protocol):
    """Common protocol for SciPy, PETSc/TAO, and future solver adapters."""

    @property
    def capability_id(self) -> str: ...

    def solve(
        self,
        problem: ConstrainedProblem,
        settings: SolverSettings,
        callback: DiagnosticCallback | None = None,
        acceptance_policy: AcceptancePolicy | None = None,
    ) -> SolverResult: ...


@dataclass(frozen=True, slots=True)
class ThresholdAcceptancePolicy:
    """Simple independent acceptance policy for verified numerical problems."""

    policy_id: str
    kkt_inf_norm: float
    constraint_inf_norm: float
    step_norm: float
    require_backend_success: bool = True
    require_admissible: bool = True

    def __post_init__(self) -> None:
        _validate_namespaced(self.policy_id, "acceptance policy ID")
        for name in ("kkt_inf_norm", "constraint_inf_norm", "step_norm"):
            _validate_positive_finite(getattr(self, name), name)

    def evaluate(self, result: SolverResult) -> ScientificAcceptance:
        checks = (
            (
                not self.require_backend_success
                or result.backend_terminated_successfully,
                "backend termination",
            ),
            (result.lagrangian_kkt_inf_norm <= self.kkt_inf_norm, "KKT residual"),
            (
                result.equality_constraint_inf_norm <= self.constraint_inf_norm,
                "equality residual",
            ),
            (result.step_norm <= self.step_norm, "step norm"),
            (not self.require_admissible or result.admissible, "admissibility"),
        )
        failed = tuple(label for passed, label in checks if not passed)
        reasons = (
            ("all independently specified numerical criteria passed",)
            if not failed
            else tuple(f"failed: {label}" for label in failed)
        )
        return ScientificAcceptance(
            policy_id=self.policy_id,
            acceptable=not failed,
            reasons=reasons,
            thresholds={
                "constraint_inf_norm": self.constraint_inf_norm,
                "kkt_inf_norm": self.kkt_inf_norm,
                "step_norm": self.step_norm,
            },
        )


def _readonly_vector(values: FloatArray, name: str) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    result.flags.writeable = False
    return result


def _validate_namespaced(value: str, name: str) -> None:
    if not value or "." not in value:
        raise ValueError(f"{name} must be namespaced")


def _validate_positive_finite(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _freeze_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({str(key): _freeze(item) for key, item in values.items()})


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _provenance_mapping(item: Provenance) -> Mapping[str, object]:
    return {
        "activity": item.activity,
        "citations": list(item.citations),
        "implementation": item.implementation,
        "implementation_version": item.implementation_version,
        "parameters": _plain(item.parameters),
        "source_ids": list(item.source_ids),
    }


def _acceptance_mapping(item: ScientificAcceptance | None) -> object:
    if item is None:
        return None
    return {
        "acceptable": item.acceptable,
        "policy_id": item.policy_id,
        "reasons": list(item.reasons),
        "thresholds": _plain(item.thresholds),
    }
