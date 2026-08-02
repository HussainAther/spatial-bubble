"""Immutable, backend-neutral scientific data model.

The classes in this module are the authoritative in-memory representation.
Export formats and visualization packages are adapters over these objects.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
NumericArray = npt.NDArray[np.generic]
Dimension = tuple[float, float, float, float, float, float, float]


class FieldAssociation(StrEnum):
    """Topological location at which the leading field axis is sampled."""

    VERTEX = "vertex"
    FACE = "face"
    GLOBAL = "global"


class Fidelity(StrEnum):
    """Scientific status required by the architecture baseline."""

    PHYSICALLY_VALIDATED = "PV"
    ENGINEERING_APPROXIMATION = "EA"
    VISUALIZATION_ONLY = "VO"
    SPECULATIVE_FUTURE = "SF"


class ValidationStatus(StrEnum):
    """Relationship between a field/model and available evidence."""

    VALIDATED = "validated"
    VERIFIED = "verified"
    UNVALIDATED = "unvalidated"
    NOT_APPLICABLE = "not_applicable"


class RunStatus(StrEnum):
    """Lifecycle state of an immutable run record."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Uncertainty:
    """Explicit uncertainty statement attached to a scientific field."""

    quantified: bool
    description: str
    standard_uncertainty: float | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if not self.description:
            raise ValueError("uncertainty description cannot be empty")
        if self.quantified and self.standard_uncertainty is None:
            raise ValueError("quantified uncertainty requires standard_uncertainty")
        if self.standard_uncertainty is not None and self.standard_uncertainty < 0.0:
            raise ValueError("standard_uncertainty must be nonnegative")


UNQUANTIFIED = Uncertainty(
    quantified=False,
    description="Uncertainty has not yet been quantified.",
)


@dataclass(frozen=True, slots=True)
class Provenance:
    """A generating activity and its traceable inputs."""

    activity: str
    implementation: str
    implementation_version: str
    source_ids: tuple[str, ...] = ()
    parameters: Mapping[str, object] = field(default_factory=dict)
    citations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.activity
            or not self.implementation
            or not self.implementation_version
        ):
            raise ValueError(
                "provenance activity, implementation, and version are required"
            )
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))


@dataclass(frozen=True, slots=True)
class FieldDescriptor:
    """Complete semantic and numerical description of a field."""

    semantic_id: str
    association: FieldAssociation
    unit: str
    unit_dimension: Dimension
    shape: tuple[int, ...]
    dtype: str
    coordinate_frame: str
    generating_model: str
    generating_implementation: str
    fidelity: Fidelity
    validation_status: ValidationStatus
    uncertainty: Uncertainty = UNQUANTIFIED
    description: str = ""
    component_names: tuple[str, ...] = ()
    coordinate_axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.semantic_id or "." not in self.semantic_id:
            raise ValueError("semantic_id must be a nonempty namespaced identifier")
        required_text = {
            "unit": self.unit,
            "coordinate_frame": self.coordinate_frame,
            "generating_model": self.generating_model,
            "generating_implementation": self.generating_implementation,
        }
        for name, value in required_text.items():
            if not value:
                raise ValueError(f"{name} cannot be empty")
        if len(self.unit_dimension) != 7:
            raise ValueError("unit_dimension must contain seven SI base exponents")
        if any(size < 0 for size in self.shape):
            raise ValueError("field shape entries must be nonnegative")
        if np.dtype(self.dtype).kind not in "biufc":
            raise TypeError("field dtype must be numerical")


@dataclass(frozen=True, slots=True)
class Field:
    """Immutable numerical values paired with their complete descriptor."""

    descriptor: FieldDescriptor
    values: NumericArray
    provenance: tuple[Provenance, ...]

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.dtype(self.descriptor.dtype), copy=True)
        if values.shape != self.descriptor.shape:
            raise ValueError(
                f"field {self.descriptor.semantic_id!r} has shape {values.shape}; "
                f"descriptor declares {self.descriptor.shape}"
            )
        if values.dtype.kind in "fc" and np.any(~np.isfinite(values)):
            raise ValueError(f"field {self.descriptor.semantic_id!r} must be finite")
        if not self.provenance:
            raise ValueError("every field requires at least one provenance record")
        values.flags.writeable = False
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class Domain:
    """Immutable triangular surface domain and its scientific fields."""

    domain_id: str
    kind: str
    coordinate_frame: str
    positions_m: FloatArray
    faces: IntArray
    fields: Mapping[str, Field]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positions = np.array(self.positions_m, dtype=np.float64, copy=True)
        faces = np.array(self.faces, dtype=np.int64, copy=True)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions_m must have shape (n_vertices, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (n_faces, 3)")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions_m must be finite")
        if faces.size and (np.min(faces) < 0 or np.max(faces) >= len(positions)):
            raise ValueError("faces contain an out-of-range vertex index")
        frozen_fields = dict(self.fields)
        for semantic_id, sampled_field in frozen_fields.items():
            if semantic_id != sampled_field.descriptor.semantic_id:
                raise ValueError("field mapping key must equal descriptor semantic_id")
            expected = {
                FieldAssociation.VERTEX: len(positions),
                FieldAssociation.FACE: len(faces),
            }.get(sampled_field.descriptor.association)
            if expected is not None and (
                not sampled_field.descriptor.shape
                or sampled_field.descriptor.shape[0] != expected
            ):
                raise ValueError(
                    f"field {semantic_id!r} has incompatible association cardinality"
                )
        positions.flags.writeable = False
        faces.flags.writeable = False
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "fields", MappingProxyType(frozen_fields))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class Frame:
    """A consistent collection of domains at one physical time/iteration."""

    frame_id: str
    time_s: float
    iteration: int
    domains: tuple[Domain, ...]

    def __post_init__(self) -> None:
        if not np.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("frame time must be finite and nonnegative")
        if self.iteration < 0 or not self.domains:
            raise ValueError(
                "frame requires nonnegative iteration and at least one domain"
            )


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Machine-readable verification or validation result."""

    evidence_id: str
    evidence_type: str
    quantity_of_interest: str
    conditions: Mapping[str, object]
    tolerance: float
    measured_error: float
    passed: bool
    implementation: str
    implementation_version: str
    artifact_references: tuple[str, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        if self.tolerance < 0.0 or self.measured_error < 0.0:
            raise ValueError(
                "evidence tolerance and measured error must be nonnegative"
            )
        if not self.evidence_id or not self.quantity_of_interest:
            raise ValueError("evidence ID and quantity of interest are required")
        if not self.artifact_references:
            raise ValueError("evidence must reference at least one artifact")
        object.__setattr__(self, "conditions", _freeze_mapping(self.conditions))


@dataclass(frozen=True, slots=True)
class Study:
    """Declarative scientific intent and reproducibility configuration."""

    study_id: str
    title: str
    configuration: Mapping[str, object]
    acceptance_criteria: Mapping[str, object]
    software_version: str
    git_revision: str
    random_seeds: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.study_id or not self.title or not self.software_version:
            raise ValueError("study ID, title, and software version are required")
        object.__setattr__(self, "configuration", _freeze_mapping(self.configuration))
        object.__setattr__(
            self, "acceptance_criteria", _freeze_mapping(self.acceptance_criteria)
        )
        object.__setattr__(self, "random_seeds", _freeze_mapping(self.random_seeds))


@dataclass(frozen=True, slots=True)
class Run:
    """Immutable evidence-bearing result of a resolved study."""

    run_id: str
    study_id: str
    status: RunStatus
    plugin_ids: tuple[str, ...]
    frames: tuple[Frame, ...]
    evidence: tuple[EvidenceRecord, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id or not self.study_id or not self.frames:
            raise ValueError("run ID, study ID, and at least one frame are required")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a recursively immutable, JSON-compatible mapping."""

    return MappingProxyType(
        {str(key): _freeze_value(value) for key, value in values.items()}
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value
