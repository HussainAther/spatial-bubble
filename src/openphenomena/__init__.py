"""Core APIs for Open Phenomena."""

from importlib.metadata import PackageNotFoundError, version

from openphenomena.data import (
    BoundaryOrientation,
    BoundarySemantics,
    Domain,
    EntitySet,
    EvidenceRecord,
    Fidelity,
    Field,
    FieldAssociation,
    FieldDescriptor,
    Frame,
    Provenance,
    Run,
    RunStatus,
    Study,
    Uncertainty,
    ValidationStatus,
)

try:
    __version__ = version("openphenomena")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = [
    "BoundaryOrientation",
    "BoundarySemantics",
    "Domain",
    "EntitySet",
    "EvidenceRecord",
    "Fidelity",
    "Field",
    "FieldAssociation",
    "FieldDescriptor",
    "Frame",
    "Provenance",
    "Run",
    "RunStatus",
    "Study",
    "Uncertainty",
    "ValidationStatus",
    "__version__",
]
