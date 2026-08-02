"""Lossless JSON + NPZ serialization for immutable study/run records."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from openphenomena.data import (
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

SCHEMA_VERSION = "1.1.0"
_SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0.0", SCHEMA_VERSION})


def write_run_bundle(study: Study, run: Run, directory: Path) -> tuple[Path, Path]:
    """Write an authoritative, restartable study/run bundle."""

    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    frame_records: list[dict[str, object]] = []
    for frame_index, frame in enumerate(run.frames):
        domain_records: list[dict[str, object]] = []
        for domain_index, domain in enumerate(frame.domains):
            prefix = f"frame_{frame_index}_domain_{domain_index}"
            arrays[f"{prefix}_positions_m"] = domain.positions_m
            arrays[f"{prefix}_faces"] = domain.faces
            field_records: list[dict[str, object]] = []
            for field_index, semantic_id in enumerate(sorted(domain.fields)):
                sampled_field = domain.fields[semantic_id]
                array_key = f"{prefix}_field_{field_index}"
                arrays[array_key] = sampled_field.values
                field_records.append(
                    {
                        "array_key": array_key,
                        "descriptor": _descriptor_to_record(sampled_field.descriptor),
                        "provenance": [
                            _provenance_to_record(item)
                            for item in sampled_field.provenance
                        ],
                    }
                )
            entity_set_records: list[dict[str, object]] = []
            for entity_set_index, entity_set_id in enumerate(
                sorted(domain.entity_sets)
            ):
                entity_set = domain.entity_sets[entity_set_id]
                indices_key = f"{prefix}_entity_set_{entity_set_index}_indices"
                orientations_key = (
                    f"{prefix}_entity_set_{entity_set_index}_orientations"
                )
                arrays[indices_key] = entity_set.entity_indices
                arrays[orientations_key] = entity_set.orientations
                entity_set_records.append(
                    _entity_set_to_record(
                        entity_set,
                        indices_key=indices_key,
                        orientations_key=orientations_key,
                    )
                )
            domain_records.append(
                {
                    "domain_id": domain.domain_id,
                    "kind": domain.kind,
                    "coordinate_frame": domain.coordinate_frame,
                    "positions_key": f"{prefix}_positions_m",
                    "faces_key": f"{prefix}_faces",
                    "fields": field_records,
                    "entity_sets": entity_set_records,
                    "metadata": _jsonable(domain.metadata),
                }
            )
        frame_records.append(
            {
                "frame_id": frame.frame_id,
                "time_s": frame.time_s,
                "iteration": frame.iteration,
                "domains": domain_records,
            }
        )

    arrays_path = directory / "arrays.npz"
    _write_deterministic_npz(arrays_path, arrays)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study": {
            "study_id": study.study_id,
            "title": study.title,
            "configuration": _jsonable(study.configuration),
            "acceptance_criteria": _jsonable(study.acceptance_criteria),
            "software_version": study.software_version,
            "git_revision": study.git_revision,
            "random_seeds": _jsonable(study.random_seeds),
        },
        "run": {
            "run_id": run.run_id,
            "study_id": run.study_id,
            "status": run.status.value,
            "plugin_ids": list(run.plugin_ids),
            "frames": frame_records,
            "evidence": [_evidence_to_record(item) for item in run.evidence],
            "metadata": _jsonable(run.metadata),
        },
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path, arrays_path


def _write_deterministic_npz(
    path: Path, arrays: Mapping[str, np.ndarray[Any, Any]]
) -> None:
    """Write a NumPy-compatible archive with stable member order and metadata."""

    with zipfile.ZipFile(path, mode="w") as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, arrays[key], allow_pickle=False)
            member = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o644 << 16
            archive.writestr(member, buffer.getvalue())


def read_run_bundle(directory: Path) -> tuple[Study, Run]:
    """Read a bundle created by :func:`write_run_bundle`."""

    raw_manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest = _migrate_manifest(raw_manifest)
    study_record = manifest["study"]
    study = Study(
        study_id=study_record["study_id"],
        title=study_record["title"],
        configuration=study_record["configuration"],
        acceptance_criteria=study_record["acceptance_criteria"],
        software_version=study_record["software_version"],
        git_revision=study_record["git_revision"],
        random_seeds=study_record["random_seeds"],
    )
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
        frames = tuple(
            _frame_from_record(frame_record, arrays)
            for frame_record in manifest["run"]["frames"]
        )
    run_record = manifest["run"]
    run = Run(
        run_id=run_record["run_id"],
        study_id=run_record["study_id"],
        status=RunStatus(run_record["status"]),
        plugin_ids=tuple(run_record["plugin_ids"]),
        frames=frames,
        evidence=tuple(_evidence_from_record(item) for item in run_record["evidence"]),
        metadata=run_record["metadata"],
    )
    return study, run


def _frame_from_record(
    record: dict[str, Any], arrays: Mapping[str, np.ndarray[Any, Any]]
) -> Frame:
    domains: list[Domain] = []
    for domain_record in record["domains"]:
        fields: dict[str, Field] = {}
        for field_record in domain_record["fields"]:
            descriptor = _descriptor_from_record(field_record["descriptor"])
            fields[descriptor.semantic_id] = Field(
                descriptor=descriptor,
                values=arrays[field_record["array_key"]],
                provenance=tuple(
                    _provenance_from_record(item) for item in field_record["provenance"]
                ),
            )
        entity_sets: dict[str, EntitySet] = {}
        for entity_set_record in domain_record["entity_sets"]:
            entity_set = _entity_set_from_record(entity_set_record, arrays)
            entity_sets[entity_set.entity_set_id] = entity_set
        domains.append(
            Domain(
                domain_id=domain_record["domain_id"],
                kind=domain_record["kind"],
                coordinate_frame=domain_record["coordinate_frame"],
                positions_m=arrays[domain_record["positions_key"]],
                faces=arrays[domain_record["faces_key"]],
                fields=fields,
                entity_sets=entity_sets,
                metadata=domain_record["metadata"],
            )
        )
    return Frame(
        frame_id=record["frame_id"],
        time_s=float(record["time_s"]),
        iteration=int(record["iteration"]),
        domains=tuple(domains),
    )


def _migrate_manifest(record: dict[str, Any]) -> dict[str, Any]:
    """Return the current in-memory representation of a supported manifest.

    Schema migration is deliberately additive and does not rewrite the source
    bundle. A v1.0 domain has no named entity sets, so migration supplies an
    empty collection and preserves all existing scientific records verbatim.
    """

    version = record.get("schema_version")
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported schema version: {version}")
    if version == SCHEMA_VERSION:
        return record
    migrated = dict(record)
    run = dict(migrated["run"])
    frames: list[dict[str, Any]] = []
    for source_frame in run["frames"]:
        frame = dict(source_frame)
        domains: list[dict[str, Any]] = []
        for source_domain in frame["domains"]:
            domain = dict(source_domain)
            domain["entity_sets"] = []
            domains.append(domain)
        frame["domains"] = domains
        frames.append(frame)
    run["frames"] = frames
    migrated["run"] = run
    migrated["schema_version"] = SCHEMA_VERSION
    return migrated


def _entity_set_to_record(
    item: EntitySet, *, indices_key: str, orientations_key: str
) -> dict[str, object]:
    semantics = item.boundary_semantics
    semantics_record: dict[str, object] | None = None
    if semantics is not None:
        semantics_record = {
            "semantic_id": semantics.semantic_id,
            "description": semantics.description,
            "parameters": _jsonable(semantics.parameters),
        }
    return {
        "entity_set_id": item.entity_set_id,
        "name": item.name,
        "owner_domain_id": item.owner_domain_id,
        "association": item.association.value,
        "indices_key": indices_key,
        "orientations_key": orientations_key,
        "coordinate_frame": item.coordinate_frame,
        "provenance": [_provenance_to_record(value) for value in item.provenance],
        "boundary_semantics": semantics_record,
        "metadata": _jsonable(item.metadata),
    }


def _entity_set_from_record(
    record: dict[str, Any], arrays: Mapping[str, np.ndarray[Any, Any]]
) -> EntitySet:
    semantics_record = record["boundary_semantics"]
    semantics = None
    if semantics_record is not None:
        semantics = BoundarySemantics(
            semantic_id=semantics_record["semantic_id"],
            description=semantics_record["description"],
            parameters=semantics_record["parameters"],
        )
    return EntitySet(
        entity_set_id=record["entity_set_id"],
        name=record["name"],
        owner_domain_id=record["owner_domain_id"],
        association=FieldAssociation(record["association"]),
        entity_indices=arrays[record["indices_key"]],
        orientations=arrays[record["orientations_key"]],
        coordinate_frame=record["coordinate_frame"],
        provenance=tuple(
            _provenance_from_record(value) for value in record["provenance"]
        ),
        boundary_semantics=semantics,
        metadata=record["metadata"],
    )


def _descriptor_to_record(item: FieldDescriptor) -> dict[str, object]:
    return {
        "semantic_id": item.semantic_id,
        "association": item.association.value,
        "unit": item.unit,
        "unit_dimension": list(item.unit_dimension),
        "shape": list(item.shape),
        "dtype": item.dtype,
        "coordinate_frame": item.coordinate_frame,
        "generating_model": item.generating_model,
        "generating_implementation": item.generating_implementation,
        "fidelity": item.fidelity.value,
        "validation_status": item.validation_status.value,
        "uncertainty": {
            "quantified": item.uncertainty.quantified,
            "description": item.uncertainty.description,
            "standard_uncertainty": item.uncertainty.standard_uncertainty,
            "unit": item.uncertainty.unit,
        },
        "description": item.description,
        "component_names": list(item.component_names),
        "coordinate_axes": list(item.coordinate_axes),
    }


def _descriptor_from_record(record: dict[str, Any]) -> FieldDescriptor:
    uncertainty = record["uncertainty"]
    return FieldDescriptor(
        semantic_id=record["semantic_id"],
        association=FieldAssociation(record["association"]),
        unit=record["unit"],
        unit_dimension=tuple(record["unit_dimension"]),
        shape=tuple(record["shape"]),
        dtype=record["dtype"],
        coordinate_frame=record["coordinate_frame"],
        generating_model=record["generating_model"],
        generating_implementation=record["generating_implementation"],
        fidelity=Fidelity(record["fidelity"]),
        validation_status=ValidationStatus(record["validation_status"]),
        uncertainty=Uncertainty(
            quantified=uncertainty["quantified"],
            description=uncertainty["description"],
            standard_uncertainty=uncertainty["standard_uncertainty"],
            unit=uncertainty["unit"],
        ),
        description=record["description"],
        component_names=tuple(record["component_names"]),
        coordinate_axes=tuple(record["coordinate_axes"]),
    )


def _provenance_to_record(item: Provenance) -> dict[str, object]:
    return {
        "activity": item.activity,
        "implementation": item.implementation,
        "implementation_version": item.implementation_version,
        "source_ids": list(item.source_ids),
        "parameters": _jsonable(item.parameters),
        "citations": list(item.citations),
    }


def _provenance_from_record(record: dict[str, Any]) -> Provenance:
    return Provenance(
        activity=record["activity"],
        implementation=record["implementation"],
        implementation_version=record["implementation_version"],
        source_ids=tuple(record["source_ids"]),
        parameters=record["parameters"],
        citations=tuple(record["citations"]),
    )


def _evidence_to_record(item: EvidenceRecord) -> dict[str, object]:
    return {
        "evidence_id": item.evidence_id,
        "evidence_type": item.evidence_type,
        "quantity_of_interest": item.quantity_of_interest,
        "conditions": _jsonable(item.conditions),
        "tolerance": item.tolerance,
        "measured_error": item.measured_error,
        "passed": item.passed,
        "implementation": item.implementation,
        "implementation_version": item.implementation_version,
        "artifact_references": list(item.artifact_references),
        "notes": item.notes,
    }


def _evidence_from_record(record: dict[str, Any]) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=record["evidence_id"],
        evidence_type=record["evidence_type"],
        quantity_of_interest=record["quantity_of_interest"],
        conditions=record["conditions"],
        tolerance=float(record["tolerance"]),
        measured_error=float(record["measured_error"]),
        passed=bool(record["passed"]),
        implementation=record["implementation"],
        implementation_version=record["implementation_version"],
        artifact_references=tuple(record["artifact_references"]),
        notes=record["notes"],
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value
