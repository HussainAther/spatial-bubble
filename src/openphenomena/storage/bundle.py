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
    Domain,
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

SCHEMA_VERSION = "1.0.0"


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
            domain_records.append(
                {
                    "domain_id": domain.domain_id,
                    "kind": domain.kind,
                    "coordinate_frame": domain.coordinate_frame,
                    "positions_key": f"{prefix}_positions_m",
                    "faces_key": f"{prefix}_faces",
                    "fields": field_records,
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

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {manifest['schema_version']}")
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
        domains.append(
            Domain(
                domain_id=domain_record["domain_id"],
                kind=domain_record["kind"],
                coordinate_frame=domain_record["coordinate_frame"],
                positions_m=arrays[domain_record["positions_key"]],
                faces=arrays[domain_record["faces_key"]],
                fields=fields,
                metadata=domain_record["metadata"],
            )
        )
    return Frame(
        frame_id=record["frame_id"],
        time_s=float(record["time_s"]),
        iteration=int(record["iteration"]),
        domains=tuple(domains),
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
