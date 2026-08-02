from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from openphenomena.data import (
    BoundaryOrientation,
    BoundarySemantics,
    Domain,
    EntitySet,
    EvidenceRecord,
    FieldAssociation,
    Frame,
    Provenance,
    Run,
    RunStatus,
    Study,
)
from openphenomena.mesh import create_icosphere
from openphenomena.storage import read_run_bundle, write_run_bundle
from openphenomena.studies.spherical_bubble import (
    StaticBubbleConfig,
    analytic_model,
    build_reference_domain,
)


def test_authoritative_bundle_round_trip(tmp_path: Path) -> None:
    config = StaticBubbleConfig()
    positions, faces = create_icosphere(1, config.radius_m)
    domain = build_reference_domain(config, 1, positions, faces, analytic_model(config))
    study = Study(
        study_id="test.study",
        title="Serialization test",
        configuration=config.as_mapping(),
        acceptance_criteria={"round_trip": True},
        software_version="test",
        git_revision="test",
        random_seeds={"mesh": 0},
    )
    evidence = EvidenceRecord(
        evidence_id="test.evidence",
        evidence_type="verification",
        quantity_of_interest="array round trip",
        conditions={"level": 1},
        tolerance=0.0,
        measured_error=0.0,
        passed=True,
        implementation="tests.test_serialization",
        implementation_version="1",
        artifact_references=("manifest.json",),
    )
    run = Run(
        run_id="test.run",
        study_id=study.study_id,
        status=RunStatus.COMPLETE,
        plugin_ids=("test.plugin",),
        frames=(Frame("test.frame", 0.0, 1, (domain,)),),
        evidence=(evidence,),
    )
    write_run_bundle(study, run, tmp_path)
    restored_study, restored_run = read_run_bundle(tmp_path)
    assert restored_study.configuration == study.configuration
    assert restored_run.evidence == run.evidence
    restored = restored_run.frames[0].domains[0]
    np.testing.assert_array_equal(restored.positions_m, domain.positions_m)
    np.testing.assert_array_equal(restored.faces, domain.faces)
    for semantic_id in domain.fields:
        assert (
            restored.fields[semantic_id].descriptor
            == domain.fields[semantic_id].descriptor
        )
        np.testing.assert_array_equal(
            restored.fields[semantic_id].values,
            domain.fields[semantic_id].values,
        )


def test_authoritative_array_archive_is_byte_deterministic(tmp_path: Path) -> None:
    config = StaticBubbleConfig()
    positions, faces = create_icosphere(1, config.radius_m)
    domain = build_reference_domain(config, 1, positions, faces, analytic_model(config))
    study = Study(
        study_id="test.study",
        title="Deterministic serialization test",
        configuration=config.as_mapping(),
        acceptance_criteria={"deterministic_archive": True},
        software_version="test",
        git_revision="test",
        random_seeds={"mesh": 0},
    )
    evidence = EvidenceRecord(
        evidence_id="test.determinism",
        evidence_type="verification",
        quantity_of_interest="archive bytes",
        conditions={},
        tolerance=0.0,
        measured_error=0.0,
        passed=True,
        implementation="tests.test_serialization",
        implementation_version="1",
        artifact_references=("manifest.json",),
    )
    run = Run(
        run_id="test.run",
        study_id=study.study_id,
        status=RunStatus.COMPLETE,
        plugin_ids=("test.plugin",),
        frames=(Frame("test.frame", 0.0, 1, (domain,)),),
        evidence=(evidence,),
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_run_bundle(study, run, first)
    write_run_bundle(study, run, second)
    assert (first / "arrays.npz").read_bytes() == (second / "arrays.npz").read_bytes()
    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()


def test_entity_sets_round_trip_with_orientation_and_semantics(tmp_path: Path) -> None:
    config = StaticBubbleConfig()
    positions, faces = create_icosphere(1, config.radius_m)
    domain = build_reference_domain(config, 1, positions, faces, analytic_model(config))
    provenance = Provenance(
        activity="define boundary",
        implementation="tests.test_serialization",
        implementation_version="1",
    )
    entity_set = EntitySet(
        entity_set_id="boundary.test_loop",
        name="Test edge loop",
        owner_domain_id=domain.domain_id,
        association=FieldAssociation.EDGE,
        entity_indices=np.array([0, 1]),
        orientations=np.array(
            [BoundaryOrientation.ALIGNED, BoundaryOrientation.REVERSED]
        ),
        coordinate_frame=domain.coordinate_frame,
        provenance=(provenance,),
        boundary_semantics=BoundarySemantics(
            semantic_id="boundary.pinned_contact_line",
            description="Pinned contact line used for serialization verification.",
            parameters={"position_components": ["x", "y", "z"]},
        ),
        metadata={"owner": "test"},
    )
    domain = replace(domain, entity_sets={entity_set.entity_set_id: entity_set})
    study, run = _minimal_study_and_run(domain)
    write_run_bundle(study, run, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == "1.1.0"
    restored = read_run_bundle(tmp_path)[1].frames[0].domains[0]
    actual = restored.entity_sets[entity_set.entity_set_id]
    np.testing.assert_array_equal(actual.entity_indices, entity_set.entity_indices)
    np.testing.assert_array_equal(actual.orientations, entity_set.orientations)
    assert actual.boundary_semantics == entity_set.boundary_semantics
    assert actual.provenance == entity_set.provenance


def test_v1_0_manifest_migrates_additively(tmp_path: Path) -> None:
    config = StaticBubbleConfig()
    positions, faces = create_icosphere(1, config.radius_m)
    domain = build_reference_domain(config, 1, positions, faces, analytic_model(config))
    study, run = _minimal_study_and_run(domain)
    write_run_bundle(study, run, tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = "1.0.0"
    for frame in manifest["run"]["frames"]:
        for domain_record in frame["domains"]:
            del domain_record["entity_sets"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    restored_study, restored_run = read_run_bundle(tmp_path)
    assert restored_study == study
    assert restored_run.frames[0].domains[0].entity_sets == {}


def test_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    config = StaticBubbleConfig()
    positions, faces = create_icosphere(1, config.radius_m)
    domain = build_reference_domain(config, 1, positions, faces, analytic_model(config))
    study, run = _minimal_study_and_run(domain)
    write_run_bundle(study, run, tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = "2.0.0"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="unsupported schema version: 2.0.0"):
        read_run_bundle(tmp_path)


def test_frozen_v0_1_bundle_is_backward_compatible() -> None:
    default_bundle = (
        Path(__file__).parents[1] / "outputs" / "static-spherical-bubble" / "scientific"
    )
    bundle = Path(os.environ.get("OPENPHENOMENA_V01_BUNDLE", str(default_bundle)))
    if not (bundle / "arrays.npz").exists():
        pytest.skip("large frozen v0.1 reproduction bundle is not materialized")
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["schema_version"] != "1.0.0":
        pytest.skip("local reproducible output is no longer the frozen v1.0 bundle")
    study, run = read_run_bundle(bundle)
    assert study.git_revision == "5652a0c1119f5c1c73363aabf30a051a3ec9c07e"
    assert study.software_version == "0.1.0"
    assert run.frames
    assert all(
        not domain.entity_sets for frame in run.frames for domain in frame.domains
    )


def _minimal_study_and_run(domain: Domain) -> tuple[Study, Run]:
    study = Study(
        study_id="test.study",
        title="Schema evolution test",
        configuration={},
        acceptance_criteria={"round_trip": True},
        software_version="test",
        git_revision="test",
        random_seeds={},
    )
    evidence = EvidenceRecord(
        evidence_id="test.schema",
        evidence_type="verification",
        quantity_of_interest="schema round trip",
        conditions={},
        tolerance=0.0,
        measured_error=0.0,
        passed=True,
        implementation="tests.test_serialization",
        implementation_version="1",
        artifact_references=("manifest.json",),
    )
    run = Run(
        run_id="test.run",
        study_id=study.study_id,
        status=RunStatus.COMPLETE,
        plugin_ids=("test.plugin",),
        frames=(Frame("test.frame", 0.0, 0, (domain,)),),
        evidence=(evidence,),
    )
    return study, run
