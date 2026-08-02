from __future__ import annotations

from pathlib import Path

import numpy as np

from openphenomena.data import EvidenceRecord, Frame, Run, RunStatus, Study
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
