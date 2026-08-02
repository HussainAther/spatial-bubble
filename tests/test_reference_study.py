from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict
from importlib.metadata import entry_points
from pathlib import Path

import numpy as np

from openphenomena.core import CapabilityKind, PluginRegistry
from openphenomena.export import read_vtp, write_vtp
from openphenomena.mesh import create_icosphere
from openphenomena.reference import run_reference_study
from openphenomena.studies.spherical_bubble import (
    StaticBubbleConfig,
    analytic_model,
    build_reference_domain,
    convergence_rows,
)


def build_level(level: int):  # type: ignore[no-untyped-def]
    config = StaticBubbleConfig()
    analytic = analytic_model(config)
    positions, faces = create_icosphere(level, config.radius_m)
    return config, build_reference_domain(config, level, positions, faces, analytic)


def test_plugin_discovery_and_capability_kinds() -> None:
    installed = entry_points(group=PluginRegistry.ENTRY_POINT_GROUP)
    assert any(item.name == "soap_bubble_reference" for item in installed)
    registry = PluginRegistry.discover()
    assert "openphenomena.reference.soap_bubble" in registry.plugin_ids
    for kind in CapabilityKind:
        assert registry.capabilities_of_kind(kind)


def test_standard_entry_point_is_declared() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert '[project.entry-points."openphenomena.plugins"]' in text
    assert "openphenomena.plugins.soap_bubble:SoapBubbleReferencePlugin" in text


def test_analytic_sphere_values_and_sign_conventions() -> None:
    config = StaticBubbleConfig()
    analytic = analytic_model(config)
    assert analytic.principal_curvature_per_m == 100.0
    assert analytic.mean_curvature_per_m == 100.0
    assert analytic.gaussian_curvature_per_m2 == 10_000.0
    assert analytic.pressure_jump_pa == 12.0
    assert config.internal_pressure_pa - config.external_pressure_pa == 12.0
    _, domain = build_level(1)
    principal = domain.fields["geometry.principal_curvature.analytic"].values
    np.testing.assert_array_equal(principal, np.full((42, 2), 100.0))


def test_discrete_curvature_and_pressure_converge_in_asymptotic_window() -> None:
    domains = tuple(build_level(level)[1] for level in (2, 3, 4))
    rows = convergence_rows(domains)
    l2_errors = [row.mean_curvature_l2_error_per_m for row in rows]
    pressure_errors = [row.pressure_l2_error_pa for row in rows]
    assert l2_errors[2] < l2_errors[1] < l2_errors[0]
    assert pressure_errors[2] < pressure_errors[1] < pressure_errors[0]
    assert rows[-1].observed_l2_rate is not None
    assert rows[-1].observed_l2_rate > 1.5


def test_optics_fields_obey_invariants() -> None:
    _, domain = build_level(1)
    rs = domain.fields["optics.reflectance.s"].values
    rp = domain.fields["optics.reflectance.p"].values
    ru = domain.fields["optics.reflectance.unpolarized"].values
    assert np.all((rs >= 0.0) & (rs <= 1.0))
    assert np.all((rp >= 0.0) & (rp <= 1.0))
    np.testing.assert_allclose(ru, 0.5 * (rs + rp), rtol=0.0, atol=1e-14)
    assert domain.fields["optics.reflectance.unpolarized"].descriptor.fidelity == "EA"
    assert domain.fields["color.tonemapped_srgb"].descriptor.fidelity == "VO"


def test_vtp_export_round_trip(tmp_path: Path) -> None:
    _, domain = build_level(1)
    path = write_vtp(domain, tmp_path / "sphere.vtp")
    positions, faces, fields = read_vtp(path)
    np.testing.assert_array_equal(positions, domain.positions_m)
    np.testing.assert_array_equal(faces, domain.faces)
    for semantic_id, field in domain.fields.items():
        np.testing.assert_array_equal(
            fields[semantic_id], field.values.reshape(fields[semantic_id].shape)
        )
    root = ET.parse(path).getroot()
    field_node = root.find(
        ".//PointData/DataArray[@Name='geometry.mean_curvature.discrete']"
    )
    assert field_node is not None
    assert field_node.attrib["Association"] == "vertex"
    assert field_node.attrib["Unit"] == "m^-1"
    assert field_node.attrib["UnitDimension"].startswith("-1.0 ")
    assert field_node.attrib["Fidelity"] == "EA"
    assert field_node.attrib["ValidationStatus"] == "verified"


def test_mesh_and_scientific_fields_are_deterministic() -> None:
    config_a, domain_a = build_level(2)
    config_b, domain_b = build_level(2)
    assert asdict(config_a) == asdict(config_b)
    np.testing.assert_array_equal(domain_a.positions_m, domain_b.positions_m)
    np.testing.assert_array_equal(domain_a.faces, domain_b.faces)
    for semantic_id in domain_a.fields:
        np.testing.assert_array_equal(
            domain_a.fields[semantic_id].values,
            domain_b.fields[semantic_id].values,
        )


def test_complete_reference_workflow(tmp_path: Path) -> None:
    study, run = run_reference_study(tmp_path)
    assert study.study_id == "openphenomena.study.static_spherical_bubble.v1"
    assert run.status == "complete"
    assert all(record.passed for record in run.evidence)
    assert (tmp_path / "scientific" / "manifest.json").exists()
    assert (tmp_path / "exports" / "bubble_finest.vtp").exists()
    assert (tmp_path / "exports" / "bubble_blender_vo.ply").exists()
    assert (tmp_path / "exports" / "bubble_vo.gltf").exists()
    visualization_manifest = json.loads(
        (tmp_path / "exports" / "visualization_manifest.json").read_text()
    )
    assert visualization_manifest["artifacts"][1]["classification"] == "VO"
    manifest = json.loads(
        (tmp_path / "scientific" / "manifest.json").read_text(encoding="utf-8")
    )
    metadata = manifest["run"]["metadata"]
    assert metadata["package_version"] == "0.1.0"
    assert len(metadata["configuration_sha256"]) == 64
    assert len(metadata["source_tree_sha256"]) == 64
    assert metadata["reproduction_command"] == "./scripts/reproduce_static_sphere.sh"
    assert set(metadata["environment"]["dependencies"]) == {
        "mypy",
        "numpy",
        "pytest",
        "ruff",
    }
    for record in run.evidence:
        for relative_path in record.artifact_references:
            assert (tmp_path / relative_path).is_file()


def test_canonical_scientific_baseline() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "static_sphere_baseline.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    domains = tuple(build_level(level)[1] for level in fixture["refinement_levels"])
    rows = convergence_rows(domains)
    for row, expected in zip(rows, fixture["convergence"], strict=True):
        assert row.vertex_count == expected["vertex_count"]
        assert row.face_count == expected["face_count"]
        for key in (
            "characteristic_edge_length_m",
            "mean_curvature_l1_error_per_m",
            "mean_curvature_l2_error_per_m",
            "mean_curvature_linf_error_per_m",
            "pressure_l2_error_pa",
        ):
            np.testing.assert_allclose(
                getattr(row, key), expected[key], rtol=1e-10, atol=1e-15
            )
        expected_rate = expected["observed_l2_rate"]
        if expected_rate is None:
            assert row.observed_l2_rate is None
        else:
            np.testing.assert_allclose(
                row.observed_l2_rate, expected_rate, rtol=1e-10, atol=1e-12
            )
