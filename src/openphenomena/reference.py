"""Command-line runner for the canonical static spherical-bubble study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from openphenomena import __version__
from openphenomena.core import PluginRegistry
from openphenomena.data import Domain, EvidenceRecord, Frame, Run, RunStatus, Study
from openphenomena.storage import write_run_bundle
from openphenomena.studies.spherical_bubble import (
    AnalyticSphere,
    ConvergenceRow,
    StaticBubbleConfig,
    convergence_rows,
)

DomainFactory = Callable[
    [int, float],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]],
]
ModelCapability = Callable[[StaticBubbleConfig], AnalyticSphere]
DerivedCapability = Callable[..., Domain]
ValidatorCapability = Callable[..., tuple[EvidenceRecord, ...]]

REPRODUCTION_COMMAND = "./scripts/reproduce_static_sphere.sh"


class ExportResult(Protocol):
    """Structural exporter result; owned by a plugin, not the core runner."""

    scientific_path: Path
    blender_path: Path
    gltf_path: Path
    roundtrip_max_absolute_error: float


class ViewRecipe(Protocol):
    """Minimum presentation recipe contract consumed by this runner."""

    recipe_id: str
    classification: str
    color_field: str
    scientific_probe_field: str


ExporterCapability = Callable[[Domain, Path], ExportResult]


def run_reference_study(output_directory: Path) -> tuple[Study, Run]:
    """Execute and persist the complete evidence-bearing vertical slice."""

    config = StaticBubbleConfig()
    registry = PluginRegistry.discover()
    plugin_id = "openphenomena.reference.soap_bubble"
    plugin = registry.plugin(plugin_id)
    domain_factory = cast(
        DomainFactory,
        registry.capability("openphenomena.soap_bubble.domain.icosphere").provider,
    )
    model = cast(
        ModelCapability,
        registry.capability("openphenomena.soap_bubble.model.static_sphere").provider,
    )
    derived = cast(
        DerivedCapability,
        registry.capability("openphenomena.soap_bubble.derived.static_fields").provider,
    )
    validator = cast(
        ValidatorCapability,
        registry.capability("openphenomena.soap_bubble.validator.reference").provider,
    )
    exporter = cast(
        ExporterCapability,
        registry.capability("openphenomena.soap_bubble.export.reference").provider,
    )
    view_recipe = cast(
        ViewRecipe,
        registry.capability(
            "openphenomena.soap_bubble.view.static_iridescence"
        ).provider,
    )

    analytic = model(config)
    domains: list[Domain] = []
    for refinement_level in config.refinement_levels:
        tracemalloc.start()
        started = time.perf_counter()
        positions, faces = domain_factory(refinement_level, config.radius_m)
        domain = derived(
            config,
            refinement_level,
            positions,
            faces,
            analytic,
        )
        runtime_s = time.perf_counter() - started
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        domains.append(
            replace(
                domain,
                metadata={
                    **domain.metadata,
                    "runtime_s": runtime_s,
                    "peak_python_memory_bytes": peak_memory,
                },
            )
        )
    domain_tuple = tuple(domains)
    convergence = convergence_rows(domain_tuple)

    output_directory.mkdir(parents=True, exist_ok=True)
    convergence_directory = output_directory / "convergence"
    reports_directory = output_directory / "reports"
    export_directory = output_directory / "exports"
    scientific_directory = output_directory / "scientific"
    convergence_directory.mkdir(exist_ok=True)
    reports_directory.mkdir(exist_ok=True)
    _write_convergence(convergence, convergence_directory)
    export_result = exporter(domain_tuple[-1], export_directory)
    evidence = validator(
        config,
        domain_tuple,
        convergence,
        export_result.roundtrip_max_absolute_error,
    )
    _write_visualization_manifest(
        domain_tuple[-1], export_result, view_recipe, export_directory
    )
    _write_reports(convergence, evidence, reports_directory)

    git_revision = _git_revision()
    source_tree_sha256 = _source_tree_sha256()
    git_dirty = _git_dirty()
    configuration_sha256 = _canonical_sha256(config.as_mapping())
    environment = _environment_metadata()
    run_hash_input = json.dumps(
        {
            "config": config.as_mapping(),
            "configuration_sha256": configuration_sha256,
            "software_version": __version__,
            "git_revision": git_revision,
            "git_dirty": git_dirty,
            "source_tree_sha256": source_tree_sha256,
            "plugin_version": plugin.manifest.version,
            "environment": environment,
        },
        sort_keys=True,
    ).encode("utf-8")
    run_id = "static-sphere-" + hashlib.sha256(run_hash_input).hexdigest()[:16]
    study = Study(
        study_id="openphenomena.study.static_spherical_bubble.v1",
        title="Canonical static spherical soap-bubble reference",
        configuration={
            **config.as_mapping(),
            "configuration_sha256": configuration_sha256,
            "source_tree_sha256": source_tree_sha256,
            "git_dirty": git_dirty,
            "reproduction_command": REPRODUCTION_COMMAND,
        },
        acceptance_criteria={
            "all_evidence_records_pass": True,
            "no_claim_of_linf_convergence_without_measured_support": True,
        },
        software_version=__version__,
        git_revision=git_revision,
        random_seeds={"mesh": config.random_seed},
    )
    frames = tuple(
        Frame(
            frame_id=f"static_refinement_{level}",
            time_s=0.0,
            iteration=level,
            domains=(domain,),
        )
        for level, domain in zip(config.refinement_levels, domain_tuple, strict=True)
    )
    run = Run(
        run_id=run_id,
        study_id=study.study_id,
        status=(
            RunStatus.COMPLETE
            if all(item.passed for item in evidence)
            else RunStatus.REJECTED
        ),
        plugin_ids=(plugin_id,),
        frames=frames,
        evidence=evidence,
        metadata={
            "convergence": [asdict(row) for row in convergence],
            "view_recipe": _view_recipe_record(view_recipe),
            "authoritative_bundle": "scientific/manifest.json",
            "scientific_export": "exports/bubble_finest.vtp",
            "source_tree_sha256": source_tree_sha256,
            "configuration_sha256": configuration_sha256,
            "git_dirty": git_dirty,
            "package_version": __version__,
            "environment": environment,
            "reproduction_command": REPRODUCTION_COMMAND,
            "vtkhdf_status": (
                "not emitted: h5py/VTK support was unavailable in the baseline "
                "environment; VTK XML PolyData used"
            ),
        },
    )
    write_run_bundle(study, run, scientific_directory)
    return study, run


def _write_convergence(rows: tuple[ConvergenceRow, ...], directory: Path) -> None:
    records = [asdict(row) for row in rows]
    (directory / "convergence.json").write_text(
        json.dumps(records, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (directory / "convergence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_visualization_manifest(
    domain: Domain,
    result: ExportResult,
    recipe: ViewRecipe,
    directory: Path,
) -> None:
    manifest = {
        "source_domain": domain.domain_id,
        "artifacts": [
            {
                "path": result.scientific_path.name,
                "classification": "scientific interchange",
                "authoritative": False,
                "fields": [
                    {
                        "semantic_id": field.descriptor.semantic_id,
                        "classification": field.descriptor.fidelity.value,
                        "unit": field.descriptor.unit,
                    }
                    for field in (
                        domain.fields[semantic_id]
                        for semantic_id in sorted(domain.fields)
                    )
                ],
            },
            {
                "path": result.blender_path.name,
                "classification": "VO",
                "authoritative": False,
                "fields": [
                    "geometry.position",
                    "geometry.normal.vertex",
                    recipe.color_field,
                ],
            },
            {
                "path": result.gltf_path.name,
                "classification": "VO",
                "authoritative": False,
                "fields": [
                    "geometry.position",
                    "geometry.normal.vertex",
                    recipe.color_field,
                ],
            },
        ],
        "authoritative_source": "../scientific/manifest.json",
        "view_recipe": _view_recipe_record(recipe),
    }
    (directory / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_reports(
    convergence: tuple[ConvergenceRow, ...],
    evidence: tuple[EvidenceRecord, ...],
    directory: Path,
) -> None:
    table_rows = [
        "| level | vertices | faces | h (m) | L1 H error | L2 H error | "
        "Linf H error | L2 rate | runtime (s) | peak Python memory (B) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in convergence:
        rate = "—" if row.observed_l2_rate is None else f"{row.observed_l2_rate:.4g}"
        table_rows.append(
            f"| {row.refinement_level} | {row.vertex_count} | {row.face_count} | "
            f"{row.characteristic_edge_length_m:.6g} | "
            f"{row.mean_curvature_l1_error_per_m:.6g} | "
            f"{row.mean_curvature_l2_error_per_m:.6g} | "
            f"{row.mean_curvature_linf_error_per_m:.6g} | {rate} | "
            f"{row.runtime_s:.6g} | {row.peak_python_memory_bytes} |"
        )
    (directory / "convergence.md").write_text(
        "# Measured convergence\n\n"
        + "\n".join(table_rows)
        + "\n\nRates are measured between adjacent RMS-edge-length refinements. "
        "The level-1 cotan mean curvature is exact to roundoff by mesh symmetry, "
        "so its transition is not used as an asymptotic rate. Levels 2–4 support "
        "the reported convergence; no universal theoretical rate is claimed.\n",
        encoding="utf-8",
    )
    evidence_rows = [
        "| evidence | quantity | tolerance | measured error | result | artifacts |",
        "|---|---|---:|---:|---|---|",
    ]
    for item in evidence:
        evidence_rows.append(
            f"| {item.evidence_id} | {item.quantity_of_interest} | "
            f"{item.tolerance:.6g} | {item.measured_error:.6g} | "
            f"{'PASS' if item.passed else 'FAIL'} | "
            f"{', '.join(item.artifact_references)} |"
        )
    (directory / "validation.md").write_text(
        "# Validation and verification evidence\n\n"
        + "\n".join(evidence_rows)
        + "\n\nThe analytic sphere and ideal two-interface Young–Laplace "
        "relation are PV. "
        "Discrete curvature, polygonal geometry, and pointwise curved-film optics "
        "are EA. Display color is VO. No SF physics is included.\n",
        encoding="utf-8",
    )


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return "UNBORN_OR_UNAVAILABLE"
    return result.stdout.strip()


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return True
    return bool(result.stdout.strip())


def _source_tree_sha256() -> str:
    repository = Path(__file__).resolve().parents[2]
    paths = _tracked_paths(repository)
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _tracked_paths(repository: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repository,
            check=True,
            capture_output=True,
            timeout=5.0,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        result = None
    if result is not None and result.stdout:
        relative_paths = sorted(
            item.decode("utf-8") for item in result.stdout.split(b"\0") if item
        )
        return [repository / item for item in relative_paths]

    roots = ("docs", "scripts", "src", "tests")
    paths = [repository / name for name in (".gitignore", "pyproject.toml")]
    paths.extend(repository / name for name in ("README.md", "CONTRIBUTING.md"))
    for root in roots:
        paths.extend(
            path
            for path in sorted((repository / root).rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        )
    return [path for path in paths if path.is_file()]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _environment_metadata() -> dict[str, object]:
    dependencies: dict[str, str] = {}
    for distribution in ("mypy", "numpy", "pytest", "ruff"):
        try:
            dependencies[distribution] = version(distribution)
        except PackageNotFoundError:
            dependencies[distribution] = "not-installed"
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "dependencies": dependencies,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
    }


def _view_recipe_record(recipe: ViewRecipe) -> dict[str, str]:
    return {
        "recipe_id": recipe.recipe_id,
        "classification": recipe.classification,
        "color_field": recipe.color_field,
        "scientific_probe_field": recipe.scientific_probe_field,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the canonical static spherical soap-bubble reference study"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/static-spherical-bubble"),
        help="output directory",
    )
    arguments = parser.parse_args()
    study, run = run_reference_study(arguments.output)
    print(f"study={study.study_id}")
    print(f"run={run.run_id}")
    print(f"status={run.status.value}")
    print(f"output={arguments.output.resolve()}")
    return 0 if run.status is RunStatus.COMPLETE else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
