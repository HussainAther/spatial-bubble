"""Evidence-bearing runner for the closed fixed-volume sphere-recovery study."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from openphenomena import __version__
from openphenomena.data import (
    UNQUANTIFIED,
    Domain,
    EvidenceRecord,
    Fidelity,
    Field,
    FieldAssociation,
    FieldDescriptor,
    Frame,
    Run,
    RunStatus,
    Study,
    ValidationStatus,
)
from openphenomena.equilibrium.closed_sphere import (
    ClosedSphereAcceptance,
    ClosedSphereConfig,
    ClosedSphereMetrics,
    InitialShape,
    SphereAcceptanceCriteria,
    SphereAnalyticReference,
    assess_closed_sphere,
    build_closed_sphere_problem,
    evaluate_closed_sphere_solution,
    generate_initial_mesh,
    sphere_evidence,
)
from openphenomena.optimization import (
    ConstrainedProblem,
    EvaluationCounts,
    IterationDiagnostic,
    SolverResult,
    SolverSettings,
    SolverTolerances,
    TerminationCategory,
)
from openphenomena.optimization.scipy_trust_constr import ScipyTrustConstrAdapter
from openphenomena.storage import write_run_bundle
from openphenomena.surface import analyze_surface

REPRODUCTION_COMMAND = "./scripts/reproduce_closed_sphere.sh"
_LENGTH = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_AREA = (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_VOLUME = (3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_PRESSURE = (-1.0, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0)
_CURVATURE = (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_DIMENSIONLESS = (0.0,) * 7


@dataclass(frozen=True, slots=True)
class SolvedCase:
    case_id: str
    refinement_level: int
    initial_shape: InitialShape
    result: SolverResult
    metrics: ClosedSphereMetrics
    acceptance: ClosedSphereAcceptance
    domain: Domain
    evidence: tuple[EvidenceRecord, ...]


@dataclass(frozen=True, slots=True)
class ConvergenceRow:
    refinement_level: int
    vertex_count: int
    face_count: int
    characteristic_edge_length_m: float
    energy_relative_error: float
    curvature_l2_error_per_m: float
    pressure_relative_error: float
    hausdorff_error_m: float
    kkt_inf_norm: float
    volume_relative_residual: float
    young_laplace_relative_l2: float
    pressure_observed_rate: float | None
    energy_observed_rate: float | None


def default_solver_settings() -> SolverSettings:
    """Return the fixed, nondimensional settings used by the reference study."""

    return SolverSettings(
        tolerances=SolverTolerances(
            optimality=3.0e-5,
            step=1.0e-8,
            constraint=1.0e-9,
            barrier=1.0e-10,
        ),
        max_iterations=1200,
        history_stride=1,
    )


def solve_case(
    refinement_level: int,
    initial_shape: InitialShape,
    config: ClosedSphereConfig | None = None,
    criteria: SphereAcceptanceCriteria | None = None,
    settings: SolverSettings | None = None,
) -> SolvedCase:
    """Solve and independently assess one predictive equilibrium case."""

    resolved_config = config or ClosedSphereConfig()
    resolved_criteria = criteria or SphereAcceptanceCriteria()
    resolved_settings = settings or default_solver_settings()
    initial, faces = generate_initial_mesh(
        refinement_level, initial_shape, resolved_config
    )
    problem = build_closed_sphere_problem(initial, faces, resolved_config)
    result = _solve_with_deterministic_restarts(problem, resolved_settings)
    metrics = evaluate_closed_sphere_solution(initial, faces, result, resolved_config)
    acceptance = assess_closed_sphere(
        result, metrics, resolved_config, resolved_criteria
    )
    case_id = f"level_{refinement_level}_{initial_shape.value}"
    domain = _solution_domain(
        case_id, faces, result, metrics, acceptance, resolved_config
    )
    evidence = sphere_evidence(
        result,
        metrics,
        acceptance,
        resolved_config,
        resolved_criteria,
        f"scientific/manifest.json#{case_id}",
        case_id,
    )
    return SolvedCase(
        case_id,
        refinement_level,
        initial_shape,
        result,
        metrics,
        acceptance,
        domain,
        evidence,
    )


def _solve_with_deterministic_restarts(
    problem: ConstrainedProblem,
    settings: SolverSettings,
    *,
    max_attempts: int = 4,
) -> SolverResult:
    """Solve with bounded deterministic BFGS restarts after iteration limits.

    ``trust-constr`` BFGS state can accumulate differently across Python, SciPy,
    BLAS, and LAPACK builds.  A restart begins from the previous candidate while
    resetting only the backend's quasi-Newton approximation.  The physical
    objective, constraints, scales, tolerances, and scientific acceptance gates
    remain unchanged.  Diagnostics from every attempt are retained.

    The bounded retry policy is numerical infrastructure, not a relaxation of
    scientific convergence: if no attempt terminates successfully, the final
    result remains non-converged and the study is rejected.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")

    adapter = ScipyTrustConstrAdapter()
    attempts: list[SolverResult] = []
    current_problem = problem
    for attempt_index in range(max_attempts):
        result = adapter.solve(current_problem, settings)
        attempts.append(result)
        if result.termination.category is not TerminationCategory.ITERATION_LIMIT:
            break
        if attempt_index + 1 < max_attempts:
            current_problem = replace(
                problem,
                variables=replace(
                    problem.variables, initial_values=result.solution_physical
                ),
            )

    if len(attempts) == 1:
        return attempts[0]

    final = attempts[-1]
    iteration_offset = 0
    combined_history: list[IterationDiagnostic] = []
    combined_warnings: list[str] = []
    total_counts = [0, 0, 0, 0, 0, 0]
    for index, attempt in enumerate(attempts, start=1):
        combined_history.extend(
            replace(item, iteration=item.iteration + iteration_offset)
            for item in attempt.iteration_history
        )
        iteration_offset += attempt.iteration_count
        for count_index, value in enumerate(attempt.evaluations.as_tuple()):
            total_counts[count_index] += value
        combined_warnings.extend(attempt.numerical_warnings)
        combined_warnings.append(
            "deterministic BFGS attempt "
            f"{index}/{max_attempts}: category={attempt.termination.category.value}, "
            f"iterations={attempt.iteration_count}, "
            f"kkt={attempt.lagrangian_kkt_inf_norm:.17g}"
        )

    counts = EvaluationCounts(
        objective=total_counts[0],
        objective_gradient=total_counts[1],
        objective_hessian=total_counts[2],
        constraints=total_counts[3],
        constraint_jacobians=total_counts[4],
        constraint_hessians=total_counts[5],
    )
    return replace(
        final,
        iteration_count=sum(item.iteration_count for item in attempts),
        evaluations=counts,
        iteration_history=tuple(combined_history),
        numerical_warnings=tuple(dict.fromkeys(combined_warnings)),
    )


def run_closed_sphere_study(output_directory: Path) -> tuple[Study, Run]:
    """Run initialization robustness and three-level measured convergence."""

    config = ClosedSphereConfig()
    criteria = SphereAcceptanceCriteria()
    settings = default_solver_settings()
    cases: list[SolvedCase] = []
    for shape in InitialShape:
        cases.append(solve_case(1, shape, config, criteria, settings))
    for level in (0, 2):
        cases.append(
            solve_case(level, InitialShape.PERTURBED_SPHERE, config, criteria, settings)
        )
    cases.sort(key=lambda item: (item.refinement_level, item.initial_shape.value))
    rows = convergence_rows(tuple(cases), config)
    output_directory.mkdir(parents=True, exist_ok=True)
    reports = output_directory / "reports"
    scientific = output_directory / "scientific"
    reports.mkdir(exist_ok=True)
    _write_reports(tuple(cases), rows, reports)
    convergence_evidence = _convergence_evidence(rows)
    evidence = tuple(record for case in cases for record in case.evidence) + (
        convergence_evidence,
    )
    revision = _git_revision()
    study = Study(
        study_id="openphenomena.study.closed_fixed_volume_sphere.v1",
        title="Predictive closed fixed-volume sphere recovery",
        configuration={
            **dict(config.as_mapping()),
            "solver": dict(settings.as_mapping()),
            "initialization_level": 1,
            "convergence_levels": [0, 1, 2],
            "reproduction_command": REPRODUCTION_COMMAND,
        },
        acceptance_criteria=asdict(criteria),
        software_version=__version__,
        git_revision=revision,
        random_seeds={
            "noisy_sphere": config.noisy_seed,
            "random_displacement": config.displacement_seed,
        },
    )
    frames = tuple(
        Frame(
            frame_id=case.case_id,
            time_s=0.0,
            iteration=case.result.iteration_count,
            domains=(case.domain,),
        )
        for case in cases
    )
    accepted = all(case.acceptance.acceptable for case in cases) and all(
        item.passed for item in evidence
    )
    run_payload = json.dumps(
        {
            "configuration": dict(config.as_mapping()),
            "git_revision": revision,
            "metrics": [asdict(case.metrics) for case in cases],
        },
        sort_keys=True,
    ).encode()
    run = Run(
        run_id="closed-sphere-" + hashlib.sha256(run_payload).hexdigest()[:16],
        study_id=study.study_id,
        status=RunStatus.COMPLETE if accepted else RunStatus.REJECTED,
        plugin_ids=(),
        frames=frames,
        evidence=evidence,
        metadata={
            "case_results": [_case_metadata(case) for case in cases],
            "convergence": [asdict(row) for row in rows],
            "environment": {
                "platform": platform.platform(),
                "python": sys.version,
            },
            "fidelity": {
                "variational_model": "PV",
                "piecewise_linear_geometry": "EA",
                "trust_constr_backend": "EA",
                "ddg_curvature_diagnostic": "EA",
                "sampled_radial_hausdorff": "EA",
                "visualization": "not emitted",
                "future_physics": "SF",
            },
            "pressure_sign_convention": "p=-lambda_V for L=E+lambda_V(V-V0)",
            "reproduction_command": REPRODUCTION_COMMAND,
        },
    )
    write_run_bundle(study, run, scientific)
    return study, run


def convergence_rows(
    cases: tuple[SolvedCase, ...], config: ClosedSphereConfig
) -> tuple[ConvergenceRow, ...]:
    selected = sorted(
        (case for case in cases if case.initial_shape is InitialShape.PERTURBED_SPHERE),
        key=lambda item: item.refinement_level,
    )
    reference = SphereAnalyticReference.from_config(config)
    rows: list[ConvergenceRow] = []
    for case in selected:
        geometry = analyze_surface(case.domain.positions_m, case.domain.faces)
        previous = rows[-1] if rows else None
        pressure_rate = _observed_rate(
            previous.characteristic_edge_length_m if previous else None,
            geometry.characteristic_edge_length_m,
            previous.pressure_relative_error if previous else None,
            case.metrics.pressure_relative_error,
        )
        energy_rate = _observed_rate(
            previous.characteristic_edge_length_m if previous else None,
            geometry.characteristic_edge_length_m,
            previous.energy_relative_error if previous else None,
            case.metrics.energy_relative_error,
        )
        rows.append(
            ConvergenceRow(
                case.refinement_level,
                len(case.domain.positions_m),
                len(case.domain.faces),
                geometry.characteristic_edge_length_m,
                case.metrics.energy_relative_error,
                case.metrics.curvature_l2_error_per_m,
                case.metrics.pressure_relative_error,
                case.metrics.hausdorff_error_m,
                case.result.lagrangian_kkt_inf_norm,
                case.metrics.volume_relative_residual,
                case.metrics.young_laplace_l2_residual_pa / reference.pressure_pa,
                pressure_rate,
                energy_rate,
            )
        )
    return tuple(rows)


def _solution_domain(
    case_id: str,
    faces: np.ndarray[tuple[int, ...], np.dtype[np.int64]],
    result: SolverResult,
    metrics: ClosedSphereMetrics,
    acceptance: ClosedSphereAcceptance,
    config: ClosedSphereConfig,
) -> Domain:
    positions = result.solution_physical.reshape((-1, 3))
    geometry = analyze_surface(positions, faces)
    reference = SphereAnalyticReference.from_config(config)
    yl = (
        2.0
        * config.interface_multiplicity
        * config.surface_tension_n_per_m
        * geometry.mean_curvature_per_m
        - metrics.pressure_pa
    )
    provenance = result.provenance
    fields: dict[str, Field] = {}

    def add(
        semantic_id: str,
        values: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        association: FieldAssociation,
        unit: str,
        dimension: tuple[float, float, float, float, float, float, float],
        description: str,
    ) -> None:
        array = np.asarray(values, dtype=np.float64)
        fields[semantic_id] = Field(
            FieldDescriptor(
                semantic_id,
                association,
                unit,
                dimension,
                array.shape,
                array.dtype.str,
                "world_si",
                "closed fixed-volume capillary equilibrium",
                "openphenomena.equilibrium.reference.solve_case",
                Fidelity.ENGINEERING_APPROXIMATION,
                ValidationStatus.VERIFIED,
                UNQUANTIFIED,
                description,
            ),
            array,
            provenance,
        )

    add(
        "geometry.mean_curvature",
        geometry.mean_curvature_per_m,
        FieldAssociation.VERTEX,
        "m^-1",
        _CURVATURE,
        "Cotan DDG mean curvature; diagnostic only.",
    )
    add(
        "geometry.vertex_area",
        geometry.vertex_areas_m2,
        FieldAssociation.VERTEX,
        "m^2",
        _AREA,
        "Mixed Voronoi vertex area used for norms.",
    )
    add(
        "mechanics.young_laplace_residual",
        yl,
        FieldAssociation.VERTEX,
        "Pa",
        _PRESSURE,
        "Independent 2*m*gamma*H-p residual.",
    )
    add(
        "mechanics.pressure_jump",
        np.array(metrics.pressure_pa),
        FieldAssociation.GLOBAL,
        "Pa",
        _PRESSURE,
        "Pressure inferred as the negative volume multiplier.",
    )
    add(
        "geometry.enclosed_volume",
        np.array(metrics.recovered_volume_m3),
        FieldAssociation.GLOBAL,
        "m^3",
        _VOLUME,
        "Exact oriented polyhedral volume.",
    )
    add(
        "geometry.hausdorff_error.sampled_radial",
        np.array(metrics.hausdorff_error_m),
        FieldAssociation.GLOBAL,
        "m",
        _LENGTH,
        "Maximum vertex, edge-midpoint, and face-centroid radial deviation; "
        "not exact continuous Hausdorff distance.",
    )
    add(
        "validation.energy_relative_error",
        np.array(metrics.energy_relative_error),
        FieldAssociation.GLOBAL,
        "1",
        _DIMENSIONLESS,
        f"Relative error against analytic sphere energy {reference.energy_j:.17g} J.",
    )
    return Domain(
        domain_id=f"openphenomena.domain.{case_id}",
        kind="closed_triangular_surface",
        coordinate_frame="world_si",
        positions_m=positions,
        faces=faces,
        fields=fields,
        metadata={
            "solver_result": result.as_metadata(),
            "scientific_acceptance": acceptance.acceptable,
        },
    )


def _case_metadata(case: SolvedCase) -> dict[str, object]:
    return {
        "acceptance": {
            "acceptable": case.acceptance.acceptable,
            "checks": dict(case.acceptance.checks),
            "reasons": list(case.acceptance.reasons),
        },
        "case_id": case.case_id,
        "initial_shape": case.initial_shape.value,
        "metrics": asdict(case.metrics),
        "refinement_level": case.refinement_level,
        "solver": case.result.as_metadata(),
    }


def _observed_rate(
    previous_h: float | None,
    current_h: float,
    previous_error: float | None,
    current_error: float,
) -> float | None:
    if (
        previous_h is None
        or previous_error is None
        or min(previous_error, current_error) <= 0.0
    ):
        return None
    return float(
        np.log(current_error / previous_error) / np.log(current_h / previous_h)
    )


def _convergence_evidence(rows: tuple[ConvergenceRow, ...]) -> EvidenceRecord:
    decreasing = all(
        later.pressure_relative_error < earlier.pressure_relative_error
        and later.energy_relative_error < earlier.energy_relative_error
        for earlier, later in zip(rows[:-1], rows[1:], strict=True)
    )
    return EvidenceRecord(
        evidence_id="openphenomena.closed_sphere.refinement.measured_convergence",
        evidence_type="measured_convergence",
        quantity_of_interest="pressure and energy error decrease under refinement",
        conditions={
            "levels": [row.refinement_level for row in rows],
            "asymptotic_claim": False,
        },
        tolerance=0.0,
        measured_error=0.0 if decreasing else 1.0,
        passed=decreasing,
        implementation="openphenomena.equilibrium.reference.convergence_rows",
        implementation_version=__version__,
        artifact_references=("reports/convergence.json", "scientific/manifest.json"),
        notes=(
            "Rates are empirical adjacent-mesh measurements; no asymptotic "
            "regime is claimed."
        ),
    )


def _write_reports(
    cases: tuple[SolvedCase, ...], rows: tuple[ConvergenceRow, ...], directory: Path
) -> None:
    (directory / "validation.json").write_text(
        json.dumps(
            _plain([_case_metadata(case) for case in cases]),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "convergence.json").write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "validation.md").write_text(
        _validation_markdown(cases), encoding="utf-8"
    )
    (directory / "convergence.md").write_text(
        _convergence_markdown(rows), encoding="utf-8"
    )


def _validation_markdown(cases: tuple[SolvedCase, ...]) -> str:
    lines = [
        "# Closed fixed-volume sphere recovery validation\n",
        "| case | backend | accepted | KKT | volume rel. | YL rel. L2 | "
        "pressure rel. | energy rel. |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        reference_pressure = 2.0 * 2.0 * 0.03 / 0.01
        lines.append(
            f"| {case.case_id} | {case.result.termination.category.value} | "
            f"{case.acceptance.acceptable} | "
            f"{case.result.lagrangian_kkt_inf_norm:.6g} | "
            f"{case.metrics.volume_relative_residual:.6g} | "
            f"{case.metrics.young_laplace_l2_residual_pa / reference_pressure:.6g} | "
            f"{case.metrics.pressure_relative_error:.6g} | "
            f"{case.metrics.energy_relative_error:.6g} |"
        )
    return (
        "\n".join(lines)
        + "\n\nThe continuous variational model and analytic sphere are PV. "
        "The polygonal discretization, SciPy solve, DDG curvature, and sampled "
        "radial Hausdorff estimate are EA. No visualization-only result is "
        "authoritative.\n"
    )


def _convergence_markdown(rows: tuple[ConvergenceRow, ...]) -> str:
    lines = [
        "# Measured refinement study\n",
        "| level | vertices | h (m) | pressure error | pressure rate | "
        "energy error | energy rate | KKT |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pr = (
            "—"
            if row.pressure_observed_rate is None
            else f"{row.pressure_observed_rate:.4g}"
        )
        er = (
            "—"
            if row.energy_observed_rate is None
            else f"{row.energy_observed_rate:.4g}"
        )
        lines.append(
            f"| {row.refinement_level} | {row.vertex_count} | "
            f"{row.characteristic_edge_length_m:.6g} | "
            f"{row.pressure_relative_error:.6g} | {pr} | "
            f"{row.energy_relative_error:.6g} | {er} | "
            f"{row.kkt_inf_norm:.6g} |"
        )
    return (
        "\n".join(lines)
        + "\n\nThese are measured adjacent-level rates only. Three levels do "
        "not establish an asymptotic regime.\n"
    )


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() or "UNBORN_OR_UNAVAILABLE"


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/closed-sphere"))
    args = parser.parse_args()
    _, run = run_closed_sphere_study(args.output)
    if run.status is not RunStatus.COMPLETE:
        raise SystemExit("closed-sphere study failed scientific acceptance")


if __name__ == "__main__":
    main()
