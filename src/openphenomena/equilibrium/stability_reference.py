"""Reproducible projected-stability study for the solved closed sphere."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from openphenomena.data import (
    UNQUANTIFIED,
    Domain,
    Fidelity,
    Field,
    FieldAssociation,
    FieldDescriptor,
    ValidationStatus,
)
from openphenomena.equilibrium.closed_sphere import (
    ClosedSphereConfig,
    InitialShape,
    build_closed_sphere_problem,
    generate_initial_mesh,
)
from openphenomena.equilibrium.reference import solve_case
from openphenomena.equilibrium.stability import (
    StabilityResult,
    StabilitySettings,
    analyze_constrained_stability,
)
from openphenomena.export import write_vtp

_DIMENSIONLESS = (0.0,) * 7
_LENGTH = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def run_stability_study(output_directory: Path) -> StabilityResult:
    """Solve a coarse sphere, analyze its constrained spectrum, and export modes."""

    config = ClosedSphereConfig()
    initial, faces = generate_initial_mesh(0, InitialShape.PERTURBED_SPHERE, config)
    case = solve_case(0, InitialShape.PERTURBED_SPHERE, config=config)
    problem = build_closed_sphere_problem(initial, faces, config)
    settings = StabilitySettings(maximum_modes=10)
    result = analyze_constrained_stability(problem, case.result, settings)
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_directory / "stability_spectrum.npz",
        eigenvalues=result.eigenvalues,
        modes_dimensionless=result.modes_dimensionless,
        modes_physical=result.modes_physical,
    )
    (output_directory / "stability_report.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True, default=_json_default)
        + "\n"
    )
    domain = _mode_domain(case.domain, result)
    write_vtp(domain, output_directory / "sphere_stability_modes.vtp")
    return result


def _mode_domain(domain: Domain, result: StabilityResult) -> Domain:
    fields = dict(domain.fields)
    provenance = next(iter(domain.fields.values())).provenance
    vertex_count = len(domain.positions_m)
    for index, (eigenvalue, flattened) in enumerate(
        zip(result.eigenvalues, result.modes_physical, strict=True)
    ):
        values = flattened.reshape(vertex_count, 3)
        semantic_id = f"stability.mode_{index:03d}"
        fields[semantic_id] = Field(
            FieldDescriptor(
                semantic_id=semantic_id,
                association=FieldAssociation.VERTEX,
                unit="m",
                unit_dimension=_LENGTH,
                shape=values.shape,
                dtype=values.dtype.str,
                coordinate_frame=domain.coordinate_frame,
                generating_model="projected constrained second variation",
                generating_implementation=(
                    "openphenomena.equilibrium.stability.analyze_constrained_stability"
                ),
                fidelity=Fidelity.ENGINEERING_APPROXIMATION,
                validation_status=ValidationStatus.VERIFIED,
                uncertainty=UNQUANTIFIED,
                description=(
                    "Unit-RMS physical displacement eigenmode; "
                    f"dimensionless eigenvalue={eigenvalue:.17g}."
                ),
                component_names=("dx", "dy", "dz"),
                coordinate_axes=("x", "y", "z"),
            ),
            values,
            provenance,
        )
    fields["stability.eigenvalues"] = Field(
        FieldDescriptor(
            semantic_id="stability.eigenvalues",
            association=FieldAssociation.GLOBAL,
            unit="1",
            unit_dimension=_DIMENSIONLESS,
            shape=result.eigenvalues.shape,
            dtype=result.eigenvalues.dtype.str,
            coordinate_frame=domain.coordinate_frame,
            generating_model="projected constrained second variation",
            generating_implementation=(
                "openphenomena.equilibrium.stability.analyze_constrained_stability"
            ),
            fidelity=Fidelity.ENGINEERING_APPROXIMATION,
            validation_status=ValidationStatus.VERIFIED,
            uncertainty=UNQUANTIFIED,
            description="Lowest dimensionless projected Hessian eigenvalues.",
        ),
        result.eigenvalues,
        provenance,
    )
    return Domain(
        domain_id=domain.domain_id + ".stability",
        kind=domain.kind,
        coordinate_frame=domain.coordinate_frame,
        positions_m=domain.positions_m,
        faces=domain.faces,
        fields=fields,
        entity_sets=domain.entity_sets,
        metadata={**dict(domain.metadata), "stability": asdict(result)},
    )


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/stability-sphere")
    )
    args = parser.parse_args()
    result = run_stability_study(args.output)
    print(
        f"stable_semidefinite={result.stable_semidefinite} "
        f"negative_modes={result.negative_mode_count} "
        f"null_modes={result.null_mode_count} "
        f"smallest={result.smallest_eigenvalue:.9g}"
    )


if __name__ == "__main__":
    main()
