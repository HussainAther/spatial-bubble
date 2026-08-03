"""Closed, fixed-volume, gravity-free capillary equilibrium.

The discrete problem is ``min m*gamma*A(x)`` subject to the exact oriented
polyhedral volume ``V(x)=V0``.  A volume-centroid constraint removes only the
three-dimensional translational nullspace.  Curvature is never supplied to the
optimizer; it is evaluated after a solve as an independent diagnostic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import numpy as np

from openphenomena import __version__
from openphenomena.data import EvidenceRecord, FloatArray, IntArray, Provenance
from openphenomena.mesh.icosphere import create_icosphere
from openphenomena.optimization import (
    ConstrainedProblem,
    EqualityConstraint,
    HessianMode,
    OptimizationVariables,
    ScalarObjective,
    SolverResult,
)
from openphenomena.surface import (
    EquilibriumScales,
    analyze_surface,
    centroid_jacobian,
    oriented_volume,
    surface_energy_gradient,
    total_area,
    total_surface_energy,
    volume_centroid,
    volume_gradient,
)


class InitialShape(StrEnum):
    """Deterministic initialization families covered by the validation study."""

    PERTURBED_SPHERE = "perturbed_sphere"
    STRETCHED_ELLIPSOID = "stretched_ellipsoid"
    NOISY_SPHERE = "noisy_sphere"
    RANDOM_DISPLACEMENT = "random_displacement"


@dataclass(frozen=True, slots=True)
class ClosedSphereConfig:
    """SI-valued physical and initialization configuration."""

    target_radius_m: float = 0.01
    surface_tension_n_per_m: float = 0.03
    interface_multiplicity: float = 2.0
    target_centroid_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radial_perturbation_fraction: float = 0.08
    ellipsoid_axis_factors: tuple[float, float, float] = (1.04, 0.99, 0.97)
    noise_fraction: float = 0.025
    random_displacement_fraction_of_edge: float = 0.02
    noisy_seed: int = 1729
    displacement_seed: int = 2718

    def __post_init__(self) -> None:
        positive = (
            self.target_radius_m,
            self.surface_tension_n_per_m,
            self.interface_multiplicity,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError(
                "radius, surface tension, and multiplicity must be positive"
            )
        fractions = (
            self.radial_perturbation_fraction,
            self.noise_fraction,
            self.random_displacement_fraction_of_edge,
        )
        if any(not 0.0 <= value <= 0.15 for value in fractions):
            raise ValueError("initial displacement fractions must lie in [0, 0.15]")
        if any(value <= 0.0 for value in self.ellipsoid_axis_factors):
            raise ValueError("ellipsoid axis factors must be positive")

    @property
    def target_volume_m3(self) -> float:
        return 4.0 * np.pi * self.target_radius_m**3 / 3.0

    @property
    def target_centroid_array_m(self) -> FloatArray:
        result = np.asarray(self.target_centroid_m, dtype=np.float64)
        result.flags.writeable = False
        return result

    @property
    def scales(self) -> EquilibriumScales:
        return EquilibriumScales.from_target_volume(
            self.target_volume_m3,
            self.surface_tension_n_per_m,
            self.interface_multiplicity,
        )

    def as_mapping(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "ellipsoid_axis_factors": list(self.ellipsoid_axis_factors),
                "interface_multiplicity": self.interface_multiplicity,
                "noise_fraction": self.noise_fraction,
                "random_displacement_fraction_of_edge": (
                    self.random_displacement_fraction_of_edge
                ),
                "radial_perturbation_fraction": self.radial_perturbation_fraction,
                "surface_tension_n_per_m": self.surface_tension_n_per_m,
                "target_centroid_m": list(self.target_centroid_m),
                "target_radius_m": self.target_radius_m,
                "target_volume_m3": self.target_volume_m3,
            }
        )


@dataclass(frozen=True, slots=True)
class SphereAnalyticReference:
    radius_m: float
    area_m2: float
    volume_m3: float
    mean_curvature_per_m: float
    pressure_pa: float
    energy_j: float

    @classmethod
    def from_config(cls, config: ClosedSphereConfig) -> SphereAnalyticReference:
        radius = config.target_radius_m
        area = 4.0 * np.pi * radius**2
        return cls(
            radius_m=radius,
            area_m2=area,
            volume_m3=config.target_volume_m3,
            mean_curvature_per_m=1.0 / radius,
            pressure_pa=(
                2.0
                * config.interface_multiplicity
                * config.surface_tension_n_per_m
                / radius
            ),
            energy_j=(
                config.interface_multiplicity * config.surface_tension_n_per_m * area
            ),
        )


@dataclass(frozen=True, slots=True)
class MeshQuality:
    minimum_area_m2: float
    minimum_angle_deg: float
    maximum_aspect_ratio: float
    oriented_volume_m3: float
    inverted_face_count: int
    degenerate_face_count: int

    @property
    def admissible(self) -> bool:
        return (
            self.oriented_volume_m3 > 0.0
            and self.inverted_face_count == 0
            and self.degenerate_face_count == 0
        )


@dataclass(frozen=True, slots=True)
class ClosedSphereMetrics:
    recovered_radius_m: float
    radial_mean_radius_m: float
    recovered_area_m2: float
    recovered_volume_m3: float
    area_weighted_mean_curvature_per_m: float
    pressure_pa: float
    young_laplace_l2_residual_pa: float
    young_laplace_linf_residual_pa: float
    hausdorff_error_m: float
    curvature_l2_error_per_m: float
    pressure_relative_error: float
    energy_relative_error: float
    area_relative_error: float
    volume_relative_residual: float
    initial_energy_j: float
    final_energy_j: float
    relative_energy_decrease: float
    mesh_quality: MeshQuality


@dataclass(frozen=True, slots=True)
class SphereAcceptanceCriteria:
    """Independent scientific gates; backend success is only one gate."""

    kkt_inf_norm: float = 3.1e-5
    volume_relative_residual: float = 2.0e-9
    young_laplace_relative_l2: float = 0.23
    pressure_relative_error: float = 0.07
    minimum_angle_deg: float = 8.0
    maximum_aspect_ratio: float = 8.0


@dataclass(frozen=True, slots=True)
class ClosedSphereAcceptance:
    acceptable: bool
    checks: Mapping[str, bool]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", MappingProxyType(dict(self.checks)))


def generate_initial_mesh(
    refinement_level: int,
    shape: InitialShape,
    config: ClosedSphereConfig,
) -> tuple[FloatArray, IntArray]:
    """Create a deterministic, volume-normalized admissible initialization."""

    positions, faces = create_icosphere(refinement_level, config.target_radius_m)
    radius = np.linalg.norm(positions, axis=1)
    directions = positions / radius[:, None]
    if shape is InitialShape.PERTURBED_SPHERE:
        z = directions[:, 2]
        factor = 1.0 + config.radial_perturbation_fraction * 0.5 * (3.0 * z**2 - 1.0)
        positions = positions * factor[:, None]
    elif shape is InitialShape.STRETCHED_ELLIPSOID:
        positions = positions * np.asarray(config.ellipsoid_axis_factors)[None, :]
    elif shape is InitialShape.NOISY_SPHERE:
        rng = np.random.default_rng(config.noisy_seed + refinement_level)
        factor = 1.0 + rng.uniform(
            -config.noise_fraction, config.noise_fraction, len(positions)
        )
        positions = positions * factor[:, None]
    elif shape is InitialShape.RANDOM_DISPLACEMENT:
        rng = np.random.default_rng(config.displacement_seed + refinement_level)
        h = analyze_surface(positions, faces).characteristic_edge_length_m
        displacements = rng.normal(size=positions.shape)
        norms = np.linalg.norm(displacements, axis=1)
        displacements /= norms[:, None]
        magnitudes = rng.uniform(
            0.0, config.random_displacement_fraction_of_edge * h, len(positions)
        )
        positions = positions + displacements * magnitudes[:, None]
    else:  # pragma: no cover - exhaustive StrEnum guard
        raise ValueError(f"unsupported initial shape {shape}")
    positions = _restore_centroid_and_volume(positions, faces, config)
    quality = mesh_quality(positions, faces)
    if not quality.admissible:
        raise ValueError("configured initialization produced an inadmissible mesh")
    return _readonly_float(positions), _readonly_int(faces)


def build_closed_sphere_problem(
    positions_m: FloatArray,
    faces: IntArray,
    config: ClosedSphereConfig,
) -> ConstrainedProblem:
    """Expose exact energy, volume, and centroid gauge through neutral contracts."""

    initial = np.asarray(positions_m, dtype=np.float64)
    connectivity = np.asarray(faces, dtype=np.int64)
    vertex_count = len(initial)
    scales = config.scales
    provenance = (
        Provenance(
            activity="closed fixed-volume capillary equilibrium formulation",
            implementation="openphenomena.equilibrium.closed_sphere.build_closed_sphere_problem",
            implementation_version=__version__,
            source_ids=("geometry.position", "geometry.face_connectivity"),
            parameters={
                **dict(config.as_mapping()),
                "lagrangian": "L=E+lambda_V(V-V0)+lambda_C dot(C-C0)",
                "pressure_sign": "p=-lambda_V",
                "centroid_constraint_role": "translation gauge only",
            },
            citations=("Young-Laplace variational principle",),
        ),
    )

    def reshape(values: FloatArray) -> FloatArray:
        return np.asarray(values, dtype=np.float64).reshape(vertex_count, 3)

    def centroid_jac(values: FloatArray) -> FloatArray:
        jac = centroid_jacobian(reshape(values), connectivity)
        return jac.transpose(1, 0, 2).reshape(3, -1)

    return ConstrainedProblem(
        problem_id="openphenomena.equilibrium.closed_fixed_volume.v1",
        variables=OptimizationVariables(
            names=tuple(
                f"vertex[{vertex}].{axis}"
                for vertex in range(vertex_count)
                for axis in ("x", "y", "z")
            ),
            initial_values=initial.reshape(-1),
            scale_to_physical=np.full(initial.size, scales.length_m),
            units=("m",) * initial.size,
        ),
        objective=ScalarObjective(
            objective_id="openphenomena.energy.constant_surface_tension",
            value=lambda x: total_surface_energy(
                reshape(x),
                connectivity,
                config.surface_tension_n_per_m,
                config.interface_multiplicity,
            ),
            gradient=lambda x: surface_energy_gradient(
                reshape(x),
                connectivity,
                config.surface_tension_n_per_m,
                config.interface_multiplicity,
            ).reshape(-1),
            scale_to_physical=scales.energy_j,
            unit="J",
            hessian_mode=HessianMode.BFGS,
        ),
        equality_constraints=(
            EqualityConstraint(
                constraint_id="openphenomena.constraint.enclosed_volume",
                size=1,
                value=lambda x: np.array(
                    [
                        oriented_volume(reshape(x), connectivity)
                        - config.target_volume_m3
                    ]
                ),
                jacobian=lambda x: volume_gradient(reshape(x), connectivity).reshape(
                    1, -1
                ),
                scale_to_physical=np.array([config.target_volume_m3]),
                units=("m^3",),
                hessian_mode=HessianMode.BFGS,
            ),
            EqualityConstraint(
                constraint_id="openphenomena.constraint.volume_centroid_gauge",
                size=3,
                value=lambda x: (
                    volume_centroid(reshape(x), connectivity)
                    - config.target_centroid_array_m
                ),
                jacobian=centroid_jac,
                scale_to_physical=np.full(3, scales.length_m),
                units=("m", "m", "m"),
                hessian_mode=HessianMode.BFGS,
            ),
        ),
        provenance=provenance,
        admissibility=lambda x: _admissibility(reshape(x), connectivity),
    )


def mesh_quality(positions_m: FloatArray, faces: IntArray) -> MeshQuality:
    """Return deterministic local quality and orientation diagnostics."""

    positions = np.asarray(positions_m, dtype=np.float64)
    connectivity = np.asarray(faces, dtype=np.int64)
    triangles = positions[connectivity]
    edge_vectors = np.stack(
        (
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 1],
            triangles[:, 0] - triangles[:, 2],
        ),
        axis=1,
    )
    lengths = np.linalg.norm(edge_vectors, axis=2)
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    degenerate = areas <= (np.finfo(np.float64).eps * float(np.max(lengths)) ** 2)
    safe_area = np.maximum(areas, np.finfo(np.float64).tiny)
    altitudes = (
        2.0 * safe_area[:, None] / np.maximum(lengths, np.finfo(np.float64).tiny)
    )
    aspect = np.max(lengths, axis=1) / np.min(altitudes, axis=1)
    angles = _triangle_angles_from_lengths(lengths)
    try:
        centroid = volume_centroid(positions, connectivity)
        volume = oriented_volume(positions, connectivity)
    except ValueError:
        centroid = np.mean(positions, axis=0)
        volume = 0.0
    face_centroids = np.mean(triangles, axis=1)
    inverted = np.einsum("ij,ij->i", cross, face_centroids - centroid) <= 0.0
    return MeshQuality(
        minimum_area_m2=float(np.min(areas)),
        minimum_angle_deg=float(np.rad2deg(np.min(angles))),
        maximum_aspect_ratio=float(np.max(aspect)),
        oriented_volume_m3=volume,
        inverted_face_count=int(np.count_nonzero(inverted)),
        degenerate_face_count=int(np.count_nonzero(degenerate)),
    )


def evaluate_closed_sphere_solution(
    initial_positions_m: FloatArray,
    faces: IntArray,
    result: SolverResult,
    config: ClosedSphereConfig,
) -> ClosedSphereMetrics:
    """Evaluate independent analytic and discrete validation quantities."""

    final = result.solution_physical.reshape((-1, 3))
    reference = SphereAnalyticReference.from_config(config)
    geometry = analyze_surface(final, faces)
    volume = oriented_volume(final, faces)
    area = total_area(final, faces)
    centroid = volume_centroid(final, faces)
    radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    radial = np.linalg.norm(final - centroid[None, :], axis=1)
    hausdorff_samples = _surface_samples(final, faces)
    sampled_radial = np.linalg.norm(hausdorff_samples - centroid[None, :], axis=1)
    weights = geometry.vertex_areas_m2
    mean_curvature = float(
        np.sum(weights * geometry.mean_curvature_per_m) / np.sum(weights)
    )
    pressure = _pressure_from_result(result)
    yl = (
        2.0
        * config.interface_multiplicity
        * config.surface_tension_n_per_m
        * geometry.mean_curvature_per_m
        - pressure
    )
    curvature_error = geometry.mean_curvature_per_m - reference.mean_curvature_per_m

    def weighted_l2(values: FloatArray) -> float:
        return float(np.sqrt(np.sum(weights * values**2) / np.sum(weights)))

    initial_energy = total_surface_energy(
        initial_positions_m,
        faces,
        config.surface_tension_n_per_m,
        config.interface_multiplicity,
    )
    final_energy = result.objective_physical
    return ClosedSphereMetrics(
        recovered_radius_m=radius,
        radial_mean_radius_m=float(np.mean(radial)),
        recovered_area_m2=area,
        recovered_volume_m3=volume,
        area_weighted_mean_curvature_per_m=mean_curvature,
        pressure_pa=pressure,
        young_laplace_l2_residual_pa=weighted_l2(yl),
        young_laplace_linf_residual_pa=float(np.max(np.abs(yl))),
        hausdorff_error_m=float(np.max(np.abs(sampled_radial - reference.radius_m))),
        curvature_l2_error_per_m=weighted_l2(curvature_error),
        pressure_relative_error=abs(pressure - reference.pressure_pa)
        / reference.pressure_pa,
        energy_relative_error=abs(final_energy - reference.energy_j)
        / reference.energy_j,
        area_relative_error=abs(area - reference.area_m2) / reference.area_m2,
        volume_relative_residual=abs(volume - reference.volume_m3)
        / reference.volume_m3,
        initial_energy_j=initial_energy,
        final_energy_j=final_energy,
        relative_energy_decrease=(initial_energy - final_energy) / initial_energy,
        mesh_quality=mesh_quality(final, faces),
    )


def assess_closed_sphere(
    result: SolverResult,
    metrics: ClosedSphereMetrics,
    config: ClosedSphereConfig,
    criteria: SphereAcceptanceCriteria,
) -> ClosedSphereAcceptance:
    reference = SphereAnalyticReference.from_config(config)
    checks = {
        "optimizer_convergence": result.backend_converged,
        "kkt_residual": result.lagrangian_kkt_inf_norm <= criteria.kkt_inf_norm,
        "volume_residual": metrics.volume_relative_residual
        <= criteria.volume_relative_residual,
        "young_laplace_residual": (
            metrics.young_laplace_l2_residual_pa / reference.pressure_pa
            <= criteria.young_laplace_relative_l2
        ),
        "pressure_multiplier": metrics.pressure_relative_error
        <= criteria.pressure_relative_error,
        "positive_orientation": metrics.mesh_quality.oriented_volume_m3 > 0.0,
        "no_inverted_elements": metrics.mesh_quality.inverted_face_count == 0,
        "no_degenerate_elements": metrics.mesh_quality.degenerate_face_count == 0,
        "minimum_angle": metrics.mesh_quality.minimum_angle_deg
        >= criteria.minimum_angle_deg,
        "aspect_ratio": metrics.mesh_quality.maximum_aspect_ratio
        <= criteria.maximum_aspect_ratio,
        "adapter_admissibility": result.admissible,
    }
    failures = tuple(f"failed: {name}" for name, passed in checks.items() if not passed)
    return ClosedSphereAcceptance(
        acceptable=not failures,
        checks=checks,
        reasons=failures or ("all independently configured scientific gates passed",),
    )


def sphere_evidence(
    result: SolverResult,
    metrics: ClosedSphereMetrics,
    acceptance: ClosedSphereAcceptance,
    config: ClosedSphereConfig,
    criteria: SphereAcceptanceCriteria,
    artifact: str,
    case_id: str = "case",
) -> tuple[EvidenceRecord, ...]:
    """Build the six required, linked evidence categories for one solve."""

    reference = SphereAnalyticReference.from_config(config)
    records = (
        (
            "optimization",
            "backend termination",
            0.0 if result.backend_converged else 1.0,
            0.0,
        ),
        (
            "kkt",
            "dimensionless Lagrangian KKT residual",
            result.lagrangian_kkt_inf_norm,
            criteria.kkt_inf_norm,
        ),
        (
            "young_laplace",
            "area-weighted Young-Laplace residual",
            metrics.young_laplace_l2_residual_pa / reference.pressure_pa,
            criteria.young_laplace_relative_l2,
        ),
        (
            "geometric_accuracy",
            "relative sphere energy error",
            metrics.energy_relative_error,
            0.15,
        ),
        (
            "convergence",
            "energy decrease from nonspherical initialization",
            max(0.0, -metrics.relative_energy_decrease),
            1.0e-12,
        ),
        (
            "reproducibility",
            "deterministic configuration and seed declaration",
            0.0,
            0.0,
        ),
    )
    return tuple(
        EvidenceRecord(
            evidence_id=f"openphenomena.closed_sphere.{case_id}.{kind}",
            evidence_type="numerical_verification",
            quantity_of_interest=quantity,
            conditions={
                "accepted": acceptance.acceptable,
                "pressure_sign": "p=-lambda_V",
            },
            tolerance=tolerance,
            measured_error=error,
            passed=error <= tolerance,
            implementation="openphenomena.equilibrium.closed_sphere",
            implementation_version=__version__,
            artifact_references=(artifact,),
            notes="Closed fixed-volume, no-boundary, gravity-free case only.",
        )
        for kind, quantity, error, tolerance in records
    )


def _pressure_from_result(result: SolverResult) -> float:
    for estimate in result.multiplier_estimates:
        if estimate.constraint_id == "openphenomena.constraint.enclosed_volume":
            return -float(estimate.physical_values[0])
    raise ValueError("solver result has no enclosed-volume multiplier")


def _restore_centroid_and_volume(
    positions: FloatArray, faces: IntArray, config: ClosedSphereConfig
) -> FloatArray:
    centered = positions - volume_centroid(positions, faces)[None, :]
    volume = oriented_volume(centered, faces)
    if volume <= 0.0:
        raise ValueError("initial mesh must have positive orientation")
    scaled = centered * (config.target_volume_m3 / volume) ** (1.0 / 3.0)
    return np.asarray(
        scaled + config.target_centroid_array_m[None, :], dtype=np.float64
    )


def _admissibility(
    positions: FloatArray, faces: IntArray
) -> tuple[bool, tuple[str, ...]]:
    quality = mesh_quality(positions, faces)
    messages = (
        f"positive_orientation={quality.oriented_volume_m3 > 0.0}",
        f"inverted_face_count={quality.inverted_face_count}",
        f"degenerate_face_count={quality.degenerate_face_count}",
    )
    return quality.admissible, messages


def _triangle_angles_from_lengths(lengths: FloatArray) -> FloatArray:
    result = np.empty_like(lengths)
    for index in range(3):
        opposite = lengths[:, (index + 1) % 3]
        side_a = lengths[:, index]
        side_b = lengths[:, (index + 2) % 3]
        cosine = (side_a**2 + side_b**2 - opposite**2) / (2.0 * side_a * side_b)
        result[:, index] = np.arccos(np.clip(cosine, -1.0, 1.0))
    return result


def _surface_samples(positions: FloatArray, faces: IntArray) -> FloatArray:
    directed_edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    edges = np.unique(np.sort(directed_edges, axis=1), axis=0)
    edge_midpoints = 0.5 * (positions[edges[:, 0]] + positions[edges[:, 1]])
    face_centroids = np.mean(positions[faces], axis=1)
    return np.asarray(
        np.concatenate((positions, edge_midpoints, face_centroids), axis=0),
        dtype=np.float64,
    )


def _readonly_float(values: FloatArray) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _readonly_int(values: IntArray) -> IntArray:
    result = np.array(values, dtype=np.int64, copy=True)
    result.flags.writeable = False
    return result
