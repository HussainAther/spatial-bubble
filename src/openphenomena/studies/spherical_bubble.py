"""Canonical static spherical soap-bubble reference study.

Physics first
-------------
For the outward normal of a convex sphere, this project defines both principal
curvatures and mean curvature as positive: k1 = k2 = H = 1/R. Gaussian
curvature is K = 1/R**2. Pressure jump is inside minus outside. A thin soap film
has two interfaces of equal surface tension, hence delta_p = 4 gamma H.

Numerics
--------
Meshes are recursively subdivided icosahedra projected to the exact radius.
Mean curvature uses the mixed-area cotangent Laplace--Beltrami operator and
Gaussian curvature uses angle defect over mixed Voronoi area (Meyer et al.,
2003). Pointwise thin-film optics uses a tangent-plane engineering
approximation and the coherent Fresnel slab kernel.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from openphenomena import __version__
from openphenomena.data import (
    Dimension,
    Domain,
    EvidenceRecord,
    Fidelity,
    Field,
    FieldAssociation,
    FieldDescriptor,
    Provenance,
    Uncertainty,
    ValidationStatus,
)
from openphenomena.optics import thin_film_phase_thickness, thin_film_reflectance
from openphenomena.surface import analyze_surface

DIMENSIONLESS: Dimension = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
LENGTH: Dimension = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
AREA: Dimension = (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
INV_LENGTH: Dimension = (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
INV_AREA: Dimension = (-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
PRESSURE: Dimension = (-1.0, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0)
SURFACE_TENSION: Dimension = (0.0, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0)
SPECTRAL_RADIANCE: Dimension = (-1.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0)

MEYER_CITATION = (
    "Meyer et al. (2003), Discrete Differential-Geometry Operators for "
    "Triangulated 2-Manifolds, doi:10.1007/978-3-662-05105-4_2"
)


@dataclass(frozen=True, slots=True)
class StaticBubbleConfig:
    """Complete SI configuration for the canonical study."""

    radius_m: float = 0.01
    surface_tension_n_per_m: float = 0.03
    external_pressure_pa: float = 101_325.0
    internal_pressure_pa: float = 101_337.0
    film_thickness_m: float = 450e-9
    incident_refractive_index: float = 1.0
    film_refractive_index: float = 1.333
    exit_refractive_index: float = 1.0
    wavelength_min_m: float = 380e-9
    wavelength_max_m: float = 780e-9
    wavelength_samples: int = 81
    refinement_levels: tuple[int, ...] = (1, 2, 3, 4)
    illumination_direction: tuple[float, float, float] = (1.0, 2.0, 3.0)
    incident_spectral_radiance_w_m2_sr_m: float = 1.0
    mesh_generation_method: str = "recursive projected icosphere"
    curvature_estimation_method: str = (
        "mixed-area cotangent Laplace-Beltrami and angle defect"
    )
    illumination_assumption: str = (
        "two opposed incoherent collimated, equal-energy, unpolarized beams; "
        "local tangent-plane response; no inter-point transport"
    )
    random_seed: int = 0

    def __post_init__(self) -> None:
        positive = (
            self.radius_m,
            self.surface_tension_n_per_m,
            self.external_pressure_pa,
            self.internal_pressure_pa,
            self.film_thickness_m,
            self.incident_refractive_index,
            self.film_refractive_index,
            self.exit_refractive_index,
            self.wavelength_min_m,
            self.wavelength_max_m,
        )
        if any(not np.isfinite(item) or item <= 0.0 for item in positive):
            raise ValueError(
                "all dimensional/material configuration values must be positive"
            )
        if self.wavelength_max_m <= self.wavelength_min_m:
            raise ValueError("wavelength_max_m must exceed wavelength_min_m")
        if self.wavelength_samples < 3 or len(self.refinement_levels) < 3:
            raise ValueError(
                "at least three wavelengths and refinement levels are required"
            )
        if tuple(sorted(set(self.refinement_levels))) != self.refinement_levels:
            raise ValueError("refinement_levels must be sorted and unique")
        expected = 4.0 * self.surface_tension_n_per_m / self.radius_m
        configured = self.internal_pressure_pa - self.external_pressure_pa
        if not np.isclose(configured, expected, rtol=0.0, atol=1e-12):
            raise ValueError(
                "configured pressure difference must equal 4*surface_tension/radius"
            )

    def as_mapping(self) -> Mapping[str, object]:
        return asdict(self)

    @property
    def wavelengths_m(self) -> npt.NDArray[np.float64]:
        return np.linspace(
            self.wavelength_min_m,
            self.wavelength_max_m,
            self.wavelength_samples,
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class AnalyticSphere:
    principal_curvature_per_m: float
    mean_curvature_per_m: float
    gaussian_curvature_per_m2: float
    pressure_jump_pa: float


@dataclass(frozen=True, slots=True)
class ConvergenceRow:
    refinement_level: int
    vertex_count: int
    face_count: int
    characteristic_edge_length_m: float
    mean_curvature_l1_error_per_m: float
    mean_curvature_l2_error_per_m: float
    mean_curvature_linf_error_per_m: float
    pressure_l2_error_pa: float
    observed_l1_rate: float | None
    observed_l2_rate: float | None
    observed_linf_rate: float | None
    runtime_s: float
    peak_python_memory_bytes: int


def analytic_model(config: StaticBubbleConfig) -> AnalyticSphere:
    """Return exact sphere and two-interface Young--Laplace quantities."""

    inverse_radius = 1.0 / config.radius_m
    return AnalyticSphere(
        principal_curvature_per_m=inverse_radius,
        mean_curvature_per_m=inverse_radius,
        gaussian_curvature_per_m2=inverse_radius**2,
        pressure_jump_pa=4.0 * config.surface_tension_n_per_m * inverse_radius,
    )


def build_reference_domain(
    config: StaticBubbleConfig,
    refinement_level: int,
    positions_m: npt.NDArray[np.float64],
    faces: npt.NDArray[np.int64],
    analytic: AnalyticSphere,
    *,
    runtime_s: float = 0.0,
    peak_python_memory_bytes: int = 0,
) -> Domain:
    """Compute all canonical geometry, pressure, optics, and display fields."""

    geometry = analyze_surface(positions_m, faces)
    vertex_count = len(positions_m)
    wavelength = config.wavelengths_m
    wavelength_count = len(wavelength)
    analytic_principal = np.full((vertex_count, 2), analytic.principal_curvature_per_m)
    analytic_h = np.full(vertex_count, analytic.mean_curvature_per_m)
    analytic_k = np.full(vertex_count, analytic.gaussian_curvature_per_m2)
    mean_abs_error = np.abs(geometry.mean_curvature_per_m - analytic_h)
    mean_rel_error = mean_abs_error / abs(analytic.mean_curvature_per_m)
    gaussian_abs_error = np.abs(geometry.gaussian_curvature_per_m2 - analytic_k)
    gaussian_rel_error = gaussian_abs_error / abs(analytic.gaussian_curvature_per_m2)
    discrete_pressure = (
        4.0 * config.surface_tension_n_per_m * geometry.mean_curvature_per_m
    )
    analytic_pressure = np.full(vertex_count, analytic.pressure_jump_pa)
    pressure_error = discrete_pressure - analytic_pressure

    light_axis = np.asarray(config.illumination_direction, dtype=np.float64)
    light_axis /= np.linalg.norm(light_axis)
    incidence_cosine = np.abs(geometry.vertex_normals @ light_axis)
    incidence_cosine = np.clip(incidence_cosine, 1e-12, 1.0)
    incidence_angle = np.arccos(incidence_cosine)
    broadcast_angle = incidence_angle[:, None]
    broadcast_wavelength = wavelength[None, :]
    optical_phase = thin_film_phase_thickness(
        broadcast_wavelength,
        config.film_thickness_m,
        config.film_refractive_index,
        incident_refractive_index=config.incident_refractive_index,
        incidence_angle_rad=broadcast_angle,
    )
    reflectance_s = thin_film_reflectance(
        broadcast_wavelength,
        config.film_thickness_m,
        config.film_refractive_index,
        incident_refractive_index=config.incident_refractive_index,
        exit_refractive_index=config.exit_refractive_index,
        incidence_angle_rad=broadcast_angle,
        polarization="s",
    )
    reflectance_p = thin_film_reflectance(
        broadcast_wavelength,
        config.film_thickness_m,
        config.film_refractive_index,
        incident_refractive_index=config.incident_refractive_index,
        exit_refractive_index=config.exit_refractive_index,
        incidence_angle_rad=broadcast_angle,
        polarization="p",
    )
    reflectance_unpolarized = thin_film_reflectance(
        broadcast_wavelength,
        config.film_thickness_m,
        config.film_refractive_index,
        incident_refractive_index=config.incident_refractive_index,
        exit_refractive_index=config.exit_refractive_index,
        incidence_angle_rad=broadcast_angle,
        polarization="unpolarized",
    )
    illuminant = np.full(
        wavelength_count,
        config.incident_spectral_radiance_w_m2_sr_m,
        dtype=np.float64,
    )
    reflected_radiance = reflectance_unpolarized * illuminant[None, :]
    xyz, linear_rgb, tone_mapped_rgb = _display_layers(
        wavelength, reflected_radiance, illuminant
    )

    model_provenance = Provenance(
        activity="analytic static spherical bubble model",
        implementation="openphenomena.studies.spherical_bubble.analytic_model",
        implementation_version=__version__,
        parameters=config.as_mapping(),
        citations=("Young-Laplace equation; two equal-tension interfaces",),
    )
    geometry_provenance = Provenance(
        activity="discrete surface geometry",
        implementation="openphenomena.surface.geometry.analyze_surface",
        implementation_version=__version__,
        source_ids=("geometry.position",),
        parameters={
            "refinement_level": refinement_level,
            "mixed_area": True,
            "normal_convention": "outward",
        },
        citations=(MEYER_CITATION,),
    )
    optics_provenance = Provenance(
        activity="local coherent thin-film response",
        implementation="openphenomena.optics.thin_film",
        implementation_version=__version__,
        source_ids=("film.thickness", "geometry.normal.vertex", "optics.wavelength"),
        parameters={
            "stack": [
                config.incident_refractive_index,
                config.film_refractive_index,
                config.exit_refractive_index,
            ],
            "illumination": config.illumination_assumption,
        },
        citations=("Coherent single-layer Fresnel amplitude summation",),
    )
    display_provenance = Provenance(
        activity="approximate colorimetric display transform",
        implementation="openphenomena.studies.spherical_bubble._display_layers",
        implementation_version=__version__,
        source_ids=("radiometry.spectral_reflected_radiance",),
        parameters={
            "observer": "Wyman analytic CIE 1931 approximation",
            "tone_map": "Reinhard+sRGB",
        },
        citations=(
            "Wyman, Sloan, Shirley (2013), Simple Analytic Approximations "
            "to the CIE XYZ Color Matching Functions",
        ),
    )
    unquantified = Uncertainty(False, "Uncertainty has not yet been quantified.")
    fields: dict[str, Field] = {}

    def add(
        semantic_id: str,
        values: npt.ArrayLike,
        association: FieldAssociation,
        unit: str,
        dimension: Dimension,
        model: str,
        implementation: str,
        fidelity: Fidelity,
        status: ValidationStatus,
        provenance: Provenance,
        description: str,
        *,
        components: tuple[str, ...] = (),
        axes: tuple[str, ...] = (),
    ) -> None:
        array = np.asarray(values)
        descriptor = FieldDescriptor(
            semantic_id=semantic_id,
            association=association,
            unit=unit,
            unit_dimension=dimension,
            shape=array.shape,
            dtype=array.dtype.str,
            coordinate_frame="bubble_centered_cartesian_m",
            generating_model=model,
            generating_implementation=implementation,
            fidelity=fidelity,
            validation_status=status,
            uncertainty=unquantified,
            description=description,
            component_names=components,
            coordinate_axes=axes,
        )
        fields[semantic_id] = Field(descriptor, array, (provenance,))

    pv = Fidelity.PHYSICALLY_VALIDATED
    ea = Fidelity.ENGINEERING_APPROXIMATION
    vo = Fidelity.VISUALIZATION_ONLY
    validated = ValidationStatus.VALIDATED
    verified = ValidationStatus.VERIFIED
    add(
        "geometry.position",
        positions_m,
        FieldAssociation.VERTEX,
        "m",
        LENGTH,
        "analytic sphere",
        "projected icosphere",
        ea,
        verified,
        geometry_provenance,
        "Projected sphere vertices",
        components=("x", "y", "z"),
    )
    add(
        "geometry.area.face",
        geometry.face_areas_m2,
        FieldAssociation.FACE,
        "m^2",
        AREA,
        "triangle geometry",
        "cross-product area",
        ea,
        verified,
        geometry_provenance,
        "Planar triangle areas",
    )
    add(
        "geometry.normal.face",
        geometry.face_normals,
        FieldAssociation.FACE,
        "1",
        DIMENSIONLESS,
        "triangle geometry",
        "oriented cross product",
        ea,
        verified,
        geometry_provenance,
        "Outward unit face normals",
        components=("x", "y", "z"),
    )
    add(
        "geometry.normal.vertex",
        geometry.vertex_normals,
        FieldAssociation.VERTEX,
        "1",
        DIMENSIONLESS,
        "triangle geometry",
        "area-weighted normal",
        ea,
        verified,
        geometry_provenance,
        "Area-weighted outward unit vertex normals",
        components=("x", "y", "z"),
    )
    add(
        "geometry.area.vertex_mixed",
        geometry.vertex_areas_m2,
        FieldAssociation.VERTEX,
        "m^2",
        AREA,
        "triangle geometry",
        "mixed Voronoi area",
        ea,
        verified,
        geometry_provenance,
        "Mixed vertex dual areas",
    )
    add(
        "geometry.mean_curvature.discrete",
        geometry.mean_curvature_per_m,
        FieldAssociation.VERTEX,
        "m^-1",
        INV_LENGTH,
        "discrete surface geometry",
        "mixed-area cotangent Laplacian",
        ea,
        verified,
        geometry_provenance,
        "Signed mean curvature; convex outward sphere is positive",
    )
    add(
        "geometry.gaussian_curvature.discrete",
        geometry.gaussian_curvature_per_m2,
        FieldAssociation.VERTEX,
        "m^-2",
        INV_AREA,
        "discrete surface geometry",
        "angle defect over mixed area",
        ea,
        verified,
        geometry_provenance,
        "Discrete Gaussian curvature",
    )
    add(
        "geometry.principal_curvature.analytic",
        analytic_principal,
        FieldAssociation.VERTEX,
        "m^-1",
        INV_LENGTH,
        "analytic sphere",
        "k1=k2=1/R",
        pv,
        validated,
        model_provenance,
        "Analytic principal curvatures",
        components=("k1", "k2"),
    )
    add(
        "geometry.mean_curvature.analytic",
        analytic_h,
        FieldAssociation.VERTEX,
        "m^-1",
        INV_LENGTH,
        "analytic sphere",
        "1/R",
        pv,
        validated,
        model_provenance,
        "Analytic positive mean curvature",
    )
    add(
        "geometry.gaussian_curvature.analytic",
        analytic_k,
        FieldAssociation.VERTEX,
        "m^-2",
        INV_AREA,
        "analytic sphere",
        "1/R^2",
        pv,
        validated,
        model_provenance,
        "Analytic Gaussian curvature",
    )
    add(
        "geometry.mean_curvature.absolute_error",
        mean_abs_error,
        FieldAssociation.VERTEX,
        "m^-1",
        INV_LENGTH,
        "comparison",
        "absolute difference",
        ea,
        verified,
        geometry_provenance,
        "Absolute discrete mean-curvature error",
    )
    add(
        "geometry.mean_curvature.relative_error",
        mean_rel_error,
        FieldAssociation.VERTEX,
        "1",
        DIMENSIONLESS,
        "comparison",
        "relative difference",
        ea,
        verified,
        geometry_provenance,
        "Relative discrete mean-curvature error",
    )
    add(
        "geometry.gaussian_curvature.absolute_error",
        gaussian_abs_error,
        FieldAssociation.VERTEX,
        "m^-2",
        INV_AREA,
        "comparison",
        "absolute difference",
        ea,
        verified,
        geometry_provenance,
        "Absolute discrete Gaussian-curvature error",
    )
    add(
        "geometry.gaussian_curvature.relative_error",
        gaussian_rel_error,
        FieldAssociation.VERTEX,
        "1",
        DIMENSIONLESS,
        "comparison",
        "relative difference",
        ea,
        verified,
        geometry_provenance,
        "Relative discrete Gaussian-curvature error",
    )
    add(
        "mechanics.pressure_jump.discrete",
        discrete_pressure,
        FieldAssociation.VERTEX,
        "Pa",
        PRESSURE,
        "two-interface Young-Laplace",
        "4*gamma*H_discrete",
        ea,
        verified,
        geometry_provenance,
        "Inside-minus-outside pressure from discrete H",
    )
    add(
        "mechanics.pressure_jump.analytic",
        analytic_pressure,
        FieldAssociation.VERTEX,
        "Pa",
        PRESSURE,
        "two-interface Young-Laplace",
        "4*gamma/R",
        pv,
        validated,
        model_provenance,
        "Analytic inside-minus-outside pressure",
    )
    add(
        "mechanics.pressure_jump.error",
        pressure_error,
        FieldAssociation.VERTEX,
        "Pa",
        PRESSURE,
        "comparison",
        "discrete minus analytic",
        ea,
        verified,
        geometry_provenance,
        "Signed pressure-jump error",
    )
    add(
        "film.thickness",
        np.full(vertex_count, config.film_thickness_m),
        FieldAssociation.VERTEX,
        "m",
        LENGTH,
        "prescribed static film",
        "uniform assignment",
        ea,
        ValidationStatus.UNVALIDATED,
        model_provenance,
        "Uniform total film thickness",
    )
    add(
        "mechanics.surface_tension",
        np.asarray(config.surface_tension_n_per_m),
        FieldAssociation.GLOBAL,
        "N m^-1",
        SURFACE_TENSION,
        "prescribed material",
        "constant",
        ea,
        ValidationStatus.UNVALIDATED,
        model_provenance,
        "Equal surface tension on both interfaces",
    )
    add(
        "mechanics.pressure.internal",
        np.asarray(config.internal_pressure_pa),
        FieldAssociation.GLOBAL,
        "Pa",
        PRESSURE,
        "prescribed equilibrium",
        "constant",
        pv,
        validated,
        model_provenance,
        "Internal absolute pressure",
    )
    add(
        "mechanics.pressure.external",
        np.asarray(config.external_pressure_pa),
        FieldAssociation.GLOBAL,
        "Pa",
        PRESSURE,
        "prescribed environment",
        "constant",
        pv,
        validated,
        model_provenance,
        "External absolute pressure",
    )
    add(
        "optics.refractive_index.incident",
        np.asarray(config.incident_refractive_index),
        FieldAssociation.GLOBAL,
        "1",
        DIMENSIONLESS,
        "prescribed air-film-air stack",
        "constant real refractive index",
        ea,
        ValidationStatus.UNVALIDATED,
        optics_provenance,
        "Incident air refractive index",
    )
    add(
        "optics.refractive_index.film",
        np.asarray(config.film_refractive_index),
        FieldAssociation.GLOBAL,
        "1",
        DIMENSIONLESS,
        "prescribed air-film-air stack",
        "constant real refractive index",
        ea,
        ValidationStatus.UNVALIDATED,
        optics_provenance,
        "Soap-film refractive index",
    )
    add(
        "optics.refractive_index.exit",
        np.asarray(config.exit_refractive_index),
        FieldAssociation.GLOBAL,
        "1",
        DIMENSIONLESS,
        "prescribed air-film-air stack",
        "constant real refractive index",
        ea,
        ValidationStatus.UNVALIDATED,
        optics_provenance,
        "Exit air refractive index",
    )
    add(
        "optics.wavelength",
        wavelength,
        FieldAssociation.GLOBAL,
        "m",
        LENGTH,
        "spectral sampling",
        "uniform wavelength grid",
        pv,
        ValidationStatus.NOT_APPLICABLE,
        optics_provenance,
        "Vacuum wavelength samples",
        axes=("wavelength",),
    )
    add(
        "optics.incidence_angle",
        incidence_angle,
        FieldAssociation.VERTEX,
        "rad",
        DIMENSIONLESS,
        "two-sided collimated illumination",
        "acos(abs(n dot d))",
        ea,
        verified,
        optics_provenance,
        "Local angle from outward normal",
    )
    add(
        "optics.phase_thickness",
        optical_phase,
        FieldAssociation.VERTEX,
        "rad",
        DIMENSIONLESS,
        "coherent plane-parallel slab",
        "Fresnel phase kernel",
        ea,
        verified,
        optics_provenance,
        "One-way optical phase thickness",
        axes=("vertex", "wavelength"),
    )
    add(
        "optics.reflectance.s",
        reflectance_s,
        FieldAssociation.VERTEX,
        "1",
        DIMENSIONLESS,
        "local plane-parallel slab",
        "coherent Fresnel kernel",
        ea,
        verified,
        optics_provenance,
        "s-polarized spectral power reflectance",
        axes=("vertex", "wavelength"),
    )
    add(
        "optics.reflectance.p",
        reflectance_p,
        FieldAssociation.VERTEX,
        "1",
        DIMENSIONLESS,
        "local plane-parallel slab",
        "coherent Fresnel kernel",
        ea,
        verified,
        optics_provenance,
        "p-polarized spectral power reflectance",
        axes=("vertex", "wavelength"),
    )
    add(
        "optics.reflectance.unpolarized",
        reflectance_unpolarized,
        FieldAssociation.VERTEX,
        "1",
        DIMENSIONLESS,
        "local plane-parallel slab",
        "equal incoherent s/p average",
        ea,
        verified,
        optics_provenance,
        "Authoritative wavelength-resolved reflectance",
        axes=("vertex", "wavelength"),
    )
    add(
        "radiometry.incident_spectral_radiance",
        illuminant,
        FieldAssociation.GLOBAL,
        "W m^-2 sr^-1 m^-1",
        SPECTRAL_RADIANCE,
        "prescribed equal-energy illuminant",
        "constant spectrum",
        ea,
        ValidationStatus.UNVALIDATED,
        optics_provenance,
        "Incident spectral radiance",
        axes=("wavelength",),
    )
    add(
        "radiometry.spectral_reflected_radiance",
        reflected_radiance,
        FieldAssociation.VERTEX,
        "W m^-2 sr^-1 m^-1",
        SPECTRAL_RADIANCE,
        "local reflection",
        "R_unpolarized times L_incident",
        ea,
        verified,
        optics_provenance,
        "Illuminant-weighted reflected spectral radiance",
        axes=("vertex", "wavelength"),
    )
    add(
        "color.cie_xyz",
        xyz,
        FieldAssociation.VERTEX,
        "1",
        DIMENSIONLESS,
        "display colorimetry",
        "approximate CIE integration",
        vo,
        ValidationStatus.NOT_APPLICABLE,
        display_provenance,
        "Colorimetric XYZ relative to illuminant",
        components=("X", "Y", "Z"),
    )
    add(
        "color.linear_srgb",
        linear_rgb,
        FieldAssociation.VERTEX,
        "1",
        DIMENSIONLESS,
        "display colorimetry",
        "XYZ to linear sRGB",
        vo,
        ValidationStatus.NOT_APPLICABLE,
        display_provenance,
        "Linear display RGB; may be out of gamut",
        components=("R", "G", "B"),
    )
    add(
        "color.tonemapped_srgb",
        tone_mapped_rgb,
        FieldAssociation.VERTEX,
        "1",
        DIMENSIONLESS,
        "display visualization",
        "clip+Reinhard+sRGB OETF",
        vo,
        ValidationStatus.NOT_APPLICABLE,
        display_provenance,
        "Tone-mapped visualization RGB",
        components=("R", "G", "B"),
    )

    return Domain(
        domain_id=f"bubble_surface_refinement_{refinement_level}",
        kind="oriented_triangular_surface",
        coordinate_frame="bubble_centered_cartesian_m",
        positions_m=positions_m,
        faces=faces,
        fields=fields,
        metadata={
            "refinement_level": refinement_level,
            "normal_convention": "outward",
            "curvature_convention": "convex outward sphere has k1=k2=H=+1/R",
            "pressure_convention": "pressure_jump = internal - external",
            "characteristic_edge_length_m": geometry.characteristic_edge_length_m,
            "runtime_s": runtime_s,
            "peak_python_memory_bytes": peak_python_memory_bytes,
        },
    )


def convergence_rows(domains: tuple[Domain, ...]) -> tuple[ConvergenceRow, ...]:
    """Compute area-weighted error norms and observed refinement rates."""

    preliminary: list[ConvergenceRow] = []
    for domain in domains:
        weights = np.asarray(
            domain.fields["geometry.area.vertex_mixed"].values, dtype=np.float64
        )
        error = np.asarray(
            domain.fields["geometry.mean_curvature.absolute_error"].values,
            dtype=np.float64,
        )
        pressure_error = np.abs(
            np.asarray(
                domain.fields["mechanics.pressure_jump.error"].values,
                dtype=np.float64,
            )
        )
        weight_sum = float(np.sum(weights))
        l1 = float(np.sum(weights * error) / weight_sum)
        l2 = float(np.sqrt(np.sum(weights * error**2) / weight_sum))
        linf = float(np.max(error))
        pressure_l2 = float(np.sqrt(np.sum(weights * pressure_error**2) / weight_sum))
        preliminary.append(
            ConvergenceRow(
                refinement_level=cast(int, domain.metadata["refinement_level"]),
                vertex_count=len(domain.positions_m),
                face_count=len(domain.faces),
                characteristic_edge_length_m=cast(
                    float, domain.metadata["characteristic_edge_length_m"]
                ),
                mean_curvature_l1_error_per_m=l1,
                mean_curvature_l2_error_per_m=l2,
                mean_curvature_linf_error_per_m=linf,
                pressure_l2_error_pa=pressure_l2,
                observed_l1_rate=None,
                observed_l2_rate=None,
                observed_linf_rate=None,
                runtime_s=cast(float, domain.metadata["runtime_s"]),
                peak_python_memory_bytes=cast(
                    int, domain.metadata["peak_python_memory_bytes"]
                ),
            )
        )
    result: list[ConvergenceRow] = []
    for index, row in enumerate(preliminary):
        if index == 0:
            result.append(row)
            continue
        previous = preliminary[index - 1]
        ratio = previous.characteristic_edge_length_m / row.characteristic_edge_length_m
        result.append(
            ConvergenceRow(
                **{
                    **asdict(row),
                    "observed_l1_rate": _observed_rate(
                        previous.mean_curvature_l1_error_per_m,
                        row.mean_curvature_l1_error_per_m,
                        ratio,
                    ),
                    "observed_l2_rate": _observed_rate(
                        previous.mean_curvature_l2_error_per_m,
                        row.mean_curvature_l2_error_per_m,
                        ratio,
                    ),
                    "observed_linf_rate": _observed_rate(
                        previous.mean_curvature_linf_error_per_m,
                        row.mean_curvature_linf_error_per_m,
                        ratio,
                    ),
                }
            )
        )
    return tuple(result)


def validate_reference_study(
    config: StaticBubbleConfig,
    domains: tuple[Domain, ...],
    convergence: tuple[ConvergenceRow, ...],
    export_roundtrip_error: float,
) -> tuple[EvidenceRecord, ...]:
    """Create explicit machine-readable evidence records."""

    finest = domains[-1]
    finest_row = convergence[-1]
    face_areas = np.asarray(
        finest.fields["geometry.area.face"].values, dtype=np.float64
    )
    analytic_area = 4.0 * np.pi * config.radius_m**2
    area_error = abs(float(np.sum(face_areas)) - analytic_area) / analytic_area
    curvature_relative_l2 = finest_row.mean_curvature_l2_error_per_m * config.radius_m
    pressure_relative_l2 = finest_row.pressure_l2_error_pa / (
        4.0 * config.surface_tension_n_per_m / config.radius_m
    )
    rs = np.asarray(finest.fields["optics.reflectance.s"].values, dtype=np.float64)
    rp = np.asarray(finest.fields["optics.reflectance.p"].values, dtype=np.float64)
    ru = np.asarray(
        finest.fields["optics.reflectance.unpolarized"].values, dtype=np.float64
    )
    bounds_error = float(
        max(0.0, -np.min(rs), np.max(rs) - 1.0, -np.min(rp), np.max(rp) - 1.0)
    )
    polarization_error = float(np.max(np.abs(ru - 0.5 * (rs + rp))))
    # The level-1 icosphere is vertex-transitive and the cotan mean curvature is
    # exact to roundoff there. It is not a legitimate baseline for an observed
    # asymptotic rate. Use the final two transitions (levels 2->3->4) while
    # retaining every measured rate in the convergence table.
    asymptotic_rows = convergence[2:]
    rates = [
        value
        for row in asymptotic_rows
        for value in (row.observed_l1_rate, row.observed_l2_rate)
        if value is not None
    ]
    convergence_deficit = max(0.0, -min(rates)) if rates else float("inf")

    def evidence(
        evidence_id: str,
        quantity: str,
        tolerance: float,
        error: float,
        conditions: Mapping[str, object],
        notes: str,
        artifacts: tuple[str, ...],
    ) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=evidence_id,
            evidence_type="verification/validation evidence",
            quantity_of_interest=quantity,
            conditions=conditions,
            tolerance=tolerance,
            measured_error=error,
            passed=error <= tolerance,
            notes=notes,
            implementation="openphenomena.plugins.soap_bubble",
            implementation_version=__version__,
            artifact_references=artifacts,
        )

    return (
        evidence(
            "sphere_geometry",
            "relative polygonal surface-area error",
            0.01,
            area_error,
            {"refinement_level": finest_row.refinement_level},
            "Projected vertices are exact; planar faces approximate spherical area.",
            (
                "convergence/convergence.json",
                "exports/bubble_finest.vtp",
                "reports/validation.md",
            ),
        ),
        evidence(
            "curvature_accuracy",
            "relative area-weighted L2 mean-curvature error",
            0.02,
            curvature_relative_l2,
            {"operator": config.curvature_estimation_method},
            "Discrete operator verification, not experimental validation.",
            (
                "convergence/convergence.json",
                "exports/bubble_finest.vtp",
                "reports/validation.md",
            ),
        ),
        evidence(
            "young_laplace_pressure",
            "relative L2 pressure-jump error",
            0.02,
            pressure_relative_l2,
            {"pressure_convention": "internal minus external", "interfaces": 2},
            "Pressure error follows discrete mean-curvature error.",
            (
                "convergence/convergence.json",
                "exports/bubble_finest.vtp",
                "reports/validation.md",
            ),
        ),
        evidence(
            "optics_bounds",
            "spectral reflectance bound violation",
            1e-12,
            bounds_error,
            {"stack": "air-film-air", "lossless": True},
            "Local tangent-plane application is EA.",
            (
                "exports/bubble_finest.vtp",
                "exports/visualization_manifest.json",
                "reports/validation.md",
            ),
        ),
        evidence(
            "polarization_consistency",
            "max |R_unpolarized-(Rs+Rp)/2|",
            1e-14,
            polarization_error,
            {"illumination": "unpolarized"},
            "Checks the incoherent polarization average.",
            (
                "exports/bubble_finest.vtp",
                "reports/validation.md",
            ),
        ),
        evidence(
            "mesh_convergence",
            "non-positive L1/L2 convergence-rate deficit",
            0.0,
            convergence_deficit,
            {
                "all_levels": list(config.refinement_levels),
                "asymptotic_rate_window": list(config.refinement_levels[1:]),
            },
            "Pass means measured L1/L2 errors decrease over levels 2-4. Level "
            "1 is exact to roundoff by icosphere symmetry and is excluded from "
            "the rate claim.",
            (
                "convergence/convergence.csv",
                "convergence/convergence.json",
                "reports/convergence.md",
                "reports/validation.md",
            ),
        ),
        evidence(
            "export_roundtrip",
            "maximum VTP round-trip absolute error",
            1e-12,
            export_roundtrip_error,
            {"format": "VTK XML PolyData ASCII"},
            "Checks geometry, topology, and exported numerical arrays.",
            (
                "exports/bubble_finest.vtp",
                "exports/visualization_manifest.json",
                "reports/validation.md",
            ),
        ),
    )


def _observed_rate(previous: float, current: float, mesh_ratio: float) -> float | None:
    if previous <= 0.0 or current <= 0.0 or mesh_ratio <= 1.0:
        return None
    return float(np.log(previous / current) / np.log(mesh_ratio))


def _display_layers(
    wavelength_m: npt.NDArray[np.float64],
    reflected: npt.NDArray[np.float64],
    illuminant: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    wavelength_nm = wavelength_m * 1e9
    x_bar, y_bar, z_bar = _cie_xyz_approximation(wavelength_nm)
    normalization = float(np.trapezoid(illuminant * y_bar, wavelength_m))
    xyz = (
        np.stack(
            [
                np.trapezoid(reflected * x_bar[None, :], wavelength_m, axis=1),
                np.trapezoid(reflected * y_bar[None, :], wavelength_m, axis=1),
                np.trapezoid(reflected * z_bar[None, :], wavelength_m, axis=1),
            ],
            axis=1,
        )
        / normalization
    )
    xyz_to_srgb = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    )
    linear_rgb = xyz @ xyz_to_srgb.T
    positive = np.maximum(linear_rgb, 0.0)
    reinhard = positive / (1.0 + positive)
    tone_mapped = np.where(
        reinhard <= 0.0031308,
        12.92 * reinhard,
        1.055 * np.power(reinhard, 1.0 / 2.4) - 0.055,
    )
    return xyz, linear_rgb, tone_mapped


def _cie_xyz_approximation(
    wavelength_nm: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    def gaussian(center: float, left: float, right: float) -> npt.NDArray[np.float64]:
        scale = np.where(wavelength_nm < center, left, right)
        return np.exp(-0.5 * ((wavelength_nm - center) / scale) ** 2)

    x_bar = (
        1.056 * gaussian(599.8, 37.9, 31.0)
        + 0.362 * gaussian(442.0, 16.0, 26.7)
        - 0.065 * gaussian(501.1, 20.4, 26.2)
    )
    y_bar = 0.821 * gaussian(568.8, 46.9, 40.5) + 0.286 * gaussian(530.9, 16.3, 31.1)
    z_bar = 1.217 * gaussian(437.0, 11.8, 36.0) + 0.681 * gaussian(459.0, 26.0, 13.8)
    return x_bar, y_bar, z_bar
