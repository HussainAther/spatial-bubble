"""Backend-independent discrete geometry on triangular surfaces."""

from openphenomena.surface.exact import (
    SurfaceFunctionalEvaluation,
    area_gradient,
    centroid_jacobian,
    evaluate_surface_functionals,
    oriented_volume,
    surface_energy_gradient,
    total_area,
    total_surface_energy,
    triangle_areas,
    volume_centroid,
    volume_gradient,
)
from openphenomena.surface.geometry import SurfaceGeometry, analyze_surface
from openphenomena.surface.scaling import EquilibriumScales

__all__ = [
    "EquilibriumScales",
    "SurfaceFunctionalEvaluation",
    "SurfaceGeometry",
    "analyze_surface",
    "area_gradient",
    "centroid_jacobian",
    "evaluate_surface_functionals",
    "oriented_volume",
    "surface_energy_gradient",
    "total_area",
    "total_surface_energy",
    "triangle_areas",
    "volume_centroid",
    "volume_gradient",
]
