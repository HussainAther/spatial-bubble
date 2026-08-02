"""Bundled canonical static soap-bubble reference plugin."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from openphenomena import __version__
from openphenomena.core import Capability, CapabilityKind, PluginManifest
from openphenomena.data import Domain
from openphenomena.export import read_vtp, write_gltf, write_ply, write_vtp
from openphenomena.mesh import create_icosphere
from openphenomena.studies.spherical_bubble import (
    analytic_model,
    build_reference_domain,
    validate_reference_study,
)


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Paths and measured error from scientific/visual exports."""

    scientific_path: Path
    blender_path: Path
    gltf_path: Path
    roundtrip_max_absolute_error: float


@dataclass(frozen=True, slots=True)
class StaticBubbleViewRecipe:
    """Presentation-only default view metadata."""

    recipe_id: str = "openphenomena.soap_bubble.view.static_iridescence"
    classification: str = "VO"
    color_field: str = "color.tonemapped_srgb"
    scientific_probe_field: str = "geometry.mean_curvature.discrete"


def export_reference_domain(domain: Domain, directory: Path) -> ExportResult:
    """Export authoritative VTP and derived Blender/glTF views."""

    directory.mkdir(parents=True, exist_ok=True)
    scientific_path = write_vtp(domain, directory / "bubble_finest.vtp")
    blender_path = write_ply(domain, directory / "bubble_blender_vo.ply")
    gltf_path = write_gltf(domain, directory / "bubble_vo.gltf")
    positions, faces, fields = read_vtp(scientific_path)
    errors = [
        float(np.max(np.abs(positions - domain.positions_m))),
        float(np.max(np.abs(faces - domain.faces))),
    ]
    for semantic_id, sampled_field in domain.fields.items():
        exported = np.asarray(fields[semantic_id], dtype=np.float64)
        source = np.asarray(sampled_field.values, dtype=np.float64).reshape(
            exported.shape
        )
        errors.append(float(np.max(np.abs(exported - source))))
    return ExportResult(
        scientific_path=scientific_path,
        blender_path=blender_path,
        gltf_path=gltf_path,
        roundtrip_max_absolute_error=max(errors),
    )


class SoapBubbleReferencePlugin:
    """Capability bundle for the canonical static spherical-bubble study."""

    _declarations = (
        (
            "openphenomena.soap_bubble.domain.icosphere",
            CapabilityKind.DOMAIN_FACTORY,
            "1.0.0",
        ),
        (
            "openphenomena.soap_bubble.model.static_sphere",
            CapabilityKind.MODEL,
            "1.0.0",
        ),
        (
            "openphenomena.soap_bubble.derived.static_fields",
            CapabilityKind.DERIVED_FIELD,
            "1.0.0",
        ),
        (
            "openphenomena.soap_bubble.validator.reference",
            CapabilityKind.VALIDATOR,
            "1.0.0",
        ),
        (
            "openphenomena.soap_bubble.export.reference",
            CapabilityKind.EXPORTER,
            "1.0.0",
        ),
        (
            "openphenomena.soap_bubble.view.static_iridescence",
            CapabilityKind.VIEW_RECIPE,
            "1.0.0",
        ),
    )

    @property
    def manifest(self) -> PluginManifest:
        return PluginManifest(
            plugin_id="openphenomena.reference.soap_bubble",
            version=__version__,
            core_api=">=0.1,<0.2",
            license_expression="Apache-2.0",
            description="Canonical static spherical soap-bubble reference study",
            capabilities=self._declarations,
            field_namespaces=(
                "geometry",
                "mechanics",
                "film",
                "optics",
                "radiometry",
                "color",
            ),
            metadata={
                "fidelity": {
                    "analytic_sphere": "PV",
                    "discrete_geometry": "EA",
                    "pointwise_curved_optics": "EA",
                    "display_color": "VO",
                }
            },
        )

    def capabilities(self) -> tuple[Capability, ...]:
        providers: tuple[object, ...] = (
            create_icosphere,
            analytic_model,
            build_reference_domain,
            validate_reference_study,
            export_reference_domain,
            StaticBubbleViewRecipe(),
        )
        return tuple(
            Capability(identifier, kind, version, provider)
            for (identifier, kind, version), provider in zip(
                self._declarations, providers, strict=True
            )
        )
