from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from openphenomena import (
    BoundaryOrientation,
    BoundarySemantics,
    Domain,
    EntitySet,
    Fidelity,
    Field,
    FieldAssociation,
    FieldDescriptor,
    Provenance,
    Uncertainty,
    ValidationStatus,
)
from openphenomena.export import read_vtp, write_vtp

LENGTH = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def make_field(sample_count: int = 3) -> Field:
    values = np.zeros((sample_count, 3), dtype=np.float64)
    descriptor = FieldDescriptor(
        semantic_id="geometry.test_vector",
        association=FieldAssociation.VERTEX,
        unit="m",
        unit_dimension=LENGTH,
        shape=values.shape,
        dtype=values.dtype.str,
        coordinate_frame="test_cartesian",
        generating_model="test model",
        generating_implementation="tests.test_data.make_field",
        fidelity=Fidelity.ENGINEERING_APPROXIMATION,
        validation_status=ValidationStatus.VERIFIED,
        uncertainty=Uncertainty(False, "Not quantified in this unit test."),
    )
    provenance = Provenance(
        activity="unit test",
        implementation="tests.test_data.make_field",
        implementation_version="1",
    )
    return Field(descriptor, values, (provenance,))


def test_units_and_schema_are_explicit() -> None:
    field = make_field()
    assert field.descriptor.unit == "m"
    assert field.descriptor.unit_dimension == LENGTH
    assert field.descriptor.coordinate_frame == "test_cartesian"
    assert field.descriptor.uncertainty.quantified is False


def test_field_shape_must_match_schema() -> None:
    field = make_field()
    with pytest.raises(ValueError, match="descriptor declares"):
        Field(field.descriptor, np.zeros((2, 3)), field.provenance)


def test_fields_and_domains_are_immutable() -> None:
    field = make_field()
    with pytest.raises(ValueError, match="read-only"):
        field.values[0, 0] = 1.0
    with pytest.raises(FrozenInstanceError):
        field.descriptor.unit = "cm"  # type: ignore[misc]

    domain = Domain(
        domain_id="test_surface",
        kind="oriented_triangular_surface",
        coordinate_frame="test_cartesian",
        positions_m=np.zeros((3, 3)),
        faces=np.array([[0, 1, 2]]),
        fields={field.descriptor.semantic_id: field},
    )
    with pytest.raises(TypeError):
        domain.fields["new.field"] = field  # type: ignore[index]
    with pytest.raises(ValueError, match="read-only"):
        domain.positions_m[0, 0] = 1.0


def test_field_registration_requires_semantic_key_match() -> None:
    field = make_field()
    with pytest.raises(ValueError, match="mapping key"):
        Domain(
            domain_id="test_surface",
            kind="oriented_triangular_surface",
            coordinate_frame="test_cartesian",
            positions_m=np.zeros((3, 3)),
            faces=np.array([[0, 1, 2]]),
            fields={"geometry.wrong_name": field},
        )


def test_edge_fields_and_oriented_boundary_sets_use_canonical_edges(
    tmp_path: Path,
) -> None:
    provenance = Provenance(
        activity="unit test",
        implementation="tests.test_data",
        implementation_version="1",
    )
    edge_values = np.array([1.0, 2.0, 3.0])
    edge_field = Field(
        FieldDescriptor(
            semantic_id="boundary.edge_weight",
            association=FieldAssociation.EDGE,
            unit="1",
            unit_dimension=(0.0,) * 7,
            shape=edge_values.shape,
            dtype=edge_values.dtype.str,
            coordinate_frame="test_cartesian",
            generating_model="test model",
            generating_implementation="tests.test_data",
            fidelity=Fidelity.ENGINEERING_APPROXIMATION,
            validation_status=ValidationStatus.VERIFIED,
        ),
        edge_values,
        (provenance,),
    )
    boundary = EntitySet(
        entity_set_id="boundary.support",
        name="Supported boundary",
        owner_domain_id="test_surface",
        association=FieldAssociation.EDGE,
        entity_indices=np.array([0, 2]),
        orientations=np.array(
            [BoundaryOrientation.ALIGNED, BoundaryOrientation.REVERSED]
        ),
        coordinate_frame="test_cartesian",
        provenance=(provenance,),
        boundary_semantics=BoundarySemantics(
            semantic_id="boundary.fixed_position",
            description="Position is fixed during equilibrium solves.",
            parameters={"components": ["x", "y", "z"]},
        ),
    )
    domain = Domain(
        domain_id="test_surface",
        kind="oriented_triangular_surface",
        coordinate_frame="test_cartesian",
        positions_m=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]]),
        fields={edge_field.descriptor.semantic_id: edge_field},
        entity_sets={boundary.entity_set_id: boundary},
    )
    np.testing.assert_array_equal(domain.edges, [[0, 1], [0, 2], [1, 2]])
    _, _, exported_fields = read_vtp(write_vtp(domain, tmp_path / "edge.vtp"))
    np.testing.assert_array_equal(
        exported_fields[edge_field.descriptor.semantic_id], edge_values
    )
    assert domain.entity_sets[boundary.entity_set_id].boundary_semantics is not None
    with pytest.raises(ValueError, match="read-only"):
        boundary.entity_indices[0] = 1
    with pytest.raises(TypeError):
        domain.entity_sets["boundary.other"] = boundary  # type: ignore[index]


def test_entity_set_indices_are_bounds_checked() -> None:
    provenance = Provenance("test", "tests.test_data", "1")
    entity_set = EntitySet(
        entity_set_id="boundary.invalid",
        name="Invalid",
        owner_domain_id="test_surface",
        association=FieldAssociation.EDGE,
        entity_indices=np.array([3]),
        orientations=np.array([BoundaryOrientation.ALIGNED]),
        coordinate_frame="test_cartesian",
        provenance=(provenance,),
    )
    with pytest.raises(ValueError, match="out-of-range"):
        Domain(
            domain_id="test_surface",
            kind="oriented_triangular_surface",
            coordinate_frame="test_cartesian",
            positions_m=np.zeros((3, 3)),
            faces=np.array([[0, 1, 2]]),
            fields={},
            entity_sets={entity_set.entity_set_id: entity_set},
        )
