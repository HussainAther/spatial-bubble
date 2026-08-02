from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openphenomena import (
    Domain,
    Fidelity,
    Field,
    FieldAssociation,
    FieldDescriptor,
    Provenance,
    Uncertainty,
    ValidationStatus,
)

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
