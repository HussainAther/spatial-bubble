import pytest

from openphenomena.physics import (
    spherical_bubble_pressure_jump,
    spherical_interface_pressure_jump,
)


def test_spherical_interface_reference_value() -> None:
    assert spherical_interface_pressure_jump(0.01, 0.072) == pytest.approx(14.4)


def test_two_interfaces_double_pressure_jump() -> None:
    one = spherical_interface_pressure_jump(0.01, 0.03)
    assert spherical_bubble_pressure_jump(0.01, 0.03) == pytest.approx(2.0 * one)


@pytest.mark.parametrize("radius", [0.0, -1.0])
def test_radius_must_be_positive(radius: float) -> None:
    with pytest.raises(ValueError):
        spherical_bubble_pressure_jump(radius, 0.03)
