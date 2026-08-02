"""Young--Laplace reference relations.

These analytical functions are reference cases for future surface-equilibrium
solvers. SI units are used throughout.
"""

from __future__ import annotations


def spherical_interface_pressure_jump(
    radius_m: float,
    surface_tension_n_per_m: float,
) -> float:
    """Return pressure jump across one spherical interface, in pascals.

    The Young--Laplace equation is

        delta_p = gamma (1/R_1 + 1/R_2) = 2 gamma / R

    for a sphere of radius ``R``. The result is the pressure on the side toward
    which the chosen interface normal points minus the pressure on the other
    side, under the usual positive-curvature magnitude convention.
    """

    _validate_positive(radius_m, "radius_m")
    _validate_nonnegative(surface_tension_n_per_m, "surface_tension_n_per_m")
    return 2.0 * surface_tension_n_per_m / radius_m


def spherical_bubble_pressure_jump(
    radius_m: float,
    surface_tension_n_per_m: float,
) -> float:
    """Return inside-minus-outside pressure for a thin spherical soap film.

    A soap bubble has two approximately spherical air--liquid interfaces. If
    film thickness is negligible relative to radius and both interfaces have
    the same surface tension, their leading-order contributions add:

        delta_p = 4 gamma / R.

    This is not the one-interface result for a gas bubble immersed in liquid.
    """

    return 2.0 * spherical_interface_pressure_jump(
        radius_m=radius_m,
        surface_tension_n_per_m=surface_tension_n_per_m,
    )


def _validate_positive(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive; got {value!r}")


def _validate_nonnegative(value: float, name: str) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative; got {value!r}")
