"""Nondimensionalization and explicit SI restoration for surface mechanics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openphenomena.data import FloatArray


@dataclass(frozen=True, slots=True)
class EquilibriumScales:
    """Reference scales for capillary equilibrium functionals.

    Length is scaled by ``L``; area by ``L^2``; volume by ``L^3``;
    energy by ``m gamma L^2``; and force by ``m gamma L``.
    """

    length_m: float
    surface_tension_n_per_m: float
    interface_multiplicity: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.length_m) or self.length_m <= 0.0:
            raise ValueError("length scale must be finite and positive")
        if (
            not np.isfinite(self.surface_tension_n_per_m)
            or self.surface_tension_n_per_m <= 0.0
        ):
            raise ValueError("surface-tension scale must be finite and positive")
        if (
            not np.isfinite(self.interface_multiplicity)
            or self.interface_multiplicity <= 0.0
        ):
            raise ValueError("interface multiplicity must be finite and positive")

    @classmethod
    def from_target_volume(
        cls,
        target_volume_m3: float,
        surface_tension_n_per_m: float,
        interface_multiplicity: float = 1.0,
    ) -> EquilibriumScales:
        """Use ``target_volume_m3 ** (1/3)`` as the reference length."""

        if not np.isfinite(target_volume_m3) or target_volume_m3 <= 0.0:
            raise ValueError("target volume must be finite and positive")
        length = target_volume_m3 ** (1.0 / 3.0)
        return cls(length, surface_tension_n_per_m, interface_multiplicity)

    @property
    def energy_j(self) -> float:
        return (
            self.interface_multiplicity
            * self.surface_tension_n_per_m
            * self.length_m**2
        )

    @property
    def force_n(self) -> float:
        return (
            self.interface_multiplicity * self.surface_tension_n_per_m * self.length_m
        )

    @property
    def pressure_pa(self) -> float:
        return (
            self.interface_multiplicity * self.surface_tension_n_per_m / self.length_m
        )

    def positions_to_dimensionless(self, values_m: FloatArray) -> FloatArray:
        return _scaled(values_m, 1.0 / self.length_m)

    def positions_to_si(self, values: FloatArray) -> FloatArray:
        return _scaled(values, self.length_m)

    def centroid_to_dimensionless(self, values_m: FloatArray) -> FloatArray:
        return _scaled(values_m, 1.0 / self.length_m)

    def centroid_to_si(self, values: FloatArray) -> FloatArray:
        return _scaled(values, self.length_m)

    def face_areas_to_dimensionless(self, values_m2: FloatArray) -> FloatArray:
        return _scaled(values_m2, 1.0 / self.length_m**2)

    def face_areas_to_si(self, values: FloatArray) -> FloatArray:
        return _scaled(values, self.length_m**2)

    def area_to_dimensionless(self, value_m2: float) -> float:
        return _scaled_scalar(value_m2, 1.0 / self.length_m**2)

    def area_to_si(self, value: float) -> float:
        return _scaled_scalar(value, self.length_m**2)

    def volume_to_dimensionless(self, value_m3: float) -> float:
        return _scaled_scalar(value_m3, 1.0 / self.length_m**3)

    def volume_to_si(self, value: float) -> float:
        return _scaled_scalar(value, self.length_m**3)

    def energy_to_dimensionless(self, value_j: float) -> float:
        return _scaled_scalar(value_j, 1.0 / self.energy_j)

    def energy_to_si(self, value: float) -> float:
        return _scaled_scalar(value, self.energy_j)

    def pressure_to_dimensionless(self, value_pa: float) -> float:
        return _scaled_scalar(value_pa, 1.0 / self.pressure_pa)

    def pressure_to_si(self, value: float) -> float:
        return _scaled_scalar(value, self.pressure_pa)

    def curvature_to_dimensionless(self, value_per_m: float) -> float:
        return _scaled_scalar(value_per_m, self.length_m)

    def curvature_to_si(self, value: float) -> float:
        return _scaled_scalar(value, 1.0 / self.length_m)

    def area_gradient_to_dimensionless(self, values_m: FloatArray) -> FloatArray:
        return _scaled(values_m, 1.0 / self.length_m)

    def area_gradient_to_si(self, values: FloatArray) -> FloatArray:
        return _scaled(values, self.length_m)

    def volume_gradient_to_dimensionless(self, values_m2: FloatArray) -> FloatArray:
        return _scaled(values_m2, 1.0 / self.length_m**2)

    def volume_gradient_to_si(self, values: FloatArray) -> FloatArray:
        return _scaled(values, self.length_m**2)

    def energy_gradient_to_dimensionless(self, values_n: FloatArray) -> FloatArray:
        return _scaled(values_n, 1.0 / self.force_n)

    def energy_gradient_to_si(self, values: FloatArray) -> FloatArray:
        return _scaled(values, self.force_n)

    def centroid_jacobian_to_dimensionless(self, values: FloatArray) -> FloatArray:
        return _scaled(values, 1.0)

    def centroid_jacobian_to_si(self, values: FloatArray) -> FloatArray:
        return _scaled(values, 1.0)


def _scaled(values: FloatArray, factor: float) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True) * factor
    if not np.all(np.isfinite(result)):
        raise ValueError("scaled values must be finite")
    result.flags.writeable = False
    return result


def _scaled_scalar(value: float, factor: float) -> float:
    result = value * factor
    if not np.isfinite(result):
        raise ValueError("scaled value must be finite")
    return result
