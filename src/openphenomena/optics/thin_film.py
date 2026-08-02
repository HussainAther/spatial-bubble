"""Coherent Fresnel reflection from a plane-parallel thin film.

The kernel evaluates complex field amplitudes, including the infinite series of
internal reflections. It does not convert spectra to RGB and does not model
surface roughness, finite coherence, absorption, fluorescence, or polarization
state beyond independent s/p intensities.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

Polarization = Literal["s", "p", "unpolarized"]
FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


def thin_film_phase_thickness(
    wavelength_m: npt.ArrayLike,
    thickness_m: npt.ArrayLike,
    film_refractive_index: npt.ArrayLike,
    *,
    incident_refractive_index: npt.ArrayLike = 1.0,
    incidence_angle_rad: npt.ArrayLike = 0.0,
) -> FloatArray:
    """Return one-way phase thickness ``2*pi*n*h*cos(theta_f)/lambda``.

    The media are lossless and isotropic, and vacuum wavelength is used. Inputs
    follow NumPy broadcasting.
    """

    wavelength = _positive_array(wavelength_m, "wavelength_m")
    thickness = _nonnegative_array(thickness_m, "thickness_m")
    n0 = _positive_array(incident_refractive_index, "incident_refractive_index")
    n1 = _positive_array(film_refractive_index, "film_refractive_index")
    theta0 = _incidence_angle_array(incidence_angle_rad)
    cos1 = np.sqrt(1.0 - (n0 * np.sin(theta0) / n1) ** 2)
    return np.asarray(
        2.0 * np.pi * n1 * thickness * cos1 / wavelength,
        dtype=np.float64,
    )


def thin_film_reflectance(
    wavelength_m: npt.ArrayLike,
    thickness_m: npt.ArrayLike,
    film_refractive_index: npt.ArrayLike,
    *,
    incident_refractive_index: npt.ArrayLike = 1.0,
    exit_refractive_index: npt.ArrayLike = 1.0,
    incidence_angle_rad: npt.ArrayLike = 0.0,
    polarization: Polarization = "unpolarized",
) -> FloatArray:
    """Return coherent spectral power reflectance of a single dielectric film.

    Inputs follow NumPy broadcasting. Refractive indices are currently real and
    positive, so this version models lossless, isotropic media. Angles are
    measured in the incident medium from the surface normal. Wavelength is the
    vacuum wavelength.

    The round-trip phase is represented by ``exp(2j*delta)``, where

        delta = 2*pi*n_film*thickness*cos(theta_film)/wavelength.

    Complex amplitude reflection is

        r = (r01 + r12 exp(2j*delta)) /
            (1 + r01 r12 exp(2j*delta)).

    ``unpolarized`` is the equal incoherent average of s and p power.
    """

    wavelength = _positive_array(wavelength_m, "wavelength_m")
    thickness = _nonnegative_array(thickness_m, "thickness_m")
    n0 = _positive_array(incident_refractive_index, "incident_refractive_index")
    n1 = _positive_array(film_refractive_index, "film_refractive_index")
    n2 = _positive_array(exit_refractive_index, "exit_refractive_index")
    theta0 = _incidence_angle_array(incidence_angle_rad)
    if polarization not in {"s", "p", "unpolarized"}:
        raise ValueError(f"unsupported polarization: {polarization!r}")

    # Complex square roots make the formulation continuous if an internal angle
    # becomes evanescent. With the intended air--water--air use, both are real.
    sin0 = np.sin(theta0)
    cos0 = np.cos(theta0).astype(np.complex128)
    cos1 = np.sqrt(1.0 - (n0 * sin0 / n1).astype(np.complex128) ** 2)
    cos2 = np.sqrt(1.0 - (n0 * sin0 / n2).astype(np.complex128) ** 2)
    phase = np.exp(2j * (2.0 * np.pi * n1 * thickness * cos1 / wavelength))

    def amplitude(kind: Literal["s", "p"]) -> ComplexArray:
        if kind == "s":
            r01 = (n0 * cos0 - n1 * cos1) / (n0 * cos0 + n1 * cos1)
            r12 = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
        else:
            r01 = (n1 * cos0 - n0 * cos1) / (n1 * cos0 + n0 * cos1)
            r12 = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)
        return np.asarray((r01 + r12 * phase) / (1.0 + r01 * r12 * phase))

    if polarization == "unpolarized":
        power = 0.5 * (np.abs(amplitude("s")) ** 2 + np.abs(amplitude("p")) ** 2)
    else:
        power = np.abs(amplitude(polarization)) ** 2
    return np.asarray(np.real_if_close(power), dtype=np.float64)


def _positive_array(value: npt.ArrayLike, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain only finite positive values")
    return result


def _nonnegative_array(value: npt.ArrayLike, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must contain only finite nonnegative values")
    return result


def _incidence_angle_array(value: npt.ArrayLike) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(result)) or np.any(np.abs(result) >= np.pi / 2.0):
        raise ValueError("incidence_angle_rad must be finite and in (-pi/2, pi/2)")
    return result
