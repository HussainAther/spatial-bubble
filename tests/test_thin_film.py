import numpy as np
import pytest

from openphenomena.optics import thin_film_phase_thickness, thin_film_reflectance


def test_zero_thickness_air_film_air_has_zero_reflectance() -> None:
    result = thin_film_reflectance(
        wavelength_m=np.linspace(400e-9, 700e-9, 11),
        thickness_m=0.0,
        film_refractive_index=1.333,
    )
    np.testing.assert_allclose(result, 0.0, atol=1e-30)


def test_half_wave_optical_thickness_is_destructive_at_normal_incidence() -> None:
    wavelength = 550e-9
    thickness = wavelength / (2.0 * 1.333)
    result = thin_film_reflectance(wavelength, thickness, 1.333)
    assert float(result) == pytest.approx(0.0, abs=1e-30)


def test_quarter_wave_optical_thickness_matches_closed_form() -> None:
    wavelength = 550e-9
    n = 1.333
    thickness = wavelength / (4.0 * n)
    r_interface = (1.0 - n) / (1.0 + n)
    expected = 4.0 * r_interface**2 / (1.0 + r_interface**2) ** 2
    result = thin_film_reflectance(wavelength, thickness, n)
    assert float(result) == pytest.approx(expected)


def test_s_and_p_agree_at_normal_incidence() -> None:
    kwargs = dict(wavelength_m=510e-9, thickness_m=220e-9, film_refractive_index=1.34)
    s = thin_film_reflectance(**kwargs, polarization="s")
    p = thin_film_reflectance(**kwargs, polarization="p")
    np.testing.assert_allclose(s, p, rtol=1e-14)


def test_phase_thickness_matches_normal_incidence_definition() -> None:
    wavelength = 500e-9
    thickness = 250e-9
    refractive_index = 1.4
    phase = thin_film_phase_thickness(wavelength, thickness, refractive_index)
    assert float(phase) == pytest.approx(
        2.0 * np.pi * refractive_index * thickness / wavelength
    )
