# Soap-film physics baseline

## Geometry and pressure

For an interface with principal curvatures `k1` and `k2`, the Young--Laplace
balance is

```text
delta p = gamma (k1 + k2) = 2 gamma H,
```

subject to an explicitly documented normal/curvature sign convention. For a
sphere, one interface gives `2 gamma / R`. A thin soap bubble has two interfaces,
so equal tensions and negligible thickness give `4 gamma / R` to leading order.

This analytical sphere is the first equilibrium-solver regression test. The
next tests are constant-mean-curvature surfaces under constrained volume and
gravity-free boundary conditions, followed by capillary surfaces with gravity.

## Thin-film optics

At each wavelength and angle, Snell's law supplies the propagation angle inside
the film. Fresnel coefficients supply complex interface amplitudes. Summing all
coherent internal reflections yields the single-layer amplitude used by
`optics.thin_film_reflectance`; its squared magnitude is power reflectance.

The current kernel is **validated physics only for its stated idealized model**:
plane-parallel, smooth, isotropic, lossless media under coherent illumination.
Applying it pointwise to a curved film is an engineering approximation whose
validity requires curvature radii and field variation scales much larger than
the wavelength and film thickness.

Required extensions are measured wavelength-dependent complex refractive index,
spectral illuminants, polarization/Jones or Mueller transport, roughness, finite
coherence, and integration into a spectral path tracer. RGB shader ramps are a
**visualization-only** method and cannot be used as scientific optical output.

## Drainage and surfactant transport

A soap film is not merely a colored zero-thickness surface. Its central state
variables will include film thickness `h`, surface surfactant concentration
`Gamma` on both interfaces (or a documented symmetry reduction), surface/bulk
velocity, pressure, and temperature/humidity-related variables where needed.

The first hydrodynamic model should be a lubrication-theory reduction derived
from incompressible Navier--Stokes plus interfacial stress and surfactant
transport. It is an engineering approximation valid when thickness is much
smaller than tangential length scales and inertia/geometry meet the declared
scaling assumptions. It must include:

- mass conservation for liquid volume;
- tangential Marangoni stress from `grad_s gamma(Gamma, T)`;
- surfactant advection-diffusion and, later, adsorption/desorption;
- gravity and capillary pressure;
- Plateau-border boundary fluxes;
- evaporation as a measured constitutive sink;
- disjoining pressure before attempting nanometric rupture.

No rupture threshold will be called a physical rupture model unless it is tied
to a resolved instability or a calibrated stochastic/nucleation law.

## Scale separation

Relevant scales span nanometric black films, micrometric colored films,
centimetric bubble radii, optical femtoseconds, capillary milliseconds, and
drainage seconds to minutes. A single brute-force discretization is therefore
neither practical nor intrinsically more rigorous. Reduced models, adaptivity,
multirate coupling, and uncertainty reporting are core scientific requirements.
