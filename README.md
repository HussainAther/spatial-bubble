# Open Phenomena (working title)

An open-source scientific visualization and spatial-computing framework whose
first reference problem is the physics of soap films and bubbles.

> Maximize physical realism first. Allow aesthetics to emerge naturally from
> accurate simulation.

The repository begins with independently testable kernels:

- Young--Laplace relations for equilibrium bubbles;
- coherent, wavelength-dependent Fresnel optics for an air--film--air slab.
- exact piecewise-linear surface area, oriented volume, centroid, surface
  energy, and their analytic first derivatives.
- backend-neutral equality-constrained optimization contracts and an isolated,
  optional SciPy `trust-constr` adapter verified on synthetic problems.
- a first predictive closed, fixed-volume capillary equilibrium solver with
  multiplier-derived pressure and independent sphere-recovery evidence.

These are deliberately not a complete bubble simulator. They establish the
project's conventions for units, validation, scientific state, and fidelity
claims before more difficult coupled solvers are introduced.

## Principles

1. Every result carries units, parameters, provenance, and a fidelity label.
2. Solvers produce scientific fields; renderers and XR clients consume them.
3. Reference solutions and convergence studies precede visual demonstrations.
4. Physically validated methods, engineering approximations, visualization-only
   methods, and speculative work are labeled separately.
5. The framework core remains independent of Blender, a game engine, or a
   particular PDE/GPU backend.

## Quick start

Requires Python 3.11, 3.12, or 3.13. The reproduction script normalizes the
editable-install path file for patched runtimes that ignore underscore-prefixed
`.pth` files. Python 3.14 support remains intentionally deferred.

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

Compute a normal-incidence spectral reflectance curve:

```python
import numpy as np
from openphenomena.optics import thin_film_reflectance

wavelength_m = np.linspace(380e-9, 780e-9, 81)
reflectance = thin_film_reflectance(
    wavelength_m=wavelength_m,
    thickness_m=450e-9,
    film_refractive_index=1.333,
)
```

Run the complete evidence-bearing static spherical-bubble study from a clean
checkout with one command:

```bash
./scripts/reproduce_static_sphere.sh
```

Run the predictive equilibrium validation study after installing the SciPy
extra (included in `dev`):

```bash
./scripts/reproduce_closed_sphere.sh
```

See the [reference-study tutorial](docs/tutorials/static-spherical-bubble.md)
for its outputs, classifications, and current limitations.

The long-term design is documented in:

- [system architecture](docs/ARCHITECTURE.md);
- [scientific data model](docs/DATA_MODEL.md);
- [verification, validation, and numerical policy](docs/VALIDATION.md);
- [phenomenon plugin architecture](docs/PLUGINS.md);
- [staged research roadmap](docs/ROADMAP.md);
- [v0.2.0 equilibrium-surface proposal](docs/proposals/v0.2.0-equilibrium-surfaces.md).

## Status

Version 0.1.0 remains the frozen validated static-sphere reference baseline.
Development toward v0.2.0 has implemented only the additive boundary schema and
backend-independent geometric kernel, followed by numerical solver
infrastructure and the closed fixed-volume sphere-recovery milestone. This is a
verified predictive discrete equilibrium solver, not an experimentally
validated complete soap-bubble simulation. Boundaries, gravity, mesh evolution,
fluid transport, rupture, stability analysis, and coupled physics remain
deferred.

## Constrained stability analysis (development)

The equilibrium layer now includes a backend-neutral projected second-variation
analysis. It finite-differences the analytic Lagrangian gradient, constructs the
constraint tangent space by SVD, and reports the lowest constrained eigenvalues,
negative-mode count, near-null modes, and ParaView-compatible displacement
modes.

```bash
./scripts/reproduce_stability.sh
```

This capability is currently **EA, verified**. It is not yet a continuum or
experimental stability validation, and catenoid-fold validation remains a
future benchmark. See [the stability method](docs/reference/constrained-stability.md).
