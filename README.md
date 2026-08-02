# Open Phenomena (working title)

An open-source scientific visualization and spatial-computing framework whose
first reference problem is the physics of soap films and bubbles.

> Maximize physical realism first. Allow aesthetics to emerge naturally from
> accurate simulation.

The repository begins with two small, independently testable kernels:

- Young--Laplace relations for equilibrium bubbles;
- coherent, wavelength-dependent Fresnel optics for an air--film--air slab.

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

Requires Python 3.11, 3.12, or 3.13. Python 3.14 editable-install support is
intentionally deferred until the build backend emits a non-hidden path file.

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

Version 0.1.0 establishes the validated static-sphere reference baseline. No
current output should be interpreted as an experimentally validated complete
soap-bubble simulation.
