# Reproducing the predictive closed-sphere study

From an editable installation containing the declared `dev` or `scipy` extra:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
./scripts/reproduce_closed_sphere.sh
```

To select another supported interpreter without changing the script:

```bash
PYTHON=.venv/bin/python ./scripts/reproduce_closed_sphere.sh
```

Outputs are written beneath `outputs/closed-sphere/`:

- `scientific/manifest.json` and `arrays.npz`: authoritative immutable bundle;
- `reports/validation.json` and `validation.md`: per-case scientific gates;
- `reports/convergence.json` and `convergence.md`: measured refinement study.

The run stores Python/platform identity, SciPy version and deterministic solver
options, all random seeds, physical parameters, multiplier sign convention,
reduced iteration histories, evidence links, and the exact reproduction command.
Generated outputs remain ignored by Git.

## Limitations and explicit exclusions

This milestone covers only one closed, fixed-volume, no-boundary, gravity-free,
constant-surface-tension equilibrium with fixed topology. It performs no
remeshing, self-intersection repair, general collision detection, boundary
contact, gravity, projected-Hessian stability analysis, PETSc/FEniCS solve,
Surface Evolver comparison, drainage, surfactant or Marangoni transport,
evaporation, airflow coupling, rupture, moving-mesh dynamics, XR, or
visualization enhancement. The smooth-sphere comparison is analytic; this is
not experimental validation of a complete soap bubble.
