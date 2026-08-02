# Contributing

Open Phenomena accepts contributions that preserve scientific traceability.

## Before implementing a model

Open an issue or design note that states:

1. governing equations and sign conventions;
2. constitutive laws and physical parameters, with units and sources;
3. assumptions and regime of validity;
4. numerical discretization and stability/convergence expectations;
5. validation cases and quantitative acceptance criteria;
6. intended fidelity label: physically validated, engineering approximation,
   visualization-only, or speculative future work.

An attractive image is useful evidence for presentation, but is not validation.

## Development checks

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/mypy src
```

Public APIs require type hints and documentation. Numerical changes require a
regression test, and accelerated kernels require comparison with a small
reference implementation. Store SI units in authoritative state unless a file
format requires otherwise; conversions must be explicit at boundaries.

## Commits

Keep physics/model changes separate from display-only changes when practical.
Record citations and changed assumptions in the same commit as their code.
