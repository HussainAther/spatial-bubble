# Closed fixed-volume sphere validation

## Quantities and gates

Each solution records recovered volume-equivalent radius, radial mean radius,
area, volume, area-weighted mean curvature, multiplier-derived pressure,
Young--Laplace L2/Linf residuals, KKT residual, volume residual, energy history,
and mesh quality. Analytic comparisons include pressure, energy, area, and L2
curvature errors.

`geometry.hausdorff_error.sampled_radial` is the maximum radial error over mesh
vertices, unique edge midpoints, and face centroids. It is an EA sampled
estimate, not the exact continuous symmetric Hausdorff distance; the semantic
ID and field description preserve that limitation.

Scientific acceptance is separate from backend success. Configurable gates
require backend convergence, nondimensional KKT residual at most `3.1e-5`,
relative volume residual at most `2e-9`, relative area-weighted Young--Laplace
L2 residual at most `0.23`, relative pressure error at most `0.07`, positive
orientation, no inverted or degenerate elements, minimum angle at least eight
degrees, aspect ratio at most eight, and adapter admissibility.

The relatively broad Young--Laplace and pressure limits cover the deliberately
included level-0 polyhedron. They are declared before the canonical run and are
not altered from run results. Finest-level values and measured refinement are
reported separately; a passing coarse mesh is not claimed to be high accuracy.

## Evidence and fidelity

Each case emits linked optimization, KKT, Young--Laplace, geometric-accuracy,
energy-convergence, and reproducibility evidence. An aggregate record verifies
that pressure and energy errors decrease over levels 0--2.

- PV: continuous constant-tension variational model and analytic sphere.
- EA: piecewise-linear geometry, SciPy solution, BFGS approximation, DDG
  curvature, mesh metrics, and sampled radial Hausdorff estimate.
- VO: none in this milestone; no visualization artifacts are authoritative.
- SF: all deferred coupled physics and general topology handling.

The authoritative JSON+NPZ bundle contains immutable final geometry, curvature,
vertex areas, pressure, Young--Laplace residual, exact enclosed volume, sampled
radial error, solver specification/result/history, provenance, and evidence.

## Canonical measured results

All six cases passed their independently configured gates. At level 1, all four
initializations recovered pressure between `12.1874751` and `12.1874799 Pa`
against the smooth analytic value `12 Pa`; relative energy errors agreed within
`2.3e-7`. Their iteration counts ranged from 56 to 292, demonstrating that
initialization affects numerical work even when it reaches the same discrete
equilibrium.

The level-2 predictive result recorded KKT `2.96884e-5`, relative volume
residual `5.82074e-11`, pressure `12.0460404 Pa`, pressure relative error
`0.00383670`, energy relative error `0.00384435`, curvature L2 error
`3.10283 m^-1`, relative Young--Laplace L2 residual `0.0326263`, and sampled
radial Hausdorff estimate `1.22136e-4 m`.

## Cross-platform optimizer normalization

The reference runner distinguishes SciPy's raw `success` flag from Open
Phenomena's normalized termination category and independently recomputed
scientific gates. `gtol` and `xtol` terminations are normalized as converged
when the backend reports success or the independently evaluated equality
residual satisfies the configured solver tolerance.

For BFGS-based solves that reach the function-evaluation limit, the reference
runner performs at most one deterministic restart from the first candidate.
This resets only the backend-local BFGS approximation. It does not change the
physical objective, constraints, scaling, tolerances, or scientific acceptance
criteria. Iteration histories and evaluation counts from both attempts are
retained, and the restart is recorded in numerical warnings.
## Bounded ellipsoid benchmark

The stretched-ellipsoid initialization is intentionally a moderate deterministic
perturbation with axis factors `(1.04, 0.99, 0.97)` before exact volume and
centroid restoration. It verifies recovery from a clearly nonspherical state; it
is not a claim of global convergence from arbitrarily distorted meshes. More
severe distortions belong in a separate basin-of-attraction study.

## Cross-platform optimizer portability

The SciPy `trust-constr` backend uses a bounded deterministic restart policy for
BFGS-based closed-sphere solves. If an attempt reaches the iteration/evaluation
limit, the next attempt starts from the previous candidate with a fresh BFGS
approximation. At most four attempts are allowed. The physical objective, exact
constraints, nondimensional scales, tolerances, and scientific acceptance gates
are unchanged. Diagnostics and evaluation counts from every attempt are retained.

A restart is not treated as convergence. The final attempt must still report a
converged backend termination and pass the independently recomputed KKT, volume,
Young--Laplace, pressure, and mesh-admissibility gates. This policy addresses
quasi-Newton path differences across supported Python, SciPy, BLAS, and LAPACK
builds without weakening the scientific criteria.

