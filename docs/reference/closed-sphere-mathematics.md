# Closed fixed-volume equilibrium: mathematics and implementation

## Continuous formulation (PV within its assumptions)

For a closed film with constant surface tension `gamma`, interface multiplicity
`m`, no gravity, no boundary, and prescribed enclosed volume `V0`, the model is

```text
minimize_x  E(x) = m gamma A(x)
subject to  V(x) = V0.
```

With the physical Lagrangian

```text
L = E - p (V - V0),
```

stationarity gives `2 m gamma H - p = 0` for the project convention in which an
outward-oriented convex sphere has `H>0`. Thus a sphere has

```text
R = (3 V0 / 4 pi)^(1/3),
A = 4 pi R^2,
H = 1/R,
p = 2 m gamma/R.
```

The backend-neutral optimizer uses `L=E+lambda_V(V-V0)`, so the reported
physical pressure is explicitly `p=-lambda_V`. Pressure is never prescribed.

## Discrete formulation (EA)

The variables are all vertex coordinates of a fixed, outward-oriented triangle
mesh. Area and oriented polyhedral volume are evaluated exactly for that
piecewise-linear surface, up to floating-point roundoff. Their analytic first
derivatives come from the Milestones 1--2 kernel. The optimizer consumes only
the backend-neutral contracts.

Three volume-centroid equalities fix the target centroid. They are a gauge that
removes the translational nullspace; they do not add physical forces. Variables,
volume, centroid, and energy are explicitly nondimensionalized using
`L=V0^(1/3)` and the corresponding capillary scales. SciPy `trust-constr` uses
explicit analytic gradients and Jacobians. BFGS Hessian approximation is
declared in solver provenance; there is no hidden regularization or fallback.

Mean curvature is computed only after optimization with the mixed-area cotan
DDG operator. The Young--Laplace residual is therefore independent of the
objective and volume Jacobian used to solve the problem.

## Initialization and admissibility

The deterministic families are a low-order radial perturbation, a modest
stretched ellipsoid, bounded seeded radial noise, and bounded seeded full-vector
vertex displacement. Each is recentered and uniformly rescaled to `V0` before
solving. Default displacement limits are stored in the study configuration.

Admissibility requires a closed positive orientation, no degenerate triangles,
and no face whose normal points inward relative to the volume centroid. The
validation policy also gates minimum angle and maximum aspect ratio. This
centroid-based inversion test is valid for the star-shaped sphere-recovery
domain; it is not a general self-intersection test.

## Architectural isolation

`openphenomena.equilibrium.closed_sphere` contains no SciPy import. It owns the
physical problem, initial conditions, metrics, acceptance, and evidence.
`openphenomena.equilibrium.reference` is the executable study layer and is the
only new equilibrium module that selects the SciPy adapter. The exact geometry
kernel remains unaware of optimization and equilibrium packages.
