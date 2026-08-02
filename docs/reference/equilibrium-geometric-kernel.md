# Exact discrete equilibrium geometry kernel

Status: implemented and verified for v0.2.0 Milestone 2. No optimizer is part of
this milestone.

## Scientific scope and classification

The kernel evaluates the exact functionals of an oriented, piecewise-linear
triangular mesh, up to floating-point roundoff. It is backend-independent NumPy
code and does not depend on SciPy, PETSc, automatic differentiation, rendering,
or a plugin.

| Component | Classification | Basis |
|---|---|---|
| Triangle area, polyhedral volume, volume centroid | **EA**, verified | Exact for the discrete polyhedral surface; a mesh is an approximation of a smooth physical interface |
| Analytic first derivatives | **EA**, verified | Exact derivatives of the same discrete functionals away from degenerate triangles |
| Surface energy `m gamma A` | **EA**, verified | Physical constant-tension energy evaluated on the discrete surface |
| Nondimensionalization and SI restoration | **EA**, verified | Exact scale transformations up to roundoff |
| Scientific `Field` packaging | **EA**, verified | Unit-, association-, provenance-, and validation-bearing representation |
| Colormaps, mesh smoothing, rendering | **VO**, absent | Not part of this kernel |
| Optimization and equilibrium claims | **SF** for this milestone, absent | Deferred to Milestone 3 and later validation gates |

The **EA** label is deliberate: algebraic exactness for a triangulation is not
experimental validation of the continuum model. No output from this milestone
alone is a solved equilibrium or a physically validated bubble shape.

## Discrete mathematics

For each consistently oriented triangle `(a,b,c)`,

```text
A_f = 1/2 |(b-a) x (c-a)|,
V_f = a . (b x c) / 6,
M_f = V_f (a+b+c) / 4.
```

The total area is `A=sum A_f`, the signed enclosed volume is `V=sum V_f`,
and the volume centroid is `C=(sum M_f)/V`. Volume and centroid require a
closed, consistently oriented two-manifold: every undirected edge must occur
exactly twice with opposing directions. The code rejects open, nonmanifold, and
inconsistently wound inputs instead of assigning them a misleading volume.

Let `n_f=((b-a)x(c-a))/|(b-a)x(c-a)|`. The local area derivatives are

```text
dA_f/da = 1/2 (b-c) x n_f,
dA_f/db = 1/2 (c-a) x n_f,
dA_f/dc = 1/2 (a-b) x n_f.
```

The local signed-volume derivatives are

```text
dV_f/da = (b x c)/6,
dV_f/db = (c x a)/6,
dV_f/dc = (a x b)/6.
```

For `M=sum V_f(a+b+c)/4`, each vertex block is differentiated by the
product rule, and the centroid Jacobian follows from

```text
dC/dx_i = (dM/dx_i - C tensor dV/dx_i) / V.
```

Surface energy and its derivative are

```text
E = m gamma A,
grad E = m gamma grad A,
```

where `m=1` is a single interface and `m=2` is the equal-tension,
negligible-separation soap-film representation established by the frozen
proposal.

## Units and scale transformations

The authoritative evaluation uses SI: positions and area gradients are metres,
areas and volume gradients are square metres, volume is cubic metres, surface
tension is newtons per metre, energy is joules, and its position gradient is
newtons. The centroid Jacobian is dimensionless. All result arrays are copied
and made read-only; `as_fields()` attaches association, SI dimension, fidelity,
validation status, coordinate frame, and generating provenance.

For length scale `L`, the utilities use `L^2` for area, `L^3` for volume,
`m gamma L^2` for energy, `m gamma L` for force, and `m gamma/L` for pressure.
The frozen proposal's `L=V0^(1/3)` is available as the deterministic
volume-controlled scale choice. Every transformation has an explicit inverse
that restores SI values.

## Verification evidence

Automated tests establish:

- hand-derived tetrahedron area, volume, centroid, volume gradient, and
  centroid Jacobian values;
- central finite-difference agreement for every implemented analytic
  derivative: area, volume, centroid, and surface energy;
- translation invariance of area and closed-surface volume, centroid
  equivariance, rotation invariance, and vector-gradient covariance;
- exact dimensional scaling laws for values and derivatives;
- nondimensional/SI round trips;
- rejection of degenerate derivatives and invalid volume topology;
- immutability, units, provenance, association, and deterministic scientific
  serialization.

These are code-verification results. Continuum convergence and physical
validation belong to later v0.2 benchmarks after an optimizer exists.

## Known limitations

The implementation uses IEEE-754 double precision and naive deterministic
summation, not exact rational arithmetic or compensated reductions. Centroid
conditioning deteriorates as signed volume approaches zero. Self-intersection
testing is not part of these two milestones. Derivatives are undefined at a
degenerate triangle and are rejected. There is no Hessian, remeshing,
regularization, boundary-condition enforcement, optimizer, KKT solve, or
Young--Laplace equilibrium claim yet.
