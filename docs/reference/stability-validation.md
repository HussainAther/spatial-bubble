# Stability-analysis validation snapshot

The first projected second-variation reference uses the level-0 closed-sphere
solution (12 vertices, 20 triangles) with four equality constraints: one volume
constraint and three centroid-gauge constraints.

At centered-difference step `2e-5`, the ten lowest dimensionless eigenvalues are
approximately:

```text
-1.99545e-5, 8.62686e-6, 1.13287e-5,
 2.61663e-1, 2.61703e-1, 2.61706e-1,
 4.41012e-1, 4.41021e-1, 4.41070e-1, 4.41070e-1
```

The first three values are classified as near-null under the declared `2e-5`
tolerance. They are consistent with the three rotational symmetries left after
translation is removed by the centroid gauge. No eigenvalue is below the
negative-mode threshold, so this discrete equilibrium is reported as
**stable semidefinite** within the current engineering approximation.

The non-null spectrum is insensitive to centered-difference steps from `1e-5`
to `4e-5` at the regression tolerances. The unsymmetrized Hessian's maximum
antisymmetric entry is below `3e-9` over that range.

These checks are code and solution verification only. A continuum stability
claim requires mesh refinement and comparison against analytic mode spectra or
an independent formulation. The catenoid fold remains the preferred first
nontrivial validation benchmark.
