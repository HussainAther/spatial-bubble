# Static spherical-bubble mathematics

## Sign conventions

The mesh is oriented with outward normals from the bubble interior to the
exterior. A convex sphere has `k1=k2=H=+1/R` and `K=+1/R²`. Pressure jump always
means `p_internal-p_external`. Each interface contributes `2 gamma H`, so two
equal-tension interfaces give `delta_p=4 gamma H=4 gamma/R`.

For canonical `R=0.01 m` and `gamma=0.03 N/m`, `delta_p=12 Pa`.

## Mesh and discrete operators

Every icosphere refinement replaces one triangle by four and radially projects
new vertices to radius `R`. Faces are oriented outward.

For mixed Voronoi area `A_i`, the cotangent Laplacian is

```text
Delta_s x_i = (1/(2 A_i)) sum_j
              (cot(alpha_ij)+cot(beta_ij)) (x_j-x_i).
```

Because `Delta_s x=-2 H n` on the outward sphere, the implemented signed mean
curvature is `H_i=-0.5 dot(Delta_s x_i,n_i)`. Gaussian curvature is the angle
defect `(2 pi-sum_f theta_if)/A_i`.

These formulas follow Meyer, Desbrun, Schröder, and Barr,
“Discrete Differential-Geometry Operators for Triangulated 2-Manifolds,”
[DOI 10.1007/978-3-662-05105-4_2](https://link.springer.com/chapter/10.1007/978-3-662-05105-4_2).

The discrete estimates are **EA**; analytical sphere values are **PV** within
the ideal model.

## Error norms and rates

Mixed-area-weighted `L1`, `L2`, and vertexwise `Linf` norms are reported. Between
RMS edge lengths `h_a>h_b`, the observed rate is
`log(e_a/e_b)/log(h_a/h_b)`. Level 1 is exact to roundoff by symmetry, so it is
reported but excluded from the asymptotic rate claim. Levels 2–4 form the
measured window.

## Thin-film optics

The one-way phase is `delta=2 pi n_f h cos(theta_f)/lambda`. The coherent slab
amplitude and power are

```text
r = (r01+r12 exp(2 i delta))/(1+r01 r12 exp(2 i delta)),
R = |r|^2.
```

The plane-parallel lossless kernel is **PV** within its assumptions. Applying it
independently in curved-surface tangent planes is **EA**.
