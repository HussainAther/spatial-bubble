# Constrained second-variation stability

Open Phenomena now computes a projected second variation for a solved equality-
constrained equilibrium.  For dimensionless variables `x`, objective `f`,
constraints `c`, and multiplier vector `lambda`, the analyzed operator is

\[
\nabla^2_{xx}\mathcal L(x,\lambda),\qquad
\mathcal L=f+\lambda^T c.
\]

The equality-constraint Jacobian `J` is decomposed by SVD.  An orthonormal basis
`Z` for `null(J)` defines the feasible tangent space, and the constrained
operator is

\[
H_T=Z^T\nabla^2\mathcal L Z.
\]

Its ordered eigenvalues and eigenvectors identify negative, near-null, and
positive directions.  Physical vertex displacement modes are exported to VTP
for ParaView animation.

## Numerical method and fidelity

The current Hessian is obtained by centered finite differences of the existing
analytic dimensionless Lagrangian gradient.  The matrix is symmetrized before
projection and diagonalization with `numpy.linalg.eigh`.

This is classified **EA, verified**, not a continuum stability proof.  Results
must be checked against finite-difference step, mesh refinement, constraint-rank
tolerance, and independent analytic or published stability references.  Near-
zero modes may include physical symmetries and mesh-parameterization modes.

## Reproduction

```bash
./scripts/reproduce_stability.sh
```

Outputs include `stability_report.json`, `stability_spectrum.npz`, and
`sphere_stability_modes.vtp`.
