# Closed-sphere measured convergence

The canonical study solves the same perturbed-sphere family at icosphere levels
0, 1, and 2. It records characteristic RMS edge length, solution errors,
residuals, multiplier error, and empirical adjacent-level rates

```text
q = log(e_i/e_(i-1)) / log(h_i/h_(i-1)).
```

The report demonstrates decreasing pressure-multiplier, energy, and sampled
radial errors. KKT and volume residuals are independently gated at every level.
Only three mesh levels are used, so the output explicitly makes no asymptotic
convergence claim. The generated numerical table is
`outputs/closed-sphere/reports/convergence.json` and its human-readable form is
`convergence.md` in the same directory.

The initialization study separately solves all four supported families at level
1 and checks convergence to the same discrete equilibrium within the analytic
pressure and energy gates. This is robustness evidence for the stated bounded
initializations, not a proof of global convergence from arbitrary meshes.

## Canonical measured table

| level | vertices | pressure relative error | energy relative error | sampled radial error (m) | pressure rate | energy rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 0.0645935 | 0.0645935 | 0.00182062 | — | — |
| 1 | 42 | 0.0156230 | 0.0156229 | 0.000498126 | 1.99687 | 1.99687 |
| 2 | 162 | 0.00383670 | 0.00384435 | 0.000122136 | 1.99952 | 1.99668 |

The near-two empirical rates are observations for this mesh sequence. They are
not promoted to an asymptotic-order claim.
