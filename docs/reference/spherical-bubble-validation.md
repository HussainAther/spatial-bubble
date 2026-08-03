# Static spherical-bubble validation plan

The generated `reports/validation.md` is the result for one run. This page
defines evidence semantics.

| Evidence | Quantity | Acceptance |
|---|---|---|
| `sphere_geometry` | relative polygonal area error | ≤ 1% |
| `curvature_accuracy` | relative weighted L2 mean-curvature error | ≤ 2% |
| `young_laplace_pressure` | relative L2 pressure-jump error | ≤ 2% |
| `optics_bounds` | violation of `0≤Rs,Rp≤1` | ≤ `1e-12` |
| `polarization_consistency` | `max|Ru-(Rs+Rp)/2|` | ≤ `1e-14` |
| `mesh_convergence` | negative-rate deficit over levels 2–4 | zero |
| `export_roundtrip` | VTP absolute error | ≤ `1e-12` |

Passing curvature and pressure evidence numerically verifies an **EA**
discretization against a **PV** analytic reference. It is not experimental
validation of real soap chemistry or curved-bubble radiance.

## Cross-platform floating-point regression policy

Canonical quantities that are mathematically zero but represented at roughly
machine precision are compared with an absolute roundoff tolerance. Meaningful
nonzero quantities retain strict relative and absolute regression tolerances.
The artificial refinement rate that transitions from a roundoff-level exact
icosphere result to a nonzero discretization error is required only to be
finite and negative; it is not treated as a portable scientific quantity.
