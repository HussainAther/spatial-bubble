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
