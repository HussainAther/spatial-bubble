# Static-sphere release audit

This audit freezes the scientific interpretation of v0.1.0. Acceptance
thresholds were not changed during release preparation.

| Audit item | Result |
|---|---|
| Curvature and pressure signs | Outward convex `H=+1/R`; `Delta_s x=-2Hn`; `delta_p=p_i-p_o=4 gamma H` for two interfaces |
| Field associations | Leading cardinalities are checked for every vertex and face field; global spectral and scalar fields are explicit |
| Error norms | L1 and L2 use normalized mixed dual-area weights; Linf is the pointwise maximum |
| Observed rates | `log(e_old/e_new)/log(h_old/h_new)` using RMS unique-edge length |
| Symmetric level 1 | Roundoff-exact result retained; level 1->2 rate reported but excluded from asymptotic acceptance |
| Units | SI dimensions and written units are attached to every field; vacuum wavelength is stored in metres |
| Spectral behavior | Vertex angles/thickness broadcast against a global wavelength axis; s/p are power reflectances and unpolarized is `(Rs+Rp)/2` |
| Fidelity | Analytic ideal-sphere references are PV; discretization and curved local optics are EA; display transforms and view exports are VO |
| Serialization | All study/run/frame/domain/field descriptors, arrays, provenance, evidence, uncertainty, classifications, and metadata round-trip |
| Evidence links | Each record names the convergence, scientific interchange, visualization manifest, or report artifacts that support it |
| Determinism | Meshes, fields, JSON keys, field/archive member order, and NPZ member metadata are deterministic; runtime/memory measurements remain observational |
| Plugin isolation | Core imports no phenomenon implementation; discovery uses only the installed `openphenomena.plugins` entry point |
| VTP | Point/cell/global association, triangle connectivity/offsets, counts, units, dimensions, provenance labels, fidelity, and validation status are checked |

The remaining scientific limitations are documented in
`spherical-bubble-limitations.md` and the release notes. In particular, the
VTP compatibility evidence uses the project's independent reader rather than
an external ParaView/VTK installation, and dependency lower bounds are recorded
as resolved versions rather than governed by a cross-platform lock.
