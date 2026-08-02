# Static spherical-bubble field dictionary

Every field records shape, dtype, association, coordinate frame, generating
model and implementation, fidelity, provenance, validation status, and an
explicit uncertainty statement.

| Semantic ID | Association | Unit | Class | Meaning |
|---|---|---|---|---|
| `geometry.position` | vertex | m | EA | projected sphere positions |
| `geometry.area.face` | face | m² | EA | planar triangle area |
| `geometry.normal.face` | face | 1 | EA | outward face normal |
| `geometry.normal.vertex` | vertex | 1 | EA | area-weighted vertex normal |
| `geometry.area.vertex_mixed` | vertex | m² | EA | mixed Voronoi area |
| `geometry.mean_curvature.discrete` | vertex | m⁻¹ | EA | cotangent mean curvature |
| `geometry.gaussian_curvature.discrete` | vertex | m⁻² | EA | angle-defect curvature |
| `geometry.principal_curvature.analytic` | vertex | m⁻¹ | PV | two components `k1=k2=1/R` |
| `geometry.mean_curvature.analytic` | vertex | m⁻¹ | PV | `1/R` |
| `geometry.gaussian_curvature.analytic` | vertex | m⁻² | PV | `1/R²` |
| `geometry.*.absolute_error` | vertex | corresponding | EA | absolute error |
| `geometry.*.relative_error` | vertex | 1 | EA | relative error |
| `mechanics.pressure_jump.discrete` | vertex | Pa | EA | `4 gamma H_discrete` |
| `mechanics.pressure_jump.analytic` | vertex | Pa | PV | `4 gamma/R` |
| `mechanics.pressure_jump.error` | vertex | Pa | EA | discrete minus analytic |
| `film.thickness` | vertex | m | EA | prescribed total thickness |
| `mechanics.surface_tension` | global | N m⁻¹ | EA | prescribed tension |
| `mechanics.pressure.internal/external` | global | Pa | PV | absolute pressures |
| `optics.refractive_index.incident/film/exit` | global | 1 | EA | real air–film–air indices |
| `optics.wavelength` | global spectral | m | PV input | vacuum wavelengths |
| `optics.incidence_angle` | vertex | rad | EA | local beam angle |
| `optics.phase_thickness` | vertex × wavelength | rad | EA | one-way phase |
| `optics.reflectance.s/p/unpolarized` | vertex × wavelength | 1 | EA | spectral reflectance |
| `radiometry.incident_spectral_radiance` | wavelength | W m⁻² sr⁻¹ m⁻¹ | EA | prescribed spectrum |
| `radiometry.spectral_reflected_radiance` | vertex × wavelength | W m⁻² sr⁻¹ m⁻¹ | EA | weighted spectrum |
| `color.cie_xyz` | vertex | 1 | VO | approximate colorimetry |
| `color.linear_srgb` | vertex | 1 | VO | linear display RGB |
| `color.tonemapped_srgb` | vertex | 1 | VO | tone-mapped RGB |

The Wyman–Sloan–Shirley analytic color-matching approximation is used only for
VO fields: [reference](https://research.nvidia.com/labs/rtr/publication/wyman2013simple/).
