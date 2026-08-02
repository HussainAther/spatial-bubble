# Verification, validation, and numerical policy

## Terminology and evidence ladder

- **Code verification:** are the equations/discrete algorithms implemented
  correctly? Use exact solutions, manufactured solutions, properties, and
  independent implementations.
- **Solution verification:** is numerical error controlled for this run? Use
  mesh/time refinement, iterative convergence, conservation, and estimator data.
- **Model validation:** does the mathematical model predict specified physical
  quantities in a stated experimental regime, considering uncertainty?
- **Calibration:** estimates parameters from data. Calibrated is not synonymous
  with validated; validation must use held-out conditions where possible.
- **Uncertainty quantification:** tracks parameter, numerical, model-form, and
  experimental uncertainty separately.

The project follows the spirit of [ASME V&V 20](https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-fluid-dynamics-and-heat-transfer/2009):
credibility belongs to specified quantities of interest and validation points,
not to a universal “accurate simulator” label.

## Required validation record

Every **PV** claim must identify the model/version, regime and nondimensional
groups, quantity of interest, reference data and its uncertainty, mesh/time
study, solver tolerances, comparison metric, acceptance threshold chosen before
the run, platform/environment, and known extrapolation limits.

CI has three levels:

1. **Fast regression:** units, properties, small exact cases, schema round trips.
2. **Numerical verification:** convergence rates, manufactured solutions,
   conservation, independent-backend comparisons.
3. **Validation corpus:** experimental comparisons and expensive benchmarks,
   run on scheduled/release infrastructure with immutable reference datasets.

Passing a regression image is never physics validation.

## Module-by-module strategy

| Module | Code/solution verification | Physical validation target | Initial status |
|---|---|---|---|
| Units/data/provenance | dimensional property tests; schema migration and lossless round trips | not a physics claim | infrastructure |
| Mesh/topology | orientation/manifold checks; Euler characteristic; refinement invariants | metrology geometry where imported | **PV-capable** |
| Surface geometry | sphere/cylinder/torus curvature; Gauss–Bonnet; observed convergence order | measured reference surfaces if needed | **EA** until convergence demonstrated |
| Young–Laplace equilibrium | sphere `Δp=2γ/R` per interface; constrained-energy residual; volume and force balance; mesh refinement | sessile/pendant bubble shapes and measured pressure with uncertainty | ideal sphere **PV**; general solver pending |
| Gravity/capillarity | hydrostatic pressure, capillary length scaling, benchmark capillary surfaces | pendant-drop/bubble profile datasets | **EA** pending experiment |
| Bulk incompressible flow | method of manufactured solutions; Poiseuille/Couette; Taylor–Green; divergence and kinetic-energy budgets | canonical laminar flows before air–film coupling | backend-dependent |
| Fixed-surface film drainage | manufactured thin-film PDE; positivity; liquid balance; spatial/time convergence | thickness-vs-time/profile measurements at controlled viscosity, gravity, boundaries | **EA** lubrication model |
| Surfactant transport | constant-state preservation; conservative advection/diffusion; adsorption equilibrium; surface-mass balance | measured Γ or proxy and surface tension under controlled chemistry | **EA** until material-specific validation |
| Marangoni coupling | prescribed surface-tension-gradient flow; stress-jump balance; coupled conservation | spreading/drainage experiments with simultaneous thickness/concentration data | **EA** |
| Moving-surface mechanics | rigid-motion invariance; geometric conservation law; sphere modes; energy/volume budgets | measured oscillation shape/frequency/damping | **EA** until mode validation |
| Vibration modes | eigenvalue convergence; orthogonality; spherical-harmonic degeneracy | Rayleigh–Lamb frequencies and damping in controlled bubbles | **PV-capable** in linear regime |
| Evaporation | diffusion/Stefan benchmark; global mass/energy balance; refinement | controlled humidity/temperature film-loss data | constitutive **EA** |
| Gas–film coupling | interface traction/flux balance; partitioned convergence; monolithic or refined-coupling reference | airflow-forced deformation/drainage experiments | one-way **EA** first |
| Disjoining pressure | linear dispersion relation and equilibrium black-film thickness; parameter sensitivity | thin-film-balance measurements | material-specific **EA/PV** |
| Rupture | instability growth rates; mesh/time/stochastic convergence; topology conservation before/after event | distributions of lifetime/location under controlled contaminants and environment | **SF** until calibrated and validated |
| Thin-film Fresnel optics | normal/oblique Fresnel limits; zero/quarter/half-wave cases; `R+T+A=1`; independent transfer-matrix comparison | angle-resolved measured reflection/transmission spectra and thickness | ideal slab **PV**; curved pointwise use **EA** |
| Spectral path transport | analytical scenes; reciprocity/energy tests; cross-renderer comparison | calibrated source/material/camera measurements | **EA/PV-capable** |
| Polarization | Brewster angle; retarders/polarizers; Jones/Mueller identities and physical realizability | polarimetric measurements | **SF** for platform integration; Mitsuba provides a reference path |
| Export/ParaView | values, units, associations, frames, time and IDs round-trip exactly | not physics | infrastructure |
| Blender/USD/glTF | geometric transform and sampled-field traceability tests | not physics | **VO** |
| XR query | coordinate-transform tests; interpolation/LOD error bounds; latency tests | user-study accuracy for education is separate | **VO** |
| AI guide | grounded-answer corpus; citation/field-ID entailment; abstention and adversarial tests | education/research usability studies | **SF**; cannot validate physics |

Mitsuba's documentation recommends spectral mode for polarized work and exposes
Mueller–Stokes representations, making it a useful independent optical backend,
not an oracle: [Mitsuba 3 polarization](https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html).

## Numerical assumptions and acceptable approximations

| Decision | Recommended policy | Classification and boundary |
|---|---|---|
| Units | SI in authoritative state; nondimensionalize solver equations with recorded reference scales | **PV-capable** |
| Precision | float64 reference path and validation; lower/mixed precision only after quantified error | lower precision is **EA** |
| Determinism | deterministic meshes/seeds/reductions where practical; record unavoidable backend nondeterminism | **EA** for parallel reductions |
| Surface representation | distinct inner/outer interfaces in the general model; mid-surface + thickness reduction when `h/L ≪ 1` | reduction is **EA** |
| Drainage model | derive lubrication equations from Navier–Stokes and interfacial balances; declare aspect ratio, Reynolds/Péclet ranges and mobility assumptions | **EA**, potentially validated in regime |
| Surface discretization | surface FEM/FV for conservation-sensitive PDEs; DDG for geometry; compare operators on reference meshes | numerically verified methods |
| Advection | conservative bounded FV/DG or stabilized FEM with reported artificial diffusion | stabilization is **EA** |
| Time integration | implicit/IMEX for stiff capillary/surfactant terms; adaptive error control; explicit only under enforced stability limits | numerically verified |
| Coupling | partitioned multirate first, with interface residual and refinement studies | **EA** until coupling error bounded |
| Remeshing | conservative field transfer and geometric conservation checks | transfer is **EA** with measured error |
| Local thin-film optics | tangent-plane transfer matrix when curvature/variation scales greatly exceed wavelength and thickness | **EA** on curved films |
| RGB rendering | derived from spectral radiance through documented observer/display transform | conversion is **VO**; RGB-only interference is **VO** |
| Airflow | prescribed or one-way traction before two-way CFD | **EA** with stated feedback neglect |
| Turbulence | DNS only when scales are resolved; LES/RANS closures labeled and validated for QoIs | LES/RANS are **EA**; no universal closure |
| Evaporation | diffusion-limited or empirical flux law with measured environment | **EA** |
| Rupture | do not equate a thickness cutoff with physical rupture; label cutoffs as numerical/visual events | threshold is **VO/EA**, resolved/calibrated model pending |
| LOD/interpolation | conservative aggregation where possible; preserve extrema and error bounds for queries | **VO**, never solver input by default |
| AI explanation | retrieve exact run/model/equation/evidence; abstain where causal attribution is unavailable | **SF** |

## Release gates

- A new numerical backend must reproduce reference kernels within declared
  tolerances and at least one convergence benchmark.
- A new model may ship as experimental **EA/SF**, but its status must be visible
  in metadata and clients.
- A **PV** badge requires reviewable evidence bundled with a release and is
  scoped to quantities/regimes.
- Breaking field semantics require a new semantic ID; schema changes require a
  migration and golden old-version fixtures.
- Performance improvements cannot weaken default tolerances or precision
  silently.
