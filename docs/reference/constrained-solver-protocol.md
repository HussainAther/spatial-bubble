# Backend-neutral constrained-solver protocol

Status: implemented and numerically verified for v0.2.0 Milestone 3. This is
optimization infrastructure, not a solved or validated capillary surface.

## Separation of responsibilities

`openphenomena.optimization.contracts` defines immutable typed contracts and
imports only NumPy and the core scientific data model. Importing it does not
load SciPy. The SciPy implementation lives exclusively in
`openphenomena.optimization.scipy_trust_constr`; future PETSc/TAO or other
adapters must consume the same `ConstrainedProblem` and return the same
`SolverResult` without changing physical-model callables.

Physical callables receive physical variables and return physical objective,
gradient, equality residual, Jacobian, and optional second derivatives. The
problem also supplies positive diagonal variable scales, objective scale, and
one scale per equality residual. If

```text
x = D_x z,       f_hat(z) = f(x)/s_f,
c_hat(z) = D_c^-1 c(x),
```

then adapters evaluate

```text
grad_z f_hat = D_x grad_x f / s_f,
J_z c_hat = D_c^-1 J_x c D_x.
```

Exact Hessians and Hessian-vector products use the corresponding chain rule.
The SciPy adapter never asks the backend to finite-difference an objective
gradient or equality Jacobian.

## Immutable contracts

The protocol represents:

- physical initial variables, names, units, and scale-to-physical vectors;
- scalar objective value and analytic gradient;
- vector equality residuals and analytic Jacobians;
- explicit exact-Hessian, Hessian-vector, or BFGS strategies;
- backend-neutral tolerances, iteration limit, trust radius, constraint
  penalty, and deterministic history stride;
- callback diagnostics without SciPy result objects;
- normalized termination, evaluation counts, warnings, admissibility, and
  reduced iteration history;
- physical and dimensionless accepted solution candidates;
- physical/dimensionless residuals and multiplier estimates;
- callable-free solver specifications, provenance, fidelity, validation
  status, and caller-owned scientific acceptance.

Solver candidates serialize as JSON-compatible `Run.metadata` through the
existing schema v1.1 bundle. Large raw traces are intentionally not embedded;
the current adapter stores a deterministic reduced history. A future raw-trace
adapter must store a separate artifact and content hash without changing these
contracts.

## Multiplier convention

All generic equality multipliers use

```text
L_physical(x, lambda) = f_physical(x) + lambda^T c_physical(x).
```

SciPy reports multipliers for the dimensionless Lagrangian. The adapter restores
physical multipliers componentwise using

```text
lambda_physical = s_f D_c^-1 lambda_dimensionless.
```

This sign is verified on known quadratic and nonlinear problems. It is not yet
a pressure claim. The frozen capillary proposal uses
`L=E-lambda_pressure(V-V0)`, so a later capillary layer must explicitly apply
`lambda_pressure=-lambda_generic`; the adapter must not guess that meaning.

## SciPy `trust-constr` adapter

The capability ID is
`openphenomena.equilibrium.solver.scipy_trust_constr.v1`. Only
`scipy.optimize.minimize(method="trust-constr")` is invoked. There is no method
fallback, hidden regularization, or mesh-quality energy.

The deterministic option map records `gtol`, `xtol`, `barrier_tol`, `maxiter`,
initial trust radius, initial constraint penalty, disabled display, and zero
verbosity. SciPy's status is normalized separately from the raw integer and
message. Current mappings follow the official
[`trust-constr` result contract](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html):

| SciPy status | Normalized category |
|---:|---|
| 0 | iteration limit |
| 1 | converged optimality |
| 2 | converged step, unless the independent equality residual fails |
| 3 | callback stop |
| 4 | infeasible |

Constraint violation, KKT residual, objective-gradient norm, final accepted
step, and objective change are recomputed through Open Phenomena's contracts;
SciPy `success=True` does not set scientific acceptance. Rank/factorization
warnings are captured. BFGS uses an explicit SciPy `BFGS` object and is named in
backend settings and provenance. The official
[`NonlinearConstraint` contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.NonlinearConstraint.html)
defines the analytic Jacobian and multiplier-weighted constraint-Hessian
interfaces used by the adapter.

## Scientific completion policy

`BackendTermination` and `ScientificAcceptance` are independent immutable
records. A caller-supplied `AcceptancePolicy` may gate:

- normalized backend success;
- dimensionless Lagrangian/KKT infinity norm;
- equality-residual infinity norm;
- final accepted step norm;
- problem-specific admissibility.

The result always retains objective-gradient norm, objective change, iteration
and evaluation counts, multipliers, warnings, and history regardless of policy.
An evaluated policy can produce an `EvidenceRecord`; the adapter does not create
physical evidence merely because it terminated.

## Fidelity and exclusions

The adapter and its diagnostic records are numerical infrastructure. Result
records use **EA / verified** because the current scientific schema requires a
fidelity label, but this is explicitly not a physical-fidelity claim. Synthetic
analytic references are mathematical verification cases. No new **PV** claim is
made. Optimizer animation would be **VO** and is absent. PETSc/TAO, projected
Hessians, stability, and capillary equilibrium remain **SF/deferred** for this
milestone.

There is no closed-sphere recovery, curvature or Young--Laplace gate, boundary
condition enforcement, remeshing, self-intersection repair, gravity, drainage,
or other new physics here.
