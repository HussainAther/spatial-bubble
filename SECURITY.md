# Security Policy

Open Phenomena is an early-stage scientific computing project.

## Supported versions

Security and reliability fixes currently target the latest state of the
`main` branch and the most recent tagged release.

| Version | Supported |
|---|---|
| `main` | Yes |
| Latest tagged release | Yes |
| Older releases | Best effort |

## Reporting a vulnerability

Please do not open a public issue for a vulnerability that could expose users,
systems, credentials, files, or scientific data. Use GitHub's private
vulnerability reporting feature when available.

Include the affected version or commit, affected subsystem, reproduction steps,
possible impact, and any proposed mitigation. Do not include secrets, private
datasets, credentials, access tokens, or personally identifying data.

## Scientific correctness concerns

A suspected scientific, numerical, or validation defect is not necessarily a
security vulnerability. Use the **Scientific validation concern** issue template
for incorrect equations, sign or unit errors, reproducibility failures, invalid
convergence claims, incorrect fidelity classifications, or discrepancies with
analytic or experimental references.

## Current scope

Open Phenomena does not currently provide a hosted service, network daemon,
authentication system, or production clinical interface. It must not be used for
medical diagnosis, clinical decision-making, safety-critical control, or other
high-stakes decisions without independent validation and appropriate review.
