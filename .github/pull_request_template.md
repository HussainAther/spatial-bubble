## Summary

Describe the change and the problem it addresses.

## Change category

- [ ] Scientific model
- [ ] Numerical method
- [ ] Solver backend
- [ ] Scientific data model
- [ ] Validation or evidence
- [ ] Export or visualization
- [ ] Documentation
- [ ] Infrastructure
- [ ] Bug fix

## Scientific scope

State the governing model, assumptions, affected quantities, and applicable
regime.

Write `Not applicable` for changes without a scientific claim.

## Fidelity classification

- [ ] PV — Physically validated within a stated regime
- [ ] EA — Engineering approximation
- [ ] VO — Visualization-only
- [ ] SF — Speculative future
- [ ] No fidelity-bearing result

Explain the classification:

## Verification and validation

Describe:

- analytic references;
- convergence studies;
- cross-implementation comparisons;
- experimental evidence;
- regression tests;
- acceptance thresholds.

## Compatibility

- [ ] Existing v0.1.0 bundles remain readable.
- [ ] Existing accepted numerical values remain unchanged.
- [ ] Schema changes are additive or include migration coverage.
- [ ] No compatibility impact.

## Reproduction

```bash
# Exact commands used
Quality gates
 Tests pass
 Ruff formatting passes
 Ruff lint passes
 Strict mypy passes
 Compilation passes
 Plugin discovery passes
 Deterministic-output checks pass where applicable
 Documentation is updated
Checklist
 No threshold was relaxed merely to make this change pass.
 Units and coordinate frames are explicit.
 Visualization outputs are not treated as authoritative state.
 Optimizer success is not treated as scientific validation.
 New dependencies are justified and documented.

---

## 10. `CODE_OF_CONDUCT.md`

A concise early-stage version:

```markdown
# Code of Conduct

## Our commitment

We are committed to providing a welcoming, respectful, and constructive
environment for contributors, users, researchers, educators, and students.

Participants are expected to engage with scientific disagreement carefully and
professionally.

## Expected behavior

Examples of positive behavior include:

- giving specific, evidence-based feedback;
- distinguishing mistakes from misconduct;
- stating assumptions and uncertainty;
- respecting differences in experience and background;
- accepting corrections;
- focusing criticism on ideas, code, methods, and evidence;
- helping others reproduce results.

## Unacceptable behavior

Unacceptable behavior includes:

- harassment, threats, or personal attacks;
- discriminatory or demeaning language;
- deliberate misrepresentation of another person's work;
- publishing private information without permission;
- repeated disruption of technical discussions;
- knowingly presenting fabricated data or evidence;
- retaliation against good-faith bug or validation reports.

## Scientific disagreement

Disagreement about models, methods, evidence, or interpretation is welcome.

Contributors should provide:

- the quantity or claim under discussion;
- the applicable physical regime;
- units and assumptions;
- reproducible evidence;
- relevant references where possible.

Strong criticism of a method is acceptable. Personal hostility is not.

## Enforcement

Project maintainers may edit, remove, or reject contributions that violate this
policy and may temporarily or permanently restrict participation.

Reports should be made privately to the project maintainer when public
discussion would expose private information or create additional harm.

## Scope

This policy applies to project repositories, issues, pull requests,
discussions, documentation, and project-related public communications.
