# Support

Open Phenomena is an early-stage open-source scientific software project.

## Where to ask questions

Use GitHub Issues for:

- reproducible software defects;
- scientific validation concerns;
- documentation problems;
- focused feature proposals.

Use GitHub Discussions, when enabled, for:

- general questions;
- research ideas;
- possible collaborations;
- interpretation of project results;
- broader roadmap discussion.

## Before opening an issue

Please check:

1. the README;
2. the relevant document under `docs/`;
3. existing issues;
4. the current project limitations;
5. whether the behavior reproduces from a clean environment.

Include the following when relevant:

```text
Operating system:
Python version:
Open Phenomena version or commit:
Installation command:
Reproduction command:
Expected result:
Observed result:
Relevant output artifact:
```

Support expectations
This project is maintained on a best-effort basis.
Response times are not guaranteed. Issues with a complete reproduction,
scientific context, units, and evidence artifacts are more likely to receive a
useful response.
Unsupported use
The project is not currently intended for:
clinical diagnosis;
safety-critical engineering;
production industrial control;
claims of experimental validation beyond the documented evidence;
unsupported physical regimes;
confidential or regulated data.

---

## 5. `.github/ISSUE_TEMPLATE/bug_report.yml`

```yaml
name: Bug report
description: Report reproducible incorrect software behavior
title: "[Bug]: "
labels:
  - bug
body:
  - type: markdown
    attributes:
      value: |
        Thank you for helping improve Open Phenomena.

  - type: input
    id: version
    attributes:
      label: Version or commit
      description: Provide the release, tag, or commit hash.
      placeholder: v0.1.0-static-sphere or 273cc7e
    validations:
      required: true

  - type: input
    id: environment
    attributes:
      label: Environment
      description: Operating system and Python version.
      placeholder: macOS 15, Python 3.13
    validations:
      required: true

  - type: textarea
    id: description
    attributes:
      label: Problem
      description: Describe what went wrong.
    validations:
      required: true

  - type: textarea
    id: reproduction
    attributes:
      label: Reproduction steps
      description: Include the smallest command or code sample that reproduces the issue.
      render: shell
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: Expected behavior
    validations:
      required: true

  - type: textarea
    id: observed
    attributes:
      label: Observed behavior
    validations:
      required: true

  - type: textarea
    id: artifacts
    attributes:
      label: Relevant artifacts
      description: Include paths, hashes, reports, manifests, or logs where appropriate.

  - type: checkboxes
    id: checks
    attributes:
      label: Checklist
      options:
        - label: I searched existing issues.
          required: true
        - label: I removed secrets and private data.
          required: true
