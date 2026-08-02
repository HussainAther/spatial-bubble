#!/usr/bin/env bash
set -euo pipefail

python_executable="${PYTHON:-}"
if [ -z "${python_executable}" ]; then
  if [ -x .venv/bin/python ]; then
    python_executable=.venv/bin/python
  else
    python_executable=python3
  fi
fi
"${python_executable}" -m openphenomena.equilibrium.reference \
  --output outputs/closed-sphere
