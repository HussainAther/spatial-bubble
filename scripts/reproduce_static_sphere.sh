#!/bin/sh
set -eu

python_command=${OPENPHENOMENA_PYTHON:-}
if [ -z "$python_command" ]; then
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 \
      && "$candidate" -c 'import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] < (3, 14)))'
    then
      python_command=$candidate
      break
    fi
  done
fi
if [ -z "$python_command" ]; then
  echo "Python 3.11, 3.12, or 3.13 is required for this release." >&2
  exit 1
fi

"$python_command" -m venv --clear .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m openphenomena.reference \
  --output outputs/static-spherical-bubble
