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

# Patched Python runtimes may ignore underscore-prefixed .pth files. Hatchling
# currently emits one for editable installs, so provide an equivalent visible
# path file when necessary. This preserves standard entry-point metadata while
# making a fresh interpreter and plugin discovery reliable.
site_packages=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
hidden_editable_path="$site_packages/_editable_impl_openphenomena.pth"
visible_editable_path="$site_packages/openphenomena_editable.pth"
if [ -f "$hidden_editable_path" ]; then
  mv "$hidden_editable_path" "$visible_editable_path"
fi

.venv/bin/python -m openphenomena.reference \
  --output outputs/static-spherical-bubble
