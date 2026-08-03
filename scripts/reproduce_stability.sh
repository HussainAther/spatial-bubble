#!/bin/sh
set -eu
PYTHON_BIN="${OPENPHENOMENA_PYTHON:-python}"
"$PYTHON_BIN" -m openphenomena.equilibrium.stability_reference --output outputs/stability-sphere
