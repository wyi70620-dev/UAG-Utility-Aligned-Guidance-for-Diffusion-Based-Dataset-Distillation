#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Install optional upstream imports in THIS project; leave the original conda env intact.
"$UAG_PYTHON" -m pip install --no-deps --target "$UAG_ROOT/.deps" -r requirements-extras.txt
