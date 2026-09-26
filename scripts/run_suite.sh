#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Example: bash scripts/run_suite.sh main --datasets imagewoof --execute
suite="${1:-main}"
if [[ $# -gt 0 ]]; then shift; fi
"$UAG_PYTHON" -m uag.experiments --suite "$suite" --plan "plans/$suite.json" "$@"
