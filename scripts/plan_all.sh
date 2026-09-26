#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
for suite in main ablation references plugins diagnostics redundancy baselines; do
  "$UAG_PYTHON" -m uag.experiments --suite "$suite" --plan "plans/$suite.json"
done
