#!/usr/bin/env bash
# Measurements to replace the README estimates on the target host.
#   scripts/bench.sh DATA_DIR/evolution [BUDGET_GB]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HERE/scripts/env.sh"
PANELS="$1"; BUDGET="${2:-180}"
BIN="$HERE/build/gpp_bench"
# 1. planner only: groups, peak panels, recomputation for a 100k depth-5 batch
"$BIN" --panels "$PANELS" --random 100000 --depth 5 --genes 20000 --budget-gb "$BUDGET" --plan-only
# 2. CPU scoring vs thread count (10k batch, same shape as the report's)
for t in 32 64 128; do
  "$BIN" --panels "$PANELS" --random 10000 --depth 3 --genes 2000 --threads "$t" --budget-gb "$BUDGET" --gpus 0
done
# 3. GPU scorer (needs a CUDA build)
for g in 1 2 4; do
  "$BIN" --panels "$PANELS" --random 10000 --depth 3 --genes 2000 --threads 128 --budget-gb "$BUDGET" --gpus "$g" || true
done
