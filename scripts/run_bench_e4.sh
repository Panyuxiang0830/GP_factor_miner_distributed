#!/bin/bash
# E4 experiments (MPI + OpenMP): strong scaling (MPI, OpenMP, hybrid at 32
# threads), weak scaling by date and the H0 sweep.
#
#   PANEL_DIR=... EXPRS_FILE=... OUT_DIR=... scripts/run_bench_e4.sh
#
# Needs build_mpi/gpe_bench. Keep the total thread count below the machine's
# CPU quota, otherwise throttling distorts the timings.
set -u
cd "$(dirname "$0")/.."
: "${PANEL_DIR:?exported panel directory}" "${EXPRS_FILE:?expression list}" "${OUT_DIR:?output directory}"
EXPRS=${EXPRS:-200}
REPEAT=${REPEAT:-3}
mkdir -p "$OUT_DIR"
CSV=$OUT_DIR/results.csv

run() {  # label P threads H0 dates
  local label=$1 p=$2 t=$3 h=$4 dates=$5
  OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores \
  mpirun -np "$p" --bind-to none -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
    scripts/pin_ranks.sh build_mpi/gpe_bench "$PANEL_DIR" "$EXPRS_FILE" --exprs "$EXPRS" --repeat "$REPEAT" \
    --H0 "$h" --dates "$dates" --csv "$CSV" --label "$label" > "$OUT_DIR/last.log" 2>&1
  tail -1 "$OUT_DIR/last.log"
}

T=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['T'])" "$PANEL_DIR/meta.json")
echo "start $(date)"
for p in 1 2 4 8 16 24; do run strong_mpi "$p" 1 0 "$T"; done
for t in 2 4 8 16 32; do run strong_omp 1 "$t" 0 "$T"; done
for pt in "1 32" "2 16" "4 8" "8 4" "16 2"; do set -- $pt; run hybrid32 "$1" "$2" 0 "$T"; done
for p in 1 2 4 8 16; do run weak_mpi "$p" 1 0 $((91 * p)); done
for h in 0 10 30 60 89; do run h0_sweep 16 2 "$h" "$T"; done
echo "done $(date)"
