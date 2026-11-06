#!/usr/bin/env bash
# Run on an allocated CPU compute node; outputs are private experiment files.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "usage: bash tools/dag_analysis/run_s4.sh panels batch.json output_dir" >&2
  exit 2
fi
: "${PBS_JOBID:?Run this script in an allocated PBS compute job}"
panels=$1
batch=$2
out=$3
binary=${GPE_EXPORT:-./build_s4/gpe_export}
[[ -x "$binary" && -f "$batch" && -d "$panels" ]] || { echo 'Missing binary or inputs' >&2; exit 2; }
mkdir -p "$out"
if compgen -G "$out/time_*.txt" > /dev/null; then
  echo 'Output already contains timing files; choose a fresh directory' >&2
  exit 2
fi
for c in 16 64 200; do
  for r in 1 2 3; do
    /usr/bin/time -f "chunk=$c repeat=$r elapsed_s=%e peak_KiB=%M user_s=%U system_s=%S exit=%x" \
      -o "$out/time_c${c}_r${r}.txt" \
      "$binary" "$panels" "$batch" --exprs 200 --chunk "$c" --no-R \
      --out "$out/c${c}_r${r}" > "$out/run_c${c}_r${r}.log" 2>&1
  done
done
python3 "$(dirname "$0")/verify_s4.py" "$out"
cat "$out"/time_*.txt
