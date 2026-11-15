#!/bin/bash
# Small source-layout regression: no market files or external manifests needed.
set -euo pipefail
: "${PCA_RUN:?set PCA_RUN to a new directory under pca/runs}"
PCA_SMOKE_CUDA=${PCA_SMOKE_CUDA:-0}
case "$PCA_SMOKE_CUDA" in 0|1) ;; *) exit 2 ;; esac
python3 - "$PCA_RUN" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]).resolve()
assert not p.exists() and p != Path('pca/runs').resolve()
p.relative_to(Path('pca/runs').resolve())
p.mkdir(parents=True)
PY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
module load SciPy-bundle/2025.06-gfbf-2025a
python3 pca/synthetic.py --out "$PCA_RUN/input" --rows 24 --cols 257 --factors 8
module purge
module load OpenMPI/5.0.8-GCC-14.3.0 CMake/4.0.3-GCCcore-14.3.0
cuda=OFF
if [[ "$PCA_SMOKE_CUDA" == 1 ]]; then
    module load CUDA/12.9.1
    cuda=ON
fi
build="${PCA_RUN}_build"
cmake -S pca -B "$build" -DPCA_MPI=ON -DPCA_CUDA="$cuda" -DPCA_CUBLAS_REFERENCE="$cuda"
cmake --build "$build" -j 2
ctest --test-dir "$build" --output-on-failure > "$PCA_RUN/ctest.log" 2>&1
export PRTE_MCA_prte_launch_agent="$PWD/scripts/vanda/prted.sh"
mpirun -np 4 --map-by slot --bind-to core "$build/pca_archive_test" > "$PCA_RUN/mpi_archive.log" 2>&1
manifest="$PCA_RUN/input/manifest.json"
for raw in 0 1; do
    "$build/pca_run" --manifest "$manifest" --out "$PCA_RUN/serial_raw${raw}" \
        --count 8 --chunk 4 --layout factor --rank-input "$raw"
    for layout in date transpose factor; do
        mpirun -np 4 --map-by slot --bind-to core "$build/pca_run" \
            --manifest "$manifest" --out "$PCA_RUN/mpi_${layout}_raw${raw}" \
            --count 8 --chunk 4 --layout "$layout" --rank-input "$raw"
    done
done
mpirun -np 4 --map-by slot --bind-to core "$build/pca_run" \
    --manifest "$manifest" --out "$PCA_RUN/mpi_archive" --count 8 --chunk 4 --archive mpi
mpirun -np 4 --map-by slot --bind-to core "$build/pca_run" \
    --manifest "$manifest" --out "$PCA_RUN/lanczos" --count 8 --chunk 4 --method lanczos --steps 6
if [[ "$PCA_SMOKE_CUDA" == 1 ]]; then
    "$build/pca_gpu_test" "$PCA_RUN/gpu_validation.json"
    for precision in float double; do
        for layout in date transpose factor; do
            mpirun -np 1 --map-by slot --bind-to core "$build/pca_run" \
                --manifest "$manifest" --out "$PCA_RUN/gpu_${precision}_${layout}" \
                --count 8 --chunk 4 --device cuda --precision "$precision" --layout "$layout" --cuda-aware 1
        done
        mpirun -np 1 --map-by slot --bind-to core "$build/pca_run" \
            --manifest "$manifest" --out "$PCA_RUN/gpu_${precision}_raw" \
            --count 8 --chunk 4 --device cuda --precision "$precision" --layout factor --rank-input 1
    done
    export NVIDIA_TF32_OVERRIDE=0
    "$build/pca_cublas_reference" 8 128 293 "$PCA_RUN/cublas_validation.json"
    ldd "$build/pca_run" > "$PCA_RUN/production_dependencies.txt"
fi
module purge
module load SciPy-bundle/2025.06-gfbf-2025a
for run in "$PCA_RUN"/*/metrics.json; do
    directory=${run%/metrics.json}
    python3 pca/validate.py --manifest "$manifest" --run "$directory" --svd-count 2 \
        > "$directory/validation.log" 2>&1
done
python3 - "$PCA_RUN" "$PCA_SMOKE_CUDA" <<'PY'
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
for raw in (0,1):
    expected=(root/f'serial_raw{raw}'/'signatures.f32').read_bytes()
    for layout in ('factor','transpose'):
        assert (root/f'mpi_{layout}_raw{raw}'/'signatures.f32').read_bytes()==expected
validations=[json.loads(p.read_text()) for p in root.glob('*/validation.json')]
assert len(validations)==(18 if sys.argv[2]=='1' else 10)
assert all(x['passed'] for x in validations)
assert max(json.loads((root/'lanczos/validation.json').read_text())['svd_one_minus_abs_cosine'])<=1e-6
if sys.argv[2]=='1':
    assert json.loads((root/'gpu_validation.json').read_text())['passed']
    assert json.loads((root/'cublas_validation.json').read_text())['passed']
    assert 'cublas' not in (root/'production_dependencies.txt').read_text().lower()
(root/'SUCCESS.json').write_text(json.dumps(dict(verified=True,scenarios=len(validations),
    synthetic_only=True,mpi_factor_transpose_bitwise=True,cuda=sys.argv[2]=='1',
    production_uses_cublas=False),indent=2)+'\n')
PY
touch "$PCA_RUN/SUCCESS"
