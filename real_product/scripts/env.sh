# Source me: Python path and OpenMP placement for the production run.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$HERE/python:$HERE/../python/src${PYTHONPATH:+:$PYTHONPATH}"
# One OpenMP team on all cores; threads stay on their cores (first-touch
# panels then stay in the local NUMA node of the thread that wrote them).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-128}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
# NumPy/BLAS in the Python driver must not oversubscribe the engine threads.
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
