#!/usr/bin/env bash
# Build libgpp_native, gpp_bench and the C++ tests.
#   scripts/build.sh            # CUDA if nvcc is found, else CPU only
#   GPP_CUDA=OFF scripts/build.sh
#   BUILD_DIR=build-cpu scripts/build.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
CUDA="${GPP_CUDA:-auto}"
if [[ "$CUDA" == auto ]]; then
  if command -v nvcc >/dev/null 2>&1; then CUDA=ON; else CUDA=OFF; fi
fi
cmake -S "$HERE" -B "$HERE/$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DGPP_ENABLE_OPENMP=ON -DGPP_ENABLE_CUDA="$CUDA" -DGPP_CUDA_ARCH="${GPP_CUDA_ARCH:-89}"
cmake --build "$HERE/$BUILD_DIR" -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
(cd "$HERE/$BUILD_DIR" && ctest --output-on-failure)
echo "built $HERE/$BUILD_DIR (CUDA=$CUDA)"
