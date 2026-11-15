cmake_minimum_required(VERSION 3.18)
project(gp_real_product LANGUAGES CXX)

# Production pipeline for one 128-core / 4 x RTX 4090 host. It reuses the
# course engine sources (../src/core, ../include) and the PCA core (../pca)
# without modifying them; everything new lives in real_product/.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(GPP_ENABLE_OPENMP "OpenMP (required for real runs)" ON)
option(GPP_ENABLE_CUDA "CUDA scorer and similarity products (RTX 4090: sm_89)" OFF)
option(GPP_NATIVE "Compile for the build machine (-march=native)" ON)
option(GPP_BUILD_TESTS "C++ tests" ON)
set(GPP_CUDA_ARCH "89" CACHE STRING "CUDA architectures (RTX 4090 = 89)")

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

if(GPP_NATIVE AND NOT MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" GPP_HAS_MARCH_NATIVE)
  if(GPP_HAS_MARCH_NATIVE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
  endif()
endif()

# No fused multiply-add contraction: the CUDA scorer reproduces the CPU
# scoring bit for bit by running the same operation sequence (nvcc
# --fmad=false), which only holds if the CPU does not contract either.
if(NOT MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
endif()

# Course engine core, compiled from the original files (unchanged).
add_library(gpe_core_reuse STATIC
  ${REPO_ROOT}/src/core/version.cpp
  ${REPO_ROOT}/src/core/json.cpp
  ${REPO_ROOT}/src/core/ops.cpp
  ${REPO_ROOT}/src/core/dag.cpp
  ${REPO_ROOT}/src/core/panel_io.cpp
  ${REPO_ROOT}/src/core/operators.cpp
  ${REPO_ROOT}/src/core/evaluate.cpp
  ${REPO_ROOT}/src/core/metrics.cpp
  ${REPO_ROOT}/src/core/ema_scan.cpp
  ${REPO_ROOT}/src/core/halo_plan.cpp
  ${REPO_ROOT}/pca/core.cpp
)
target_include_directories(gpe_core_reuse PUBLIC ${REPO_ROOT}/include ${REPO_ROOT}/pca)

add_library(gpp_prod STATIC
  src/scoring.cpp
  src/engine.cpp
  src/archive.cpp
  src/screening.cpp
)
target_include_directories(gpp_prod PUBLIC include)
target_link_libraries(gpp_prod PUBLIC gpe_core_reuse)

if(GPP_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(gpe_core_reuse PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(gpe_core_reuse PUBLIC GPE_HAVE_OPENMP)
  else()
    message(WARNING "OpenMP not found: building a single-threaded engine (fine for tests, not for runs)")
  endif()
endif()

if(GPP_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_library(gpp_cuda STATIC cuda/gpu_scorer.cu cuda/gpu_products.cu)
  set_target_properties(gpp_cuda PROPERTIES CUDA_ARCHITECTURES "${GPP_CUDA_ARCH}"
                                            POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION OFF)
  target_include_directories(gpp_cuda PUBLIC include ${REPO_ROOT}/include ${REPO_ROOT}/pca)
  target_compile_options(gpp_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo --fmad=false -prec-div=true -prec-sqrt=true
                                -Xcompiler=-ffp-contract=off>)
  target_link_libraries(gpp_cuda PUBLIC CUDA::cudart CUDA::cublas)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(gpp_cuda PUBLIC OpenMP::OpenMP_CXX)
  endif()
  target_link_libraries(gpp_prod PUBLIC gpp_cuda)
  target_compile_definitions(gpp_prod PUBLIC GPP_HAVE_CUDA)
endif()

# ctypes library used by python/gpp.
add_library(gpp_native SHARED src/c_api.cpp)
target_link_libraries(gpp_native PRIVATE gpp_prod)
set_target_properties(gpp_native PROPERTIES OUTPUT_NAME gpp_native)

add_executable(gpp_bench apps/gpp_bench.cpp)
target_link_libraries(gpp_bench PRIVATE gpp_prod)

if(GPP_BUILD_TESTS)
  enable_testing()
  add_executable(test_gpp_engine tests/cpp/test_engine.cpp)
  target_link_libraries(test_gpp_engine PRIVATE gpp_prod)
  add_test(NAME gpp_engine COMMAND test_gpp_engine)
  add_executable(test_gpp_archive tests/cpp/test_archive.cpp)
  target_link_libraries(test_gpp_archive PRIVATE gpp_prod)
  add_test(NAME gpp_archive COMMAND test_gpp_archive)
  if(GPP_ENABLE_CUDA)
    add_executable(test_gpp_gpu tests/cpp/test_gpu.cpp)
    target_link_libraries(test_gpp_gpu PRIVATE gpp_prod)
    add_test(NAME gpp_gpu COMMAND test_gpp_gpu)
  endif()
endif()
