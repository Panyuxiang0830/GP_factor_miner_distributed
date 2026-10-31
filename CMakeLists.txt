cmake_minimum_required(VERSION 3.18)
project(gp_factor_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(GPE_ENABLE_OPENMP "Build the OpenMP backend" ON)
option(GPE_ENABLE_MPI    "Build the MPI-distributed engine" OFF)
option(GPE_ENABLE_CUDA   "Build the CUDA backend" OFF)

# Serial core: parser, DAG, operators, RankIC (E1–E8)
add_library(gpe_core
  src/core/version.cpp
  src/core/json.cpp
  src/core/ops.cpp
  src/core/dag.cpp
  src/core/panel_io.cpp
  src/core/operators.cpp
  src/core/evaluate.cpp
)
target_include_directories(gpe_core PUBLIC include)

if(GPE_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(gpe_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(gpe_core PUBLIC GPE_HAVE_OPENMP)
endif()

if(GPE_ENABLE_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(gpe_core PUBLIC MPI::MPI_CXX)
  target_compile_definitions(gpe_core PUBLIC GPE_HAVE_MPI)
endif()

if(GPE_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 86)   # RTX 3090; override for other GPUs
  target_compile_definitions(gpe_core PUBLIC GPE_HAVE_CUDA)
endif()

enable_testing()
add_executable(test_smoke tests/test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE gpe_core)
add_test(NAME smoke COMMAND test_smoke)

add_executable(test_dag tests/test_dag.cpp)
target_link_libraries(test_dag PRIVATE gpe_core)
target_compile_definitions(test_dag PRIVATE GPE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
add_test(NAME dag COMMAND test_dag)

# Needs exported market data, so it runs only when GPE_REF_DIR is given.
add_executable(test_ops_reference tests/test_ops_reference.cpp)
target_link_libraries(test_ops_reference PRIVATE gpe_core)
if(GPE_REF_DIR)
  add_test(NAME ops_reference COMMAND test_ops_reference ${GPE_REF_DIR})
endif()
