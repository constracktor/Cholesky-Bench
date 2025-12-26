cmake_minimum_required(VERSION 3.16)
project(mkl_benchmark CXX)

find_package(MKL CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

set(HEADER_FILES mkl_adapter.hpp tiled_cholesky.hpp)
# Try to find OpenBLAS
find_library(OpenBLAS_LIB NAMES openblas REQUIRED)

if(OpenBLAS_LIB)
      message(STATUS "OpenBLAS Library found at ${OpenBLAS_LIB}")
else()
      message(FATAL_ERROR "No BLAS Library found")
endif()
find_path(OpenBLAS_INCLUDE_DIR
    NAMES cblas.h
    PATH_SUFFIXES openblas
)
if (NOT OpenBLAS_INCLUDE_DIR)
    message(FATAL_ERROR "OpenBLAS include directory not found")
endif()

message(STATUS "OpenBLAS include dir: ${OpenBLAS_INCLUDE_DIR}")

add_executable(mkl_benchmark main.cpp ${HEADER_FILES})
target_link_libraries(mkl_benchmark PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(mkl_benchmark PUBLIC ${OpenBLAS_LIB})
target_include_directories(mkl_benchmark PUBLIC ${OpenBLAS_INCLUDE_DIR})
#target_link_libraries(mkl_benchmark PUBLIC MKL::MKL)
                        #MKL::mkl_intel_lp64
                        #MKL::mkl_core
                        #MKL::mkl_sequential
                        #                       )
