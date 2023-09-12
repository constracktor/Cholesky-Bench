#!/bin/bash
################################################################################
# Diagnostics
################################################################################
set +x

################################################################################
# Variables
################################################################################
# Set MKL directory
export MKL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd )/../hpxsc_installations/hpx_mkl_v1.8.1/build/mkl/mkl/2023.0.0/lib"
# Configure MKL
export MKL_CONFIG='-DMKL_ARCH=intel64 -DMKL_LINK=dynamic -DMKL_INTERFACE_FULL=intel_lp64 -DMKL_THREADING=sequential'

################################################################################
# Compile code
################################################################################
rm -rf build && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release --DMKL_DIR="${MKL_DIR}/cmake/mkl" ${MKL_CONFIG} && make all

################################################################################
# Run BLAS benchmark
################################################################################
OUTPUT_FILE_BLAS="runtimes_mkl.txt"
rm $OUTPUT_FILE_BLAS
touch $OUTPUT_FILE_BLAS
../build/hpx_blas | tee $OUTPUT_FILE_BLAS