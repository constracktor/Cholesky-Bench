#!/bin/bash
################################################################################
# Diagnostics
################################################################################
set +x

################################################################################
# Variables
################################################################################
# Set MKL directory
#export MKL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd )/../hpxsc_installations/hpx_dist_v1.8.1/build/mkl/mkl/2023.0.0/lib/cmake/mkl"
# Configure MKL
export MKL_CONFIG='-DMKL_ARCH=intel64 -DMKL_LINK=dynamic -DMKL_INTERFACE_FULL=intel_lp64 -DMKL_THREADING=openmp'
module load gcc
#llvm/20.1.8
#spack load openblas%gcc@15.1.0 threads=openmp
spack load openblas@0.3.28%gcc@14.2.0 arch=linux-almalinux9-zen3 threads=none
spack load intel-oneapi-mkl%gcc@15.1.0 threads=openmp
#export CC=clang
#export CXX=clang++
#export LD_LIBRARY_PATH=/home/alex/spack/opt/spack/linux-almalinux9-zen3/clang-20.1.8/openblas-0.3.28-*/lib:$LD_LIBRARY_PATH

################################################################################
# Compile code
################################################################################
rm -rf build && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make all VERBOSE=1
#-DMKL_DIR="${MKL_DIR}" ${MKL_CONFIG} && make all

################################################################################
# Run BLAS benchmark
################################################################################
OUTPUT_FILE_BLAS="runtimes_openblas.txt"
touch $OUTPUT_FILE_BLAS
OMP_NUM_THREADS=128 OMP_PROC_BIND=close OMP_PLACES=cores ./mkl_benchmark
#| tee $OUTPUT_FILE_BLAS
