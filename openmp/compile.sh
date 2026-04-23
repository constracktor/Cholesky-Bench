#!/bin/bash
# $1: python/cpp
# $2: cpu/gpu
# $3: mkl
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.

################################################################################
# Configurations
################################################################################
if command -v spack &> /dev/null; then
    echo "Spack command found. Loading libraries."
    # Get current hostname
    HOSTNAME=$(hostname -s)

    if [[ "$HOSTNAME" == "ipvs-epyc1" ]]; then
	# GCC
	# module load gcc/14.2.0
	# export CC=gcc
	# export CXX=g++
	# LLVM
	spack load llvm@22.1.2
	export CC=clang
	export CXX=clang++
	export LD_LIBRARY_PATH=$(spack location -i llvm@22.1.2)/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH
	# OpenBLAS
	spack load openblas@0.3.28%gcc@14.2.0 threads=none
	
    elif [[ "$HOSTNAME" == "nasrin0" || "$HOSTNAME" == "nasrin1" ]]; then
	module load gcc/14.2.0
	spack load openblas@0.3.28%gcc@14.2.0 arch=linux-almalinux9-zen3 threads=none

    else
    	echo "Hostname is $HOSTNAME — no action taken."
    fi
else
    echo "Spack command not found. Exiting."
fi

################################################################################
# Compile code
################################################################################
rm -rf build && mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j VERBOSE=1

cd ..

OMP_NUM_THREADS=128 OMP_PROC_BIND=close OMP_PLACES=cores OMP_MAX_TASK_PRIORITY=16 ./build/cholesky_openmp #--size_stop 256 --loop 5
