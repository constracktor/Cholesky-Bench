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
	module load gcc/14.2.0
	export CC=gcc
	export CXX=g++
	spack load hpx@1.11.0%gcc@14.2.0 malloc=jemalloc
	spack load openblas@0.3.28%gcc@14.2.0 threads=none
	
    elif [[ "$HOSTNAME" == "nasrin0" || "$HOSTNAME" == "nasrin1" ]]; then
	module load gcc/14.2.0
	spack load hpx@1.11.0%gcc@14.2.0 arch=linux-almalinux9-zen3
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

cmake -DCMAKE_BUILD_TYPE=Release \
      -DHPX_IGNORE_BOOST_COMPATIBILITY=ON ..
make -j

