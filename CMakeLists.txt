cmake_minimum_required(VERSION 3.16)
project(mkl_benchmark CXX)

find_package(MKL CONFIG REQUIRED)

set(HEADER_FILES mkl_adapter.hpp)

add_executable(mkl_benchmark main.cpp ${HEADER_FILES})
target_link_libraries(mkl_benchmark 
                        MKL::mkl_intel_lp64
                        MKL::mkl_core
                        MKL::MKL
                        MKL::mkl_sequential)