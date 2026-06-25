// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "functions.hpp"
#include "tile_generation.hpp"
#ifdef ENABLE_VALIDATION
#include "validate.hpp"
#endif

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <ttg.h>
#include <vector>

namespace
{
// Minimal "--name value" command-line lookup (TTG has no program_options).
std::size_t arg_value(int argc, char **argv, const std::string &name, std::size_t fallback)
{
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (name == argv[i])
        {
            return static_cast<std::size_t>(std::strtoull(argv[i + 1], nullptr, 10));
        }
    }
    return fallback;
}

long arg_threads(int argc, char **argv)
{
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (std::string("--threads") == argv[i])
        {
            return std::strtol(argv[i + 1], nullptr, 10);
        }
    }
    return -1;  // let the runtime decide
}
}  // namespace

int main(int argc, char *argv[])
{
    ///////////////////////////////////////////////////////////////////////////
    // cmdline arguments
    const std::size_t LOOP = arg_value(argc, argv, "--loop", 1);

    const std::size_t START_SIZE = arg_value(argc, argv, "--size_start", 32);
    const std::size_t STOP_SIZE = arg_value(argc, argv, "--size_stop", 128);
    const std::size_t STEP_SIZE = 2;

    const std::size_t START_TILES = arg_value(argc, argv, "--tiles_start", 16);
    const std::size_t STOP_TILES = arg_value(argc, argv, "--tiles_stop", 32);
    const std::size_t STEP_TILES = 2;

    const long nthreads = arg_threads(argc, argv);

    ///////////////////////////////////////////////////////////////////////////
    // Start the TTG runtime
    ttg::initialize(argc, argv, static_cast<int>(nthreads));
    const bool is_root = (ttg::default_execution_context().rank() == 0);

    const std::size_t reported_threads =
        (nthreads > 0) ? static_cast<std::size_t>(nthreads) : std::thread::hardware_concurrency();

    // print and write results
    bool HEADER_FLAG = true;
    std::string runtime_file_path = "runtimes_ttg_cholesky_";
    if (START_TILES != STOP_TILES)
    {
        runtime_file_path += std::string("tile_");
    }
    if (START_SIZE != STOP_SIZE)
    {
        runtime_file_path += std::string("size_");
    }
    runtime_file_path += std::to_string(LOOP) + std::string(".txt");
    std::ofstream runtime_file;
    if (is_root)
    {
        runtime_file.open(runtime_file_path, std::ios_base::app);
    }

    for (std::size_t n_tiles = START_TILES; n_tiles <= STOP_TILES; n_tiles = n_tiles * STEP_TILES)
    {
        for (std::size_t size = START_SIZE; size <= STOP_SIZE; size = size * STEP_SIZE)
        {
            for (std::size_t l = 0; l < LOOP; l++)
            {
                // header for output file
                std::string header = "threads;problem_size;tile_size;n_tiles";
                // runtime config and values
                std::string values = std::to_string(reported_threads);
                values += std::string(";") + std::to_string(size);
                values += std::string(";") + std::to_string(size / n_tiles);
                values += std::string(";") + std::to_string(n_tiles);
#ifdef ENABLE_VALIDATION
                // Relative residual ||A - L L^T||_F / ||A||_F
                constexpr double residual_tol = 1e-10;
                auto report_residual = [&](const std::string &mode, double residual)
                {
                    std::cout << "[validate] mode=" << mode << " size=" << size << " n_tiles=" << n_tiles
                              << " residual=" << residual << std::endl;
                    if (!(residual <= residual_tol))  // catches NaN too
                    {
                        std::cerr << "Validation warning: variant '" << mode << "' residual " << residual
                                  << " exceeds tolerance " << residual_tol << " (size=" << size
                                  << ", n_tiles=" << n_tiles << ")" << std::endl;
                    }
                };
#endif
                ///////////////////////////////////////////////////////////////////////////
                // TTG dataflow
                {
                    auto tiled_matrix = gen_tiled_matrix(size, n_tiles);
                    auto cholesky_cpu = cpu::cholesky_flow(tiled_matrix);

                    header += ";flow";
                    values += ";" + std::to_string(cholesky_cpu);

#ifdef ENABLE_VALIDATION
                    double residual = cpu::cholesky_residual(size, n_tiles, tiled_matrix);
                    if (is_root)
                    {
                        report_residual("flow", residual);
                    }
#endif
                }
                ///////////////////////////////////////////////////////////////////////////
                // print/write header only once
                if (is_root)
                {
                    if (HEADER_FLAG)
                    {
                        HEADER_FLAG = false;
                        std::cout << header << std::endl;
                        runtime_file << header << std::endl;
                    }
                    // print/write runtimes
                    std::cout << values << std::endl;
                    runtime_file << values << std::endl;
                }
            }
        }
    }

    if (is_root)
    {
        runtime_file.close();
    }

    ttg::finalize();
    return 0;
}
