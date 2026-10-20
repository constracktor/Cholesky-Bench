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
#include <starpu.h>
#include <string>
#include <vector>

namespace
{
// Minimal command-line lookup accepting both "--name value" and "--name=value".
// Returns nullptr if the option is absent.
const char *find_arg(int argc, char **argv, const std::string &name)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == name)
        {
            return (i + 1 < argc) ? argv[i + 1] : nullptr;
        }
        if (arg.size() > name.size() && arg.compare(0, name.size(), name) == 0 && arg[name.size()] == '=')
        {
            return argv[i] + name.size() + 1;
        }
    }
    return nullptr;
}

std::size_t arg_value(int argc, char **argv, const std::string &name, std::size_t fallback)
{
    const char *value = find_arg(argc, argv, name);
    return value ? static_cast<std::size_t>(std::strtoull(value, nullptr, 10)) : fallback;
}

long arg_threads(int argc, char **argv)
{
    const char *value = find_arg(argc, argv, "--threads");
    return value ? std::strtol(value, nullptr, 10) : -1;  // -1: let the runtime decide
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
    // Initialize StarPU: CPU workers only. The scheduler is chosen via the
    // STARPU_SCHED environment variable (default: lws).
    starpu_conf conf;
    starpu_conf_init(&conf);
    conf.ncuda = 0;
    conf.nopencl = 0;
    if (nthreads > 0)
    {
        conf.ncpus = static_cast<int>(nthreads);
    }
    STARPU_CHECK_RETURN_VALUE(starpu_init(&conf), "starpu_init");

    // STARPU_NCPU in the environment overrides --threads, so always report the actual worker count.
    const std::size_t reported_threads = static_cast<std::size_t>(starpu_cpu_worker_get_count());
    std::cout << "[starpu] scheduler=" << starpu_sched_get_sched_policy()->policy_name
              << " workers=" << reported_threads << std::endl;

    // print and write results
    bool HEADER_FLAG = true;
    std::string runtime_file_path = "runtimes_starpu_cholesky_";
    if (START_TILES != STOP_TILES)
    {
        runtime_file_path += std::string("tile_");
    }
    if (START_SIZE != STOP_SIZE)
    {
        runtime_file_path += std::string("size_");
    }
    runtime_file_path += std::to_string(LOOP) + std::string(".txt");
    std::ofstream runtime_file(runtime_file_path, std::ios_base::app);

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
                // StarPU sequential task flow (asynchronous)
                {
                    auto tiled_matrix = gen_tiled_matrix(size, n_tiles);
                    auto cholesky_cpu = cpu::cholesky_stf(tiled_matrix);

                    header += ";stf";
                    values += ";" + std::to_string(cholesky_cpu);

#ifdef ENABLE_VALIDATION
                    double residual = cpu::cholesky_residual(size, n_tiles, tiled_matrix);
                    report_residual("stf", residual);
#endif
                }
                ///////////////////////////////////////////////////////////////////////////
                // StarPU sequential task flow with a barrier after each phase
                {
                    auto tiled_matrix = gen_tiled_matrix(size, n_tiles);
                    auto cholesky_cpu = cpu::cholesky_stf(tiled_matrix, true);

                    header += ";stf_sync";
                    values += ";" + std::to_string(cholesky_cpu);

#ifdef ENABLE_VALIDATION
                    double residual = cpu::cholesky_residual(size, n_tiles, tiled_matrix);
                    report_residual("stf_sync", residual);
#endif
                }
                ///////////////////////////////////////////////////////////////////////////
                // print/write header only once
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

    runtime_file.close();

    starpu_shutdown();
    return 0;
}
