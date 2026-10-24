// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "tile_generation.hpp"

#ifndef DISABLE_COMPUTATION
#include <random>
#endif

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace
{
// Spread the pages of the tile storage round-robin over all NUMA nodes while
// the tiles are generated (the pages are first touched by the generating
// thread). Without this, every tile lives on the NUMA node of the generating
// thread and the memory-bound tasks of all other NUMA domains compete for one
// node's memory bandwidth. Uses the raw syscall, so no libnuma is required;
// failure (non-NUMA or restricted systems) is silently ignored.
struct InterleavePolicy
{
#if defined(__linux__) && defined(SYS_set_mempolicy)
    static constexpr int mpol_default = 0;
    constexpr static int mpol_interleave = 3;

    InterleavePolicy()
    {
        unsigned long mask[16];
        for (auto &m : mask)
        {
            m = ~0UL;
        }
        syscall(SYS_set_mempolicy, mpol_interleave, mask, 1024UL);
    }

    ~InterleavePolicy() { syscall(SYS_set_mempolicy, mpol_default, nullptr, 0UL); }
#endif
};
}  // namespace

std::vector<double> gen_tile(std::size_t row, std::size_t col, std::size_t N, std::size_t n_tiles)
{
#ifdef DISABLE_COMPUTATION
    // No-op path for task-overhead measurements: return an empty tile.
    (void) row;
    (void) col;
    (void) N;
    (void) n_tiles;
    return {};
#else
    std::size_t i_global, j_global;
    double random_value;
    // Create random generator
    size_t seed = row * n_tiles + col;
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribute(0, 1);
    // Preallocate required memory
    std::vector<double> tile;
    tile.resize(N * N);
    // Compute entries
    // Check for diagonal tile
    if (row == col)
    {
        for (std::size_t i = 0; i < N; i++)
        {
            i_global = N * row + i;
            for (std::size_t j = 0; j <= i; j++)
            {
                j_global = N * col + j;
                random_value = distribute(generator);

                if (i_global == j_global)
                {
                    random_value += N * n_tiles;
                }
                tile[i * N + j] = random_value;
                tile[j * N + i] = random_value;
            }
        }
    }
    else
    {
        for (std::size_t i = 0; i < N; i++)
        {
            for (std::size_t j = 0; j < N; j++)
            {
                random_value = distribute(generator);
                tile[i * N + j] = random_value;
            }
        }
    }
    return tile;
#endif
}

Tiled_vector_matrix gen_tiled_matrix(std::size_t problem_size, std::size_t n_tiles)
{
    // Tiled data structure
    InterleavePolicy numa_interleave;
    Tiled_vector_matrix tiled_matrix;
    // Preallocate memory
    tiled_matrix.resize(n_tiles * n_tiles);  // No reserve because of triangular structure

#ifdef DISABLE_COMPUTATION
    // No-op path: leave all inner vectors empty.
    (void) problem_size;
#else
    std::size_t tile_size = problem_size / n_tiles;
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            tiled_matrix[i * n_tiles + j] = gen_tile(i, j, tile_size, n_tiles);
        }
    }
#endif

    return tiled_matrix;
}
