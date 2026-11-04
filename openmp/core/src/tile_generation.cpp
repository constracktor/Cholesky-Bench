// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "tile_generation.hpp"

#ifndef DISABLE_COMPUTATION
#include <random>
#endif
#include <omp.h>

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

std::size_t tile_owner(std::size_t row, std::size_t col, std::size_t n_threads)
{
    // Assume threads are bound in order (OMP_PLACES=cores, OMP_PROC_BIND=close) and 8 NUMA domains
    const std::size_t D = (n_threads % 8 == 0 && n_threads >= 8) ? 8 : 1;
    const std::size_t per = n_threads / D;
    return (row % D) * per + (col + row / D) % per;
}

Tiled_vector_matrix gen_tiled_matrix(std::size_t problem_size, std::size_t n_tiles, bool owner_placed)
{
    // Tiled data structure
    Tiled_vector_matrix tiled_matrix;
    // Preallocate memory
    tiled_matrix.resize(n_tiles * n_tiles);  // No reserve because of triangular structure
    ///////////////////////////////////////////////////////////////////////////
    // Launch synchronous assembly
#ifdef DISABLE_COMPUTATION
    // No-op path: leave all inner vectors empty.
    (void) problem_size;
#else
    std::size_t tile_size = problem_size / n_tiles;
    if (owner_placed)
    {
        // First touch on the owning thread (same map as the owner variant)
#pragma omp parallel
        {
            const std::size_t nt = omp_get_num_threads();
            const std::size_t tid = omp_get_thread_num();
            for (std::size_t i = 0; i < n_tiles; ++i)
            {
                for (std::size_t j = 0; j < i + 1; ++j)
                {
                    if (tile_owner(i, j, nt) == tid)
                    {
                        tiled_matrix[i * n_tiles + j] = gen_tile(i, j, tile_size, n_tiles);
                    }
                }
            }
        }
        return tiled_matrix;
    }
#pragma omp parallel for collapse(2)
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j < i + 1; ++j)
        {
            tiled_matrix[i * n_tiles + j] = gen_tile(i, j, tile_size, n_tiles);
        }
    }
#endif

    return tiled_matrix;
}
