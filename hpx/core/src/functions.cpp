// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "functions.hpp"

#include "cholesky_factor.hpp"
#include "tile_generation.hpp"
#include <chrono>
#include <cmath>
#include <hpx/future.hpp>

namespace cpu
{

namespace
{
constexpr double nanoseconds_per_second = 1e9;

double elapsed_seconds(std::chrono::high_resolution_clock::time_point start)
{
    return (std::chrono::high_resolution_clock::now() - start).count() / nanoseconds_per_second;
}

// Wait on the lower-triangle futures only: the upper-triangle slots are
// default-constructed (invalid) and waiting on them is undefined.
template <typename Matrix>
void wait_lower_triangle(Matrix &tiles)
{
    const std::size_t n_tiles = static_cast<std::size_t>(std::sqrt(tiles.size()));
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            tiles[i * n_tiles + j].wait();
        }
    }
}
}  // namespace

double cholesky_future(Tiled_future_matrix &tiled_matrix, const std::string &variant)
{
    auto start = std::chrono::high_resolution_clock::now();
    ///////////////////////////////////////////////////////////////////////////
    // Launch Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(to_variant(variant), tiled_matrix);
    wait_lower_triangle(tiled_matrix);
    ///////////////////////////////////////////////////////////////////////////
    return elapsed_seconds(start);
}

double cholesky_loop(Tiled_vector_matrix &tiled_matrix, const std::string &variant)
{
    auto start = std::chrono::high_resolution_clock::now();
    ///////////////////////////////////////////////////////////////////////////
    // Launch Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled_loop(to_variant(variant), tiled_matrix);
    ///////////////////////////////////////////////////////////////////////////
    return elapsed_seconds(start);
}

double cholesky_void(Tiled_vector_matrix &tiled_matrix, Tiled_void_matrix &dep_tiles)
{
    auto start = std::chrono::high_resolution_clock::now();
    ///////////////////////////////////////////////////////////////////////////
    // Launch Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled_void(tiled_matrix, dep_tiles);
    wait_lower_triangle(dep_tiles);
    ///////////////////////////////////////////////////////////////////////////
    return elapsed_seconds(start);
}

}  // end of namespace cpu
