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

namespace cpu
{

double cholesky_stf(Tiled_vector_matrix &tiled_matrix)
{
    const std::size_t n_tiles = static_cast<std::size_t>(std::sqrt(static_cast<double>(tiled_matrix.size())) + 0.5);
    Tile_handles handles(n_tiles * n_tiles, nullptr);
    register_tiles(tiled_matrix, handles);

    auto start = std::chrono::high_resolution_clock::now();
    ///////////////////////////////////////////////////////////////////////////
    // Launch Cholesky decomposition: K = L * L^T
    submit_right_looking_cholesky(handles, n_tiles);
    // Synchronize: wait for all submitted tasks
    starpu_task_wait_for_all();
    ///////////////////////////////////////////////////////////////////////////
    auto stop = std::chrono::high_resolution_clock::now();

    unregister_tiles(handles, n_tiles);
    return (stop - start).count() / 1e9;
}

}  // end of namespace cpu
