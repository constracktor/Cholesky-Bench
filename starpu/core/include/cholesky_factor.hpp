// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CPU_CHOLESKY_FACTOR_H
#define CPU_CHOLESKY_FACTOR_H

#pragma once

#include <cstddef>
#include <starpu.h>
#include <vector>

using Tiled_vector_matrix = std::vector<std::vector<double>>;
using Tile_handles = std::vector<starpu_data_handle_t>;

namespace cpu
{

/**
 * @brief Register the lower-triangle tiles with StarPU (no copy; handles wrap the existing tile memory).
 *        With DISABLE_COMPUTATION, void handles are registered instead since the tiles are empty.
 * @param tiles   flat lower-triangular tile data
 * @param handles output handles (n_tiles * n_tiles; upper triangle stays nullptr)
 */
void register_tiles(Tiled_vector_matrix &tiles, Tile_handles &handles);

/**
 * @brief Submit the right-looking tiled Cholesky as a StarPU sequential task flow.
 *        Dependencies are inferred by StarPU from the access modes; this call does not wait for completion.
 * @param handles tile handles from register_tiles
 * @param n_tiles tiles per dimension
 */
void submit_right_looking_cholesky(Tile_handles &handles, std::size_t n_tiles);

/**
 * @brief Unregister the lower-triangle tile handles (waits for pending tasks on them).
 * @param handles tile handles from register_tiles
 * @param n_tiles tiles per dimension
 */
void unregister_tiles(Tile_handles &handles, std::size_t n_tiles);

}  // end of namespace cpu
#endif  // end of CPU_CHOLESKY_FACTOR_H
