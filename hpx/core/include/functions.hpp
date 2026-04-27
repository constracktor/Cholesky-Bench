#ifndef CPU_FUNCTIONS_H
#define CPU_FUNCTIONS_H

#pragma once

#include <hpx/future.hpp>
#include <string>
#include <vector>

using Tiled_vector_matrix = std::vector<std::vector<double>>;
using Tiled_future_matrix = std::vector<hpx::shared_future<std::vector<double>>>;
using Tiled_void_matrix   = std::vector<hpx::shared_future<void>>;

namespace cpu
{

double cholesky_future(Tiled_future_matrix &tiled_matrix, std::string variant);

double cholesky_loop(Tiled_vector_matrix &tiled_matrix, std::string variant);

/**
 * @brief Run the void-future Cholesky variant and return wall-clock time in seconds.
 *        Tile data lives in @p tiles (caller allocates via gen_void_tiled_matrix);
 *        @p dep_tiles carries only completion signals. Both are kept alive by the
 *        caller so the factorization can be validated afterwards.
 * @param tiles     flat lower-triangular tile data (mutated in-place)
 * @param dep_tiles matching void futures (updated to track each tile's latest operation)
 * @param n_tiles   tiles per dimension
 * @return elapsed time in seconds
 */
double cholesky_void(Tiled_vector_matrix &tiles, Tiled_void_matrix &dep_tiles, std::size_t n_tiles);

}  // namespace cpu
#endif  // end of CPU_FUNCTIONS_H
