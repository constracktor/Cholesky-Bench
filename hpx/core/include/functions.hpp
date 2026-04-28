#ifndef CPU_FUNCTIONS_H
#define CPU_FUNCTIONS_H

#pragma once

#include <hpx/future.hpp>
#include <string>
#include <vector>

using Tiled_void_matrix = std::vector<hpx::shared_future<void>>;
using Tiled_vector_matrix = std::vector<std::vector<double>>;
using Tiled_future_matrix = std::vector<hpx::shared_future<std::vector<double>>>;

namespace cpu
{

/**
 * @brief Run the vector-future Cholesky variants and return wall-clock time in seconds.
 * @param tiled_matrix     futurized flat lower-triangular tile data (mutated in-place)
 * @param variant          task async (async_future) and task sync (sync_future)
 * @return elapsed time in seconds
 */
double cholesky_future(Tiled_future_matrix &tiled_matrix, std::string variant);

/**
 * @brief Run the fork-join Cholesky variants and return wall-clock time in seconds.
 * @param tiled_matrix     flat lower-triangular tile data (mutated in-place)
 * @param variant          fork-join (loop_one) and fork-join collapsed (loop_two)
 * @return elapsed time in seconds
 */
double cholesky_loop(Tiled_vector_matrix &tiled_matrix, std::string variant);

/**
 * @brief Run the void-future Cholesky variant and return wall-clock time in seconds.
 * @param tiled_matrix     flat lower-triangular tile data (mutated in-place)
 * @param dep_tiles        matching void futures
 * @return elapsed time in seconds
 */
double cholesky_void(Tiled_vector_matrix &tiled_matrix, Tiled_void_matrix &dep_tiles);

}  // namespace cpu
#endif  // end of CPU_FUNCTIONS_H
