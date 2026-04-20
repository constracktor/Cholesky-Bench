#ifndef CPU_FUNCTIONS_H
#define CPU_FUNCTIONS_H

#pragma once

#include <hpx/future.hpp>
#include <string>
#include <vector>

using Tiled_vector_matrix = std::vector<std::vector<double>>;
using Tiled_future_matrix = std::vector<hpx::shared_future<std::vector<double>>>;

namespace cpu
{

double cholesky_future(Tiled_future_matrix &tiled_matrix, std::string variant);

double cholesky_loop(Tiled_vector_matrix &tiled_matrix, std::string variant);

/**
 * @brief Run the void-future Cholesky variant and return wall-clock time in seconds.
 *        Generates tile data internally; no shared_future<vector> copies involved.
 * @param problem_size total matrix dimension
 * @param n_tiles      tiles per dimension
 * @return elapsed time in seconds
 */
double cholesky_void(std::size_t problem_size, std::size_t n_tiles);

}  // namespace cpu
#endif  // end of CPU_FUNCTIONS_H
