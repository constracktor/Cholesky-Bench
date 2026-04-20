#ifndef TILE_GENERATION_H
#define TILE_GENERATION_H

#pragma once

#include <hpx/future.hpp>
#include <vector>

using Tiled_vector_matrix = std::vector<std::vector<double>>;
using Tiled_future_matrix = std::vector<hpx::shared_future<std::vector<double>>>;
using Tiled_void_matrix   = std::vector<hpx::shared_future<void>>;

std::vector<double> gen_tile(std::size_t row, std::size_t col, std::size_t N, std::size_t n_tiles);

Tiled_vector_matrix gen_tiled_matrix(std::size_t problem_size, std::size_t n_tiles);

Tiled_future_matrix gen_futurized_tiled_matrix(std::size_t problem_size, std::size_t n_tiles);

/**
 * @brief Generate a tiled matrix as a flat vector of tiles (data) plus a matching
 *        Tiled_void_matrix of ready void futures (one per tile slot).
 *        The void futures serve as dependency tokens for the void-future Cholesky variant.
 * @param tiles      output: flat row-major lower-triangular tile data
 * @param dep_tiles  output: matching void futures (all ready after return)
 * @param problem_size total matrix dimension
 * @param n_tiles    tiles per dimension
 */
void gen_void_tiled_matrix(Tiled_vector_matrix &tiles,
                           Tiled_void_matrix &dep_tiles,
                           std::size_t problem_size,
                           std::size_t n_tiles);
#endif
