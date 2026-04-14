#ifndef TILE_GENERATION_H
#define TILE_GENERATION_H

#pragma once

#include <hpx/future.hpp>
#include <vector>

using Tiled_vector_matrix = std::vector<std::vector<double>>;
using Tiled_future_matrix = std::vector<hpx::shared_future<std::vector<double>>>;

std::vector<double> gen_tile(std::size_t row, std::size_t col, std::size_t N, std::size_t n_tiles);

Tiled_vector_matrix gen_tiled_matrix(std::size_t problem_size, std::size_t n_tiles);

Tiled_future_matrix gen_futurized_tiled_matrix(std::size_t problem_size, std::size_t n_tiles);
#endif
