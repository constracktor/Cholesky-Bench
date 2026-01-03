#ifndef TILE_GENERATION_H
#define TILE_GENERATION_H

#pragma once

#include "tile_data.hpp"
#include <hpx/future.hpp>
#include <vector>

using Tiled_vector_matrix = std::vector<std::vector<double>>;
using Tiled_future_matrix = std::vector<hpx::shared_future<std::vector<double>>>;
using Tiled_mutable_matrix = std::vector<hpx::shared_future<mutable_tile_data<double>>>;

std::vector<double> gen_tile(std::size_t row, std::size_t col, std::size_t N, std::size_t n_tiles);

mutable_tile_data<double> gen_mutable_tile(std::size_t row, std::size_t col, std::size_t N, std::size_t n_tiles);

Tiled_vector_matrix gen_tiled_matrix(std::size_t problem_size, std::size_t n_tiles);

Tiled_future_matrix gen_futurized_tiled_matrix(std::size_t problem_size, std::size_t n_tiles);

Tiled_mutable_matrix gen_mutable_tiled_matrix(std::size_t problem_size, std::size_t n_tiles);
#endif
