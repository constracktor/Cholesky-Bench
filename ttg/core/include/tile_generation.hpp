// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef TILE_GENERATION_H
#define TILE_GENERATION_H

#pragma once

#include <cstddef>
#include <vector>

using Tiled_vector_matrix = std::vector<std::vector<double>>;

/**
 * @brief Generate a tile of a random symmetric positive-definite tiled matrix
 * @param row        row index of tile in tiled matrix
 * @param col        col index of tile in tiled matrix
 * @param N          tile size (number of rows/columns per tile)
 * @param n_tiles    tiles per dimension
 * @return Single tile as a row-major N×N vector
 */
std::vector<double> gen_tile(std::size_t row, std::size_t col, std::size_t N, std::size_t n_tiles);

/**
 * @brief Generate a Tiled_vector_matrix of filled vectors
 * @param problem_size total dimension of the matrix
 * @param n_tiles    tiles per dimension
 * @return Tiled matrix where lower triangle populated with filled vector
 */
Tiled_vector_matrix gen_tiled_matrix(std::size_t problem_size, std::size_t n_tiles);
#endif
