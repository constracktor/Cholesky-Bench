#ifndef TILE_GENERATION_H
#define TILE_GENERATION_H

#pragma once

#include <hpx/future.hpp>
#include <vector>

using Tiled_vector_matrix = std::vector<std::vector<double>>;
using Tiled_future_matrix = std::vector<hpx::shared_future<std::vector<double>>>;
using Tiled_void_matrix = std::vector<hpx::shared_future<void>>;

/**
 * @brief Generate a tile of a random symmetric positive-definite tiled matrix
 * @param row        row index of tile in tiled matrix
 * @param col        col index of tile in tiled matrix
 * @param N          total dimension of matrix (problem_size)
 * @param n_tiles    tiles per dimension
 * @return Tiled matrix where lower triangle populated with filled vector
 */
std::vector<double> gen_tile(std::size_t row, std::size_t col, std::size_t N, std::size_t n_tiles);

/**
 * @brief Generate a Tiled_vector_matrix of filled vectors
 * @param problem_size total dimension of the matrix
 * @param n_tiles    tiles per dimension
 * @return Tiled matrix where lower triangle populated with filled vector
 */
Tiled_vector_matrix gen_tiled_matrix(std::size_t problem_size, std::size_t n_tiles);

/**
 * @brief Generate a Tiled_future_matrix of ready and filled vector futures
 * @param problem_size total dimension of the matrix
 * @param n_tiles    tiles per dimension
 * @return Tiled matrix where lower triangle populated with ready vector futures
 */
Tiled_future_matrix gen_futurized_tiled_matrix(std::size_t problem_size, std::size_t n_tiles);

/**
 * @brief Generate a Tiled_void_matrix of ready void futures
 * @param n_tiles    tiles per dimension
 * @return Tiled matrix where lower triangle populated with ready void futures
 */
Tiled_void_matrix gen_void_tiled_matrix(std::size_t n_tiles);
#endif
