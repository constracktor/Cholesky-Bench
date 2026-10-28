// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef TILE_GENERATION_H
#define TILE_GENERATION_H

#pragma once

#include <hpx/future.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <vector>

using Tiled_vector_matrix = std::vector<std::vector<double>>;
using Tiled_future_matrix = std::vector<hpx::shared_future<std::vector<double>>>;
using Tiled_void_matrix = std::vector<hpx::shared_future<void>>;

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
 * @brief Worker thread that owns tile (row, col): rows are distributed cyclically over the NUMA domains and
 *        the columns over the threads within a domain. Consecutive worker threads are assumed to share a NUMA
 *        domain (run with --hpx:bind=compact).
 * @param row        row index of tile in tiled matrix
 * @param col        col index of tile in tiled matrix
 * @return Worker thread index
 */
std::size_t tile_owner(std::size_t row, std::size_t col);

/**
 * @brief Launch policy that schedules a task on the owner of tile (row, col)
 * @param row        row index of the tile the task writes
 * @param col        col index of the tile the task writes
 * @param priority   scheduling priority of the task
 */
hpx::launch::async_policy tile_policy(
    std::size_t row, std::size_t col, hpx::threads::thread_priority priority = hpx::threads::thread_priority::normal);

/**
 * @brief Generate a Tiled_vector_matrix of filled vectors
 * @param problem_size total dimension of the matrix
 * @param n_tiles    tiles per dimension
 * @param owner_placed generate each tile on its owner thread (first touch places its pages in the owner's NUMA domain)
 * @return Tiled matrix where lower triangle populated with filled vector
 */
Tiled_vector_matrix gen_tiled_matrix(std::size_t problem_size, std::size_t n_tiles, bool owner_placed = false);

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
