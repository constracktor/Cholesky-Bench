// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef TILE_GENERATION_H
#define TILE_GENERATION_H

#pragma once

#include <vector>

using Tiled_vector_matrix = std::vector<std::vector<double>>;

std::vector<double> gen_tile(std::size_t row, std::size_t col, std::size_t N, std::size_t n_tiles);

// placed: 0 = any thread, 1 = first touch by owner threads (flat team), 2 = same via nested domain teams
Tiled_vector_matrix gen_tiled_matrix(std::size_t problem_size, std::size_t n_tiles, int placed = 0);

// Number of NUMA domains (teams) for a given thread count
std::size_t numa_domains(std::size_t n_threads);

// Owning thread of tile (row, col): NUMA domain row % D, core within the domain by (col + row / D)
std::size_t tile_owner(std::size_t row, std::size_t col, std::size_t n_threads);
#endif
