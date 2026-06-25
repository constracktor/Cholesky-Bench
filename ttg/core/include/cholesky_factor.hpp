// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CPU_CHOLESKY_FACTOR_H
#define CPU_CHOLESKY_FACTOR_H

#pragma once

#include <cstddef>
#include <vector>

using Tiled_vector_matrix = std::vector<std::vector<double>>;

namespace cpu
{

/**
 * @brief Right-looking tiled Cholesky expressed as a TTG task graph.
 *
 * The flat lower-triangular tile data in @p tiles is streamed into a dataflow
 * graph (dispatcher → POTRF → TRSM → SYRK/GEMM), factorized fully
 * asynchronously across the TTG worker threads, and written back in place.
 *
 * Only the asynchronous execution (invoke → fence) is timed; the one-time graph
 * construction is excluded so the result is comparable with the OpenMP/HPX
 * variants and the upstream TTG potrf benchmark.
 *
 * @param tiles  flat lower-triangular tile data (mutated in-place)
 * @return elapsed factorization time in seconds
 */
double right_looking_cholesky_tiled_ttg(Tiled_vector_matrix &tiles);

}  // end of namespace cpu
#endif  // end of CPU_CHOLESKY_FACTOR_H
