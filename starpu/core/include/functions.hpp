// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CPU_FUNCTIONS_H
#define CPU_FUNCTIONS_H

#pragma once

#include <vector>

using Tiled_vector_matrix = std::vector<std::vector<double>>;

namespace cpu
{

/**
 * @brief Run the StarPU sequential-task-flow Cholesky variant and return wall-clock time in seconds.
 *        Tile registration and unregistration are excluded from the measured time.
 *        starpu_init() must have been called by the caller.
 * @param tiled_matrix     flat lower-triangular tile data (mutated in-place)
 * @param synchronize      barrier after each phase (POTRF, TRSM, SYRK/GEMM) of every step
 * @return elapsed time in seconds
 */
double cholesky_stf(Tiled_vector_matrix &tiled_matrix, bool synchronize = false);

}  // namespace cpu
#endif  // end of CPU_FUNCTIONS_H
