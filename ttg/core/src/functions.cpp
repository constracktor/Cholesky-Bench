// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "functions.hpp"

#include "cholesky_factor.hpp"

namespace cpu
{

double cholesky_flow(Tiled_vector_matrix &tiled_matrix)
{
    ///////////////////////////////////////////////////////////////////////////
    // Launch Cholesky decomposition: K = L * L^T
    //
    // The wall-clock measurement lives inside right_looking_cholesky_tiled_ttg,
    // which times only the asynchronous execution (invoke -> fence) and excludes
    // the one-time TTG graph construction. This keeps the reported runtime
    // apples-to-apples with the OpenMP/HPX variants (which time the
    // factorization only) and with the upstream TTG potrf benchmark.
    ///////////////////////////////////////////////////////////////////////////
    return right_looking_cholesky_tiled_ttg(tiled_matrix);
}

}  // end of namespace cpu
