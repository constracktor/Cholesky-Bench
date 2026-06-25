// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CPU_ADAPTER_CBLAS_FP64_H
#define CPU_ADAPTER_CBLAS_FP64_H

#pragma once

#include <vector>

using vector = std::vector<double>;

// Constants that are compatible with CBLAS
typedef enum BLAS_TRANSPOSE { Blas_no_trans = 111, Blas_trans = 112 } BLAS_TRANSPOSE;

typedef enum BLAS_SIDE { Blas_left = 141, Blas_right = 142 } BLAS_SIDE;

typedef enum BLAS_ALPHA { Blas_add = 1, Blas_substract = -1 } BLAS_ALPHA;

// BLAS level 3 operations

/**
 * @brief FP64 In-place Cholesky decomposition of A
 * @param A matrix to be factorized
 * @param N matrix dimension
 */
void potrf(vector &A, const int N);

/**
 * @brief FP64 In-place solve L(^T) * X = A or X * L(^T) = A where L lower triangular
 * @param L Cholesky factor matrix
 * @param A right hand side matrix
 * @param N first dimension
 * @param M second dimension
 */
void trsm(
    const vector &L, vector &A, const int N, const int M, const BLAS_TRANSPOSE transpose_L, const BLAS_SIDE side_L);

/**
 * @brief FP64 Symmetric rank-k update: A = A - B * B^T
 * @param A Base matrix
 * @param B Symmetric update matrix
 * @param N matrix dimension
 */
void syrk(vector &A, const vector &B, const int N);

/**
 * @brief FP64 General matrix-matrix multiplication: C = C - A(^T) * B(^T)
 * @param C Base matrix
 * @param B Right update matrix
 * @param A Left update matrix
 * @param N first matrix dimension
 * @param M second matrix dimension
 * @param K third matrix dimension
 * @param transpose_A transpose left matrix
 * @param transpose_B transpose right matrix
 */
void gemm(const vector &A,
          const vector &B,
          vector &C,
          const int N,
          const int M,
          const int K,
          const BLAS_TRANSPOSE transpose_A,
          const BLAS_TRANSPOSE transpose_B);

#endif  // end of CPU_ADAPTER_CBLAS_FP64_H
