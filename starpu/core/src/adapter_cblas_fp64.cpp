// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "adapter_cblas_fp64.hpp"

#ifndef DISABLE_COMPUTATION
#ifdef ENABLE_MKL
// MKL CBLAS and LAPACKE
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif
#endif  // !DISABLE_COMPUTATION

// BLAS level 3 operations

void potrf(double *A, const int N)
{
#ifndef DISABLE_COMPUTATION
    // POTRF: in-place Cholesky decomposition of A
    // use dpotrf2 recursive version for better stability
    lapack_int info = LAPACKE_dpotrf2(LAPACK_ROW_MAJOR, 'L', N, A, N);
    if (info != 0)
    {
        // clang-format off
        fprintf(stderr, "LAPACKE_dpotrf2 failed: info=%d (tile not positive definite)\n", (int) info);
        // clang-format on
    }
#else
    (void) A;
    (void) N;
#endif
}

void trsm(
    const double *L, double *A, const int N, const int M, const BLAS_TRANSPOSE transpose_L, const BLAS_SIDE side_L)

{
#ifndef DISABLE_COMPUTATION
    // TRSM constants
    const double alpha = 1.0;
    // TRSM: in-place solve L(^T) * X = A or X * L(^T) = A where L lower triangular
    cblas_dtrsm(
        CblasRowMajor,
        static_cast<CBLAS_SIDE>(side_L),
        CblasLower,
        static_cast<CBLAS_TRANSPOSE>(transpose_L),
        CblasNonUnit,
        N,
        M,
        alpha,
        L,
        N,
        A,
        M);
#else
    (void) L;
    (void) A;
    (void) N;
    (void) M;
    (void) transpose_L;
    (void) side_L;
#endif
}

void syrk(double *A, const double *B, const int N)
{
#ifndef DISABLE_COMPUTATION
    // SYRK constants
    const double alpha = -1.0;
    const double beta = 1.0;
    // SYRK:A = A - B * B^T
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, N, alpha, B, N, beta, A, N);
#else
    (void) A;
    (void) B;
    (void) N;
#endif
}

void gemm(const double *A,
          const double *B,
          double *C,
          const int N,
          const int M,
          const int K,
          const BLAS_TRANSPOSE transpose_A,
          const BLAS_TRANSPOSE transpose_B)
{
#ifndef DISABLE_COMPUTATION
    // GEMM constants
    const double alpha = -1.0;
    const double beta = 1.0;
    // GEMM: C = C - A(^T) * B(^T)
    cblas_dgemm(
        CblasRowMajor,
        static_cast<CBLAS_TRANSPOSE>(transpose_A),
        static_cast<CBLAS_TRANSPOSE>(transpose_B),
        K,
        M,
        N,
        alpha,
        A,
        K,
        B,
        M,
        beta,
        C,
        M);
#else
    (void) A;
    (void) B;
    (void) C;
    (void) N;
    (void) M;
    (void) K;
    (void) transpose_A;
    (void) transpose_B;
#endif
}
