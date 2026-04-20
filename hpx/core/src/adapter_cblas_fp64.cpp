#include "adapter_cblas_fp64.hpp"

#ifdef GPRAT_ENABLE_MKL
// MKL CBLAS and LAPACKE
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

// BLAS level 3 operations

vector f_potrf(vector_future f_A, const int N)
{
    vector A = f_A.get();
    // POTRF: in-place Cholesky decomposition of A
    // use dpotrf2 recursive version for better stability
    LAPACKE_dpotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);
    // return factorized matrix L
    return A;
}

vector f_trsm(vector_future f_L,
                     vector_future f_A,
                     const int N,
                     const int M,
                     const BLAS_TRANSPOSE transpose_L,
                     const BLAS_SIDE side_L)

{
    const vector& L = f_L.get();
    vector A = f_A.get();
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
        L.data(),
        N,
        A.data(),
        M);
    // return vector
    return A;
}

vector f_syrk(vector_future f_A, vector_future f_B, const int N)
{
    const vector& B = f_B.get();
    vector A = f_A.get();
    // SYRK constants
    const double alpha = -1.0;
    const double beta = 1.0;
    // SYRK:A = A - B * B^T
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, N, alpha, B.data(), N, beta, A.data(), N);
    // return updated matrix A
    return A;
}

vector
f_gemm(vector_future f_A,
       vector_future f_B,
       vector_future f_C,
       const int N,
       const int M,
       const int K,
       const BLAS_TRANSPOSE transpose_A,
       const BLAS_TRANSPOSE transpose_B)
{
    vector C = f_C.get();
    const vector& B = f_B.get();
    const vector& A = f_A.get();
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
        A.data(),
        K,
        B.data(),
        M,
        beta,
        C.data(),
        M);
    // return updated matrix C
    return C;
}

//////////////////////////////////////////////////////////

void potrf(vector &A, const int N)
{
    // POTRF: in-place Cholesky decomposition of A
    // use dpotrf2 recursive version for better stability
    LAPACKE_dpotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);
}

void trsm(
    const vector &L, vector &A, const int N, const int M, const BLAS_TRANSPOSE transpose_L, const BLAS_SIDE side_L)

{
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
        L.data(),
        N,
        A.data(),
        M);
}

void syrk(vector &A, const vector &B, const int N)
{
    // SYRK constants
    const double alpha = -1.0;
    const double beta = 1.0;
    // SYRK:A = A - B * B^T
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, N, alpha, B.data(), N, beta, A.data(), N);
}

void gemm(const vector &A,
          const vector &B,
          vector &C,
          const int N,
          const int M,
          const int K,
          const BLAS_TRANSPOSE transpose_A,
          const BLAS_TRANSPOSE transpose_B)
{
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
        A.data(),
        K,
        B.data(),
        M,
        beta,
        C.data(),
        M);
}

//////////////////////////////////////////////////////////
// Void-future variants: dependency futures are awaited by hpx::dataflow before
// the lambda is invoked, so by the time the body runs all deps are satisfied.
// The BLAS call operates directly on the vector& — no copy of tile data.

void_future potrf_f(void_future dep_future, vector &A, const int N)
{
    // dep_future already consumed by dataflow; just run in-place
    potrf(A, N);
    return hpx::make_ready_future();
}

void_future trsm_f(void_future dep_L,
                   void_future dep_A,
                   vector &L,
                   vector &A,
                   const int N,
                   const int M,
                   const BLAS_TRANSPOSE transpose_L,
                   const BLAS_SIDE side_L)
{
    // dep_L and dep_A already consumed by dataflow
    trsm(L, A, N, M, transpose_L, side_L);
    return hpx::make_ready_future();
}

void_future syrk_f(void_future dep_A, void_future dep_B, vector &A, const vector &B, const int N)
{
    // dep_A and dep_B already consumed by dataflow
    syrk(A, B, N);
    return hpx::make_ready_future();
}

void_future gemm_f(void_future dep_A,
                   void_future dep_B,
                   void_future dep_C,
                   const vector &A,
                   const vector &B,
                   vector &C,
                   const int N,
                   const int M,
                   const int K,
                   const BLAS_TRANSPOSE transpose_A,
                   const BLAS_TRANSPOSE transpose_B)
{
    // dep_A, dep_B, dep_C already consumed by dataflow
    gemm(A, B, C, N, M, K, transpose_A, transpose_B);
    return hpx::make_ready_future();
}
