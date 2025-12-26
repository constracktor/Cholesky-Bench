#pragma once
#include "mkl.h"
//#include "mkl_cblas.h"
//#include "mkl_lapacke.h"
// #include "cblas.h"
// #include "lapacke.h"

#include <vector>

// Constants that are compatible with CBLAS

typedef enum BLAS_TRANSPOSE { Blas_no_trans = 111, Blas_trans = 112 } BLAS_TRANSPOSE;

typedef enum BLAS_SIDE { Blas_left = 141, Blas_right = 142 } BLAS_SIDE;

typedef enum BLAS_ALPHA { Blas_add = 1, Blas_substract = -1 } BLAS_ALPHA;
////////////////////////////////////////////////////////////////////////////////
// BLAS operations in float precision
// in-place Cholesky decomposition of A -> return factorized matrix L
void mkl_potrf(std::vector<float>& A,
               std::size_t N)
{
  // use ?potrf2 recursive version for better stability
  // POTRF  
  LAPACKE_spotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);
}

// in-place solve L * X = A^T where L triangular
void mkl_trsm(std::vector<float>& L,
              std::vector<float>& A,
              std::size_t N)
{
  // TRSM constants
  const float alpha = 1.0f; 
  // TRSM kernel
  cblas_strsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, 
              N, N, alpha, L.data(), N, A.data(), N);
}

// A = A - B * B^T
void mkl_syrk(std::vector<float>& A,
              std::vector<float>& B,
              std::size_t N)
{
  // SYRK constants
  const float alpha = -1.0f;
  const float beta = 1.0f;
  // SYRK kernel
  cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans,
              N, N, alpha, B.data(), N, beta, A.data(), N);
}

//C = C - A * B^T
void mkl_gemm(std::vector<float>& A,
              std::vector<float>& B,
              std::vector<float>& C,
              std::size_t N)
{
  // GEMM constants
  const float alpha = -1.0f;
  const float beta = 1.0f;
  // GEMM kernel
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              N, N, N, alpha, A.data(), N, B.data(), N, beta, C.data(), N);
}

////////////////////////////////////////////////////////////////////////////////
// BLAS operations in double precision
// in-place Cholesky decomposition of A -> return factorized matrix L
void mkl_potrf(std::vector<double>& A,
               std::size_t N)
{
  // use ?potrf2 recursive version for better stability
  // POTRF  
  LAPACKE_dpotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);
}

// in-place solve L * X = A^T where L triangular
void mkl_trsm(std::vector<double>& L,
              std::vector<double>& A,
              std::size_t N)
{
  // TRSM constants
  const double alpha = 1.0f; 
  // TRSM kernel
  cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, 
              N, N, alpha, L.data(), N, A.data(), N);
}

// A = A - B * B^T
void mkl_syrk(std::vector<double>& A,
              std::vector<double>& B,
              std::size_t N)
{
  // SYRK constants
  const double alpha = -1.0f;
  const double beta = 1.0f;
  // SYRK kernel
  cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
              N, N, alpha, B.data(), N, beta, A.data(), N);
}

//C = C - A * B^T
void mkl_gemm(std::vector<double>& A,
              std::vector<double>& B,
              std::vector<double>& C,
              std::size_t N)
{
  // GEMM constants
  const double alpha = -1.0f;
  const double beta = 1.0f;
  // GEMM kernel
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              N, N, N, alpha, A.data(), N, B.data(), N, beta, C.data(), N);
}

//////////////////////////////////////////////////////////

void potrf(std::vector<double>& A,
               const int N)
{
    // POTRF: in-place Cholesky decomposition of A
    // use dpotrf2 recursive version for better stability
    LAPACKE_dpotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);
}

void trsm(std::vector<double>& L,
              std::vector<double>& A,
                   const int N,
                   const int M,
                   const BLAS_TRANSPOSE transpose_L,
                   const BLAS_SIDE side_L)

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

void syrk(std::vector<double>& A,
              std::vector<double>& B,
              const int N)
{
    // SYRK constants
    const double alpha = -1.0;
    const double beta = 1.0;
    // SYRK:A = A - B * B^T
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, N, alpha, B.data(), N, beta, A.data(), N);
}

void gemm(std::vector<double>& A,
              std::vector<double>& B,
              std::vector<double>& C,
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
