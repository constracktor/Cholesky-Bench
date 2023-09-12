//#include "mkl.h
#include "mkl_cblas.h"
#include "mkl_lapacke.h"

#include <vector>
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
  // return vector
  return A;
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