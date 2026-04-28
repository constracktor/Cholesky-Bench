#ifndef CPU_ADAPTER_CBLAS_FP64_H
#define CPU_ADAPTER_CBLAS_FP64_H

#pragma once

#include <hpx/future.hpp>
#include <vector>

using void_future = hpx::shared_future<void>;
using vector_future = hpx::shared_future<std::vector<double>>;
using vector = std::vector<double>;

// Constants that are compatible with CBLAS
typedef enum BLAS_TRANSPOSE { Blas_no_trans = 111, Blas_trans = 112 } BLAS_TRANSPOSE;

typedef enum BLAS_SIDE { Blas_left = 141, Blas_right = 142 } BLAS_SIDE;

typedef enum BLAS_ALPHA { Blas_add = 1, Blas_substract = -1 } BLAS_ALPHA;

// BLAS level 3 operations

/**
 * @brief FP64 In-place Cholesky decomposition of A
 * @param f_A matrix to be factorized
 * @param N matrix dimension
 * @return factorized, lower triangular matrix L
 */
vector f_potrf(vector_future f_A, const int N);

/**
 * @brief FP64 In-place solve L(^T) * X = A or X * L(^T) = A where L lower triangular
 * @param f_L Cholesky factor matrix
 * @param f_A right hand side matrix
 * @param N first dimension
 * @param M second dimension
 * @return solution matrix X
 */
vector f_trsm(vector_future f_L,
              vector_future f_A,
              const int N,
              const int M,
              const BLAS_TRANSPOSE transpose_L,
              const BLAS_SIDE side_L);

/**
 * @brief FP64 Symmetric rank-k update: A = A - B * B^T
 * @param f_A Base matrix
 * @param f_B Symmetric update matrix
 * @param N matrix dimension
 * @return updated matrix A
 */
vector f_syrk(vector_future f_A, vector_future f_B, const int N);

/**
 * @brief FP64 General matrix-matrix multiplication: C = C - A(^T) * B(^T)
 * @param f_C Base matrix
 * @param f_B Right update matrix
 * @param f_A Left update matrix
 * @param N first matrix dimension
 * @param M second matrix dimension
 * @param K third matrix dimension
 * @param transpose_A transpose left matrix
 * @param transpose_B transpose right matrix
 * @return updated matrix X
 */
vector f_gemm(vector_future f_A,
              vector_future f_B,
              vector_future f_C,
              const int N,
              const int M,
              const int K,
              const BLAS_TRANSPOSE transpose_A,
              const BLAS_TRANSPOSE transpose_B);

///////////////////////////////////////////////////////////////////////////////////////

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
 * @param f_A Base matrix
 * @param f_B Symmetric update matrix
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

///////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief FP64 In-place Cholesky decomposition of A, synchronized via void futures
 * @param dep_future dependency future to wait on before executing
 * @param A matrix to be factorized (mutated in-place)
 * @param N matrix dimension
 * @return void future signalling completion
 */
void_future potrf_f(void_future dep_future, vector &A, const int N);

/**
 * @brief FP64 In-place solve L(^T) * X = A or X * L(^T) = A, synchronized via void futures
 * @param dep_L dependency future for L (must be ready before reading L)
 * @param dep_A dependency future for A (must be ready before writing A)
 * @param L Cholesky factor matrix
 * @param A right hand side matrix (mutated in-place)
 * @param N first dimension
 * @param M second dimension
 * @param transpose_L transpose flag for L
 * @param side_L side flag for L
 * @return void future signalling completion
 */
void_future trsm_f(void_future dep_L,
                   void_future dep_A,
                   vector &L,
                   vector &A,
                   const int N,
                   const int M,
                   const BLAS_TRANSPOSE transpose_L,
                   const BLAS_SIDE side_L);

/**
 * @brief FP64 Symmetric rank-k update: A = A - B * B^T, synchronized via void futures
 * @param dep_A dependency future for A
 * @param dep_B dependency future for B
 * @param A base matrix (mutated in-place)
 * @param B symmetric update matrix
 * @param N matrix dimension
 * @return void future signalling completion
 */
void_future syrk_f(void_future dep_A, void_future dep_B, vector &A, const vector &B, const int N);

/**
 * @brief FP64 General matrix-matrix multiplication: C = C - A(^T) * B(^T), synchronized via void futures
 * @param dep_A dependency future for A
 * @param dep_B dependency future for B
 * @param dep_C dependency future for C
 * @param A left update matrix
 * @param B right update matrix
 * @param C base matrix (mutated in-place)
 * @param N first matrix dimension
 * @param M second matrix dimension
 * @param K third matrix dimension
 * @param transpose_A transpose flag for A
 * @param transpose_B transpose flag for B
 * @return void future signalling completion
 */
void_future
gemm_f(void_future dep_A,
       void_future dep_B,
       void_future dep_C,
       const vector &A,
       const vector &B,
       vector &C,
       const int N,
       const int M,
       const int K,
       const BLAS_TRANSPOSE transpose_A,
       const BLAS_TRANSPOSE transpose_B);
#endif  // end of CPU_ADAPTER_CBLAS_FP64_H
