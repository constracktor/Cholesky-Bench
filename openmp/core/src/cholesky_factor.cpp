#include "cholesky_factor.hpp"

#include "adapter_cblas_fp64.hpp"
#include <cmath>
#include <iostream>

namespace cpu
{

void right_looking_cholesky_tiled(Variant variant, Tiled_vector_matrix &tiles)
{
    // Parameters
    int N = std::sqrt(tiles[0].size());
    std::size_t n_tiles = std::sqrt(tiles.size());
    // Variants
    switch (variant)
    {
        case Variant::for_collapse:
            for (std::size_t k = 0; k < n_tiles; ++k)
            {
                // POTRF: Compute Cholesky factor L
                potrf(tiles[k * n_tiles + k], N);

                // TRSM over the panel below k
#pragma omp parallel for schedule(static)
                for (std::size_t m = k + 1; m < n_tiles; ++m)
                {
                    trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                }

                // Trailing matrix update
#pragma omp parallel for collapse(2)
                for (std::size_t m = k + 1; m < n_tiles; ++m)
                {
                    for (std::size_t n = k + 1; n < m + 1; ++n)
                    {
                        if (n == m)
                        {
                            // SYRK: A = A - B * B^T
                            syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                        }
                        else
                        {
                            // GEMM: C = C - A * B^T
                            gemm(tiles[m * n_tiles + k],
                                 tiles[n * n_tiles + k],
                                 tiles[m * n_tiles + n],
                                 N,
                                 N,
                                 N,
                                 Blas_no_trans,
                                 Blas_trans);
                        }
                    }
                }
            }
            break;
        case Variant::for_split:
            for (std::size_t k = 0; k < n_tiles; ++k)
            {
                // POTRF: Compute Cholesky factor L
                potrf(tiles[k * n_tiles + k], N);

// TRSM over the panel below k
#pragma omp parallel for schedule(static)
                for (std::size_t m = k + 1; m < n_tiles; ++m)
                {
                    trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                }

                // Trailing matrix update
                // Diagonal updates
#pragma omp parallel for schedule(static)
                for (std::size_t m = k + 1; m < n_tiles; ++m)
                {
                    syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                }

// Off-diagonal updates
#pragma omp parallel for schedule(static)
                for (std::size_t m = k + 1; m < n_tiles; ++m)
                {
                    for (std::size_t n = k + 1; n < m; ++n)
                    {
                        gemm(tiles[m * n_tiles + k],
                             tiles[n * n_tiles + k],
                             tiles[m * n_tiles + n],
                             N,
                             N,
                             N,
                             Blas_no_trans,
                             Blas_trans);
                    }
                }
            }
            break;
        case Variant::for_naive:
            for (std::size_t k = 0; k < n_tiles; ++k)
            {
                // POTRF: Compute Cholesky factor L
                potrf(tiles[k * n_tiles + k], N);

// TRSM over the panel below k
#pragma omp parallel for schedule(static)
                for (std::size_t m = k + 1; m < n_tiles; ++m)
                {
                    trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                }

// Trailing matrix update
#pragma omp parallel for schedule(static)
                for (std::size_t m = k + 1; m < n_tiles; ++m)
                {
                    for (std::size_t n = k + 1; n < m + 1; ++n)
                    {
                        if (n == m)
                        {
                            // SYRK: A = A - B * B^T
                            syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                        }
                        else
                        {
                            // GEMM: C = C - A * B^T
                            gemm(tiles[m * n_tiles + k],
                                 tiles[n * n_tiles + k],
                                 tiles[m * n_tiles + n],
                                 N,
                                 N,
                                 N,
                                 Blas_no_trans,
                                 Blas_trans);
                        }
                    }
                }
            }
            break;
        default: std::cout << "Variant not supported.\n"; break;
    }
}

}  // end of namespace cpu
