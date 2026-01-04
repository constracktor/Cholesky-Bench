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

#pragma omp parallel for schedule(static)
                for (std::size_t m = k + 1; m < n_tiles; ++m)
                {
                    // TRSM:  Solve X * L^T = A
                    trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                }

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
        case Variant::for_naive:
            for (std::size_t k = 0; k < n_tiles; ++k)
            {
                // POTRF: Compute Cholesky factor L
                potrf(tiles[k * n_tiles + k], N);

#pragma omp parallel for schedule(static)
                for (std::size_t m = k + 1; m < n_tiles; ++m)
                {
                    // TRSM:  Solve X * L^T = A
                    trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                }

#pragma omp parallel for schedule(static)
                for (std::size_t m = k + 1; m < n_tiles; ++m)
                {
                    // SYRK: A = A - B * B^T
                    syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                    for (std::size_t n = k + 1; n < m + 1; ++n)
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
            break;
        case Variant::task_naive:
#pragma omp parallel
            {
#pragma omp single
                {
                    for (std::size_t k = 0; k < n_tiles; ++k)
                    {
                        // POTRF: Compute Cholesky factor L
                        potrf(tiles[k * n_tiles + k], N);

                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
#pragma omp task firstprivate(m)
                            {
                                // TRSM:  Solve X * L^T = A
                                trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                            }
                        }
#pragma omp taskwait
                        // Trailing matrix update
                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
#pragma omp task firstprivate(m)
                            {
                                // SYRK: A = A - B * B^T
                                syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                            }
                            for (std::size_t n = k + 1; n <= m; ++n)
                            {
#pragma omp task firstprivate(m, n)
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
#pragma omp taskwait
                    }
                }
            }
            break;
        case Variant::task_depend:
#pragma omp parallel
            {
#pragma omp single
                {
                    for (std::size_t k = 0; k < n_tiles; ++k)
                    {
                        auto &tile_kk = tiles[k * n_tiles + k];
#pragma omp task depend(inout : tile_kk)
                        {
                            // POTRF: Compute Cholesky factor L
                            potrf(tiles[k * n_tiles + k], N);
                        }

                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
                            auto &tile_mk = tiles[m * n_tiles + k];

#pragma omp task depend(in : tile_kk) depend(inout : tile_mk)
                            {
                                // TRSM:  Solve X * L^T = A
                                trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                            }
                        }

                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
                            auto &tile_mk = tiles[m * n_tiles + k];
                            auto &tile_mm = tiles[m * n_tiles + m];

#pragma omp task depend(in : tile_mk) depend(inout : tile_mm)
                            {
                                // SYRK: A = A - B * B^T
                                syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                            }

                            for (std::size_t n = k + 1; n < m; ++n)
                            {
                                auto &tile_nk = tiles[n * n_tiles + k];
                                auto &tile_mn = tiles[m * n_tiles + n];

#pragma omp task depend(in : tile_mk, tile_nk) depend(inout : tile_mn)
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
#pragma omp taskwait
                }
            }
            break;
        default: std::cout << "Variant not supported.\n"; break;
    }
}

}  // end of namespace cpu
