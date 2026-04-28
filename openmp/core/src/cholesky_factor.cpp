#include "cholesky_factor.hpp"

#include "adapter_cblas_fp64.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>

// Trailing-update schedule:
// libgomp through GCC 14.x rejects schedule(dynamic, ...) on a
// non-rectangular collapsed loop nest.
#ifdef ENABLE_DYNAMIC_SCHEDULE
#define CHOL_OMP_FOR_TRAILING_COLLAPSE() _Pragma("omp for collapse(2) schedule(dynamic, 1)")
#else
#define CHOL_OMP_FOR_TRAILING_COLLAPSE() _Pragma("omp for collapse(2)")
#endif

namespace cpu
{

// Robust integer square root for exact squares
inline std::size_t isqrt_exact(std::size_t x)
{
    return static_cast<std::size_t>(std::lround(std::sqrt(static_cast<double>(x))));
}

void right_looking_cholesky_tiled(Variant variant, Tiled_vector_matrix &tiles)
{
    // Parameters
    const int N = static_cast<int>(isqrt_exact(tiles[0].size()));
    const std::size_t n_tiles = isqrt_exact(tiles.size());
    // Variants
    switch (variant)
    {
        case Variant::for_collapse:
#pragma omp parallel
            {
                for (std::size_t k = 0; k < n_tiles; ++k)
                {
                    // POTRF: Compute Cholesky factor L
#pragma omp single
                    {
                        potrf(tiles[k * n_tiles + k], N);
                    }

                    // TRSM: Solve X * L^T = A
#pragma omp for schedule(static)
                    for (std::size_t m = k + 1; m < n_tiles; ++m)
                    {
                        trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                    }

                    CHOL_OMP_FOR_TRAILING_COLLAPSE()
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
            }
            break;
        case Variant::for_naive:
#pragma omp parallel
            {
                for (std::size_t k = 0; k < n_tiles; ++k)
                {
                    // POTRF: Compute Cholesky factor L
#pragma omp single
                    {
                        potrf(tiles[k * n_tiles + k], N);
                    }

                    // TRSM: Solve X * L^T = A
#pragma omp for schedule(static)
                    for (std::size_t m = k + 1; m < n_tiles; ++m)
                    {
                        trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                    }

                    // Dynamic scheduling due to triangular structure
#pragma omp for schedule(dynamic, 1)
                    for (std::size_t m = k + 1; m < n_tiles; ++m)
                    {
                        // SYRK: A = A - B * B^T
                        syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);

                        for (std::size_t n = k + 1; n < m; ++n)
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
                            // TRSM:  Solve X * L^T = A
#pragma omp task firstprivate(k, m)
                            {
                                trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                            }
                        }
#pragma omp taskwait

                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
                            // SYRK: A = A - B * B^T
#pragma omp task firstprivate(k, m)
                            {
                                syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                            }

                            for (std::size_t n = k + 1; n < m; ++n)
                            {
                                // GEMM: C = C - A * B^T
#pragma omp task firstprivate(k, m, n)
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
                        std::vector<double> &tile_kk = tiles[k * n_tiles + k];
                        // POTRF: Compute Cholesky factor L
#pragma omp task depend(inout : tile_kk)
                        {
                            potrf(tiles[k * n_tiles + k], N);
                        }

                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
                            std::vector<double> &tile_mk = tiles[m * n_tiles + k];
                            // TRSM:  Solve X * L^T = A
#pragma omp task depend(in : tile_kk) depend(inout : tile_mk)
                            {
                                trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                            }
                        }

                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
                            const std::vector<double> &tile_mk = tiles[m * n_tiles + k];
                            std::vector<double> &tile_mm = tiles[m * n_tiles + m];
                            // SYRK: A = A - B * B^T
#pragma omp task depend(in : tile_mk) depend(inout : tile_mm)
                            {
                                syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                            }

                            for (std::size_t n = k + 1; n < m; ++n)
                            {
                                const std::vector<double> &tile_nk = tiles[n * n_tiles + k];
                                std::vector<double> &tile_mn = tiles[m * n_tiles + n];
                                // GEMM: C = C - A * B^T
#pragma omp task depend(in : tile_mk, tile_nk) depend(inout : tile_mn)
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
                    }
                }
            }
            break;
        case Variant::task_prio:
#pragma omp parallel
            {
#pragma omp single
                {
                    for (std::size_t k = 0; k < n_tiles; ++k)
                    {
                        std::vector<double> &tile_kk = tiles[k * n_tiles + k];
                        const int potrf_prio = static_cast<int>(n_tiles - k);
                        // POTRF: Compute Cholesky factor L
#pragma omp task depend(inout : tile_kk) priority(potrf_prio)
                        {
                            potrf(tiles[k * n_tiles + k], N);
                        }

                        const int trsm_prio = static_cast<int>(n_tiles - k) - 1;
                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
                            std::vector<double> &tile_mk = tiles[m * n_tiles + k];
                            // TRSM:  Solve X * L^T = A
#pragma omp task depend(in : tile_kk) depend(inout : tile_mk) priority(trsm_prio)
                            {
                                trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                                ;
                            }
                        }

                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
                            const std::vector<double> &tile_mk = tiles[m * n_tiles + k];
                            std::vector<double> &tile_mm = tiles[m * n_tiles + m];
                            const int syrk_prio = (m == k + 1) ? static_cast<int>(n_tiles - k) - 1 : 1;
                            // SYRK: A = A - B * B^T
#pragma omp task depend(in : tile_mk) depend(inout : tile_mm) priority(syrk_prio)
                            {
                                syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                            }

                            for (std::size_t n = k + 1; n < m; ++n)
                            {
                                const std::vector<double> &tile_nk = tiles[n * n_tiles + k];
                                std::vector<double> &tile_mn = tiles[m * n_tiles + n];
                                // GEMM: C = C - A * B^T
#pragma omp task depend(in : tile_mk, tile_nk) depend(inout : tile_mn) untied priority(0)
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
                    }
                }
            }
            break;
        default: std::cout << "Variant not supported.\n"; break;
    }
}

}  // end of namespace cpu
