#include "cholesky_factor.hpp"

#include "adapter_cblas_fp64.hpp"
#include <cmath>
#include <cstddef>
#include <iostream>

namespace cpu
{

namespace
{
// Robust integer square root for exact squares; guards against std::sqrt
// returning something like 1.9999... on an exact square.
inline std::size_t isqrt_exact(std::size_t x)
{
    return static_cast<std::size_t>(std::lround(std::sqrt(static_cast<double>(x))));
}
}  // namespace

void right_looking_cholesky_tiled(Variant variant, Tiled_vector_matrix &tiles)
{
    // Parameters
    const int N = static_cast<int>(isqrt_exact(tiles[0].size()));
    const std::size_t n_tiles = isqrt_exact(tiles.size());
    // Variants
    switch (variant)
    {
        case Variant::for_collapse:
            // Single parallel region hoisted above the k-loop avoids
            // n_tiles fork/join cycles. Worksharing constructs inside
            // supply the needed barriers.
#pragma omp parallel
            {
                for (std::size_t k = 0; k < n_tiles; ++k)
                {
                    // POTRF: Compute Cholesky factor L (serial, one tile).
                    // The implicit barrier at the end of single publishes
                    // the updated tile to every thread before TRSM starts.
#pragma omp single
                    {
                        potrf(tiles[k * n_tiles + k], N);
                    }

                    // TRSM: Solve X * L^T = A (panel update).
#pragma omp for schedule(static)
                    for (std::size_t m = k + 1; m < n_tiles; ++m)
                    {
                        trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                    }

                    // Trailing-matrix update.
                    // dynamic,1 handles the SYRK (~N^3/3 FLOPs) vs GEMM
                    // (~2 N^3 FLOPs) cost asymmetry and the shrinking
                    // iteration count in late k.
#pragma omp for collapse(2) schedule(dynamic, 1)
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
                    // POTRF: Compute Cholesky factor L.
#pragma omp single
                    {
                        potrf(tiles[k * n_tiles + k], N);
                    }

                    // TRSM: Solve X * L^T = A.
#pragma omp for schedule(static)
                    for (std::size_t m = k + 1; m < n_tiles; ++m)
                    {
                        trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                    }

                    // Trailing-matrix update.
                    // Work per m-iteration grows linearly with m
                    // (one SYRK + (m - k - 1) GEMMs), so dynamic,1 is a
                    // much better fit than static.
#pragma omp for schedule(dynamic, 1)
                    for (std::size_t m = k + 1; m < n_tiles; ++m)
                    {
                        // SYRK: A = A - B * B^T on the diagonal tile.
                        syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                        // GEMM: strictly below the diagonal, n < m.
                        // (Fix: original bound n < m + 1 double-updated
                        //  tile_mm after the SYRK above.)
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
                        // POTRF runs directly in the single region; wrapping
                        // it in a task would only add scheduler overhead.
                        potrf(tiles[k * n_tiles + k], N);

                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
#pragma omp task firstprivate(k, m)
                            {
                                // TRSM:  Solve X * L^T = A
                                trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                            }
                        }
#pragma omp taskwait
                        // Trailing matrix update
                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
#pragma omp task firstprivate(k, m)
                            {
                                // SYRK: A = A - B * B^T
                                syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                            }
                            // (Fix: upper bound was n <= m, which caused a
                            //  GEMM to double-update tile_mm after SYRK.)
                            for (std::size_t n = k + 1; n < m; ++n)
                            {
#pragma omp task firstprivate(k, m, n)
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

                        // POTRF is on the critical path; give it the
                        // highest remaining priority so the runtime
                        // schedules it (and its successors) first.
                        // Requires OMP_MAX_TASK_PRIORITY > 0 at runtime.
                        const int potrf_prio = static_cast<int>(n_tiles - k);
#pragma omp task depend(inout : tile_kk) priority(potrf_prio)
                        {
                            // POTRF: Compute Cholesky factor L
                            potrf(tiles[k * n_tiles + k], N);
                        }

                        const int trsm_prio = static_cast<int>(n_tiles - k) - 1;
                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
                            auto &tile_mk = tiles[m * n_tiles + k];

#pragma omp task depend(in : tile_kk) depend(inout : tile_mk) priority(trsm_prio)
                            {
                                // TRSM:  Solve X * L^T = A
                                trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                            }
                        }

                        for (std::size_t m = k + 1; m < n_tiles; ++m)
                        {
                            auto &tile_mk = tiles[m * n_tiles + k];
                            auto &tile_mm = tiles[m * n_tiles + m];

                            // The SYRK for m == k+1 produces the next
                            // panel's diagonal tile, which is the input
                            // to the next POTRF -> highest priority among
                            // trailing updates. Other SYRKs are off-path.
                            const int syrk_prio = (m == k + 1) ? static_cast<int>(n_tiles - k) - 1 : 1;
#pragma omp task depend(in : tile_mk) depend(inout : tile_mm) priority(syrk_prio)
                            {
                                // SYRK: A = A - B * B^T
                                syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                            }

                            for (std::size_t n = k + 1; n < m; ++n)
                            {
                                auto &tile_nk = tiles[n * n_tiles + k];
                                auto &tile_mn = tiles[m * n_tiles + n];

                                // GEMMs dominate the FLOP count but are
                                // off the critical path. untied lets a
                                // thread suspend and resume elsewhere;
                                // priority 0 keeps them behind POTRF/TRSM.
#pragma omp task depend(in : tile_mk, tile_nk) depend(inout : tile_mn) untied priority(0)
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
                    // The implicit barrier at the end of the parallel
                    // region waits for all descendant tasks; the earlier
                    // explicit taskwait here was redundant.
                }
            }
            break;
        default: std::cout << "Variant not supported.\n"; break;
    }
}

}  // end of namespace cpu
