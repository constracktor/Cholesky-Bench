// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "cholesky_factor.hpp"

#include "adapter_cblas_fp64.hpp"
#include "tile_generation.hpp"
#include <atomic>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <omp.h>
#include <sched.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

// Trailing-update schedule:
// libgomp through GCC 14.x rejects schedule(dynamic, ...) on a
// non-rectangular collapsed loop nest.
#ifdef ENABLE_DYNAMIC_SCHEDULE
#define CHOL_OMP_FOR_TRAILING_COLLAPSE() _Pragma("omp for collapse(2) schedule(dynamic, 1)")
#else
#define CHOL_OMP_FOR_TRAILING_COLLAPSE() _Pragma("omp for collapse(2)")
#endif

namespace
{
// Portable spin-wait hint
inline void cpu_relax()
{
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
}

// Spin until flag is set; yield the CPU periodically so oversubscribed runs (threads > cores) make progress
inline void spin_wait(const std::atomic<char> &f)
{
    constexpr unsigned spins_before_yield = 4096;
    unsigned spins = 0;
    while (!f.load(std::memory_order_acquire))
    {
        cpu_relax();
        if (++spins >= spins_before_yield)
        {
            spins = 0;
            sched_yield();
        }
    }
}

// Robust integer square root for exact squares
inline std::size_t isqrt_exact(std::size_t x)
{
    return static_cast<std::size_t>(std::lround(std::sqrt(static_cast<double>(x))));
}

// Per-tile readiness flags, all cleared
inline std::unique_ptr<std::atomic<char>[]> make_flags(std::size_t n)
{
    std::unique_ptr<std::atomic<char>[]> flag(new std::atomic<char>[n]);
    for (std::size_t i = 0; i < n; ++i)
    {
        flag[i].store(0, std::memory_order_relaxed);
    }
    return flag;
}
}  // namespace

namespace cpu
{

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
                    // Rows are handed out from the bottom: the longest rows (most GEMMs) start first
#pragma omp for schedule(dynamic, 1)
                    for (std::size_t mi = k + 1; mi < n_tiles; ++mi)
                    {
                        const std::size_t m = n_tiles - 1 - (mi - k - 1);
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
            // Tasks are generated in parallel by all threads (worksharing loops over rows with nowait),
            // instead of by a single thread. The barriers replace the taskwaits and also execute pending tasks.
#pragma omp parallel
            {
                for (std::size_t k = 0; k < n_tiles; ++k)
                {
                    // POTRF: Compute Cholesky factor L
#pragma omp single
                    {
                        potrf(tiles[k * n_tiles + k], N);
                    }

#pragma omp for schedule(static) nowait
                    for (std::size_t m = k + 1; m < n_tiles; ++m)
                    {
                        // TRSM:  Solve X * L^T = A
#pragma omp task firstprivate(k, m)
                        {
                            trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                        }
                    }
#pragma omp barrier

#pragma omp for schedule(dynamic, 1) nowait
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
#pragma omp barrier
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
        case Variant::task_domain:
            {
                // One team per NUMA domain (nested parallel regions). Each team creates and runs the tasks of
                // the tiles of its domain (row % D == d). Local dependencies use depend clauses, panel tiles of other
                // domains are awaited through atomic flags before the dependent task is created.
                auto flag = make_flags(n_tiles * n_tiles);
                auto wait = [&](std::size_t idx)
                {
                    // Safe because numa_domains() guarantees >= 2 threads per team: another thread of the team
                    // keeps executing the tasks (taskyield is not reliable in libgomp).
                    spin_wait(flag[idx]);
                };
                const int nt = omp_get_max_threads();
                const int D = static_cast<int>(numa_domains(nt));
                const int per = nt / D;
                omp_set_max_active_levels(2);
#pragma omp parallel num_threads(D)
                {
                    const std::size_t d = omp_get_thread_num();
#pragma omp parallel num_threads(per)
#pragma omp single
                    {
                        for (std::size_t k = 0; k < n_tiles; ++k)
                        {
                            std::vector<double> &tile_kk = tiles[k * n_tiles + k];
                            const int hi = static_cast<int>(n_tiles - k);
                            if (k % D == d)
                            {
#pragma omp task depend(inout : tile_kk) priority(hi)
                                {
                                    potrf(tiles[k * n_tiles + k], N);
                                    flag[k * n_tiles + k].store(1, std::memory_order_release);
                                }
                            }
                            else
                            {
                                wait(k * n_tiles + k);
                            }
                            for (std::size_t m = k + 1; m < n_tiles; ++m)
                            {
                                if (m % D != d)
                                {
                                    continue;
                                }
                                std::vector<double> &tile_mk = tiles[m * n_tiles + k];
#pragma omp task depend(in : tile_kk) depend(inout : tile_mk) priority(hi - 1)
                                {
                                    trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                                    flag[m * n_tiles + k].store(1, std::memory_order_release);
                                }
                            }
                            for (std::size_t m = k + 1; m < n_tiles; ++m)
                            {
                                if (m % D != d)
                                {
                                    continue;
                                }
                                const std::vector<double> &tile_mk = tiles[m * n_tiles + k];
                                std::vector<double> &tile_mm = tiles[m * n_tiles + m];
                                const int syrk_prio = (m == k + 1) ? hi - 1 : 1;
#pragma omp task depend(in : tile_mk) depend(inout : tile_mm) priority(syrk_prio)
                                {
                                    syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                                }
                                for (std::size_t n = k + 1; n < m; ++n)
                                {
                                    if (n % D != d)
                                    {
                                        wait(n * n_tiles + k);
                                    }
                                    const std::vector<double> &tile_nk = tiles[n * n_tiles + k];
                                    std::vector<double> &tile_mn = tiles[m * n_tiles + n];
                                    const int gemm_prio = (n == k + 1) ? hi - 1 : 0;
#pragma omp task depend(in : tile_mk, tile_nk) depend(inout : tile_mn) priority(gemm_prio)
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
            }
            break;
        default:
            // clang-format off
            std::cout << "Variant not supported."; break;
            // clang-format on
    }
}

}  // end of namespace cpu
