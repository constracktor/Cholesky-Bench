// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "cholesky_factor.hpp"

#include "adapter_cblas_fp64.hpp"
#include "tile_generation.hpp"
#include <cmath>
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/functional.hpp>
#include <hpx/future.hpp>
#include <iostream>

namespace cpu
{

// Robust integer square root for exact squares
inline std::size_t isqrt_exact(std::size_t x)
{
    return static_cast<std::size_t>(std::lround(std::sqrt(static_cast<double>(x))));
}

// Submit one task on the owner of the tile it writes: dataflow on annotated function. The critical-path kernels
// (POTRF, TRSM) run at high priority.
template <typename F, typename... Args>
static auto df(bool critical, std::size_t row, std::size_t col, F f, const char *name, Args &&...args)
{
    const auto priority = critical ? hpx::threads::thread_priority::high : hpx::threads::thread_priority::normal;
    return hpx::dataflow(
        tile_policy(row, col, priority), hpx::annotated_function(f, name), std::forward<Args>(args)...);
}

// Submission of independent tile tasks (distinct output tiles). With parallel == true the
// submission itself is spread over the workers; the tasks remain dependency-driven.
template <typename Fn>
static void submit_loop(bool parallel, std::size_t b, std::size_t e, Fn &&fn)
{
    if (parallel)
    {
        hpx::experimental::for_loop(
            hpx::execution::par.with(hpx::execution::experimental::dynamic_chunk_size(1)), b, e, fn);
    }
    else
    {
        for (std::size_t i = b; i < e; ++i)
        {
            fn(i);
        }
    }
}

// Tiled Cholesky Algorithm
void right_looking_cholesky_tiled(Variant variant, Tiled_future_matrix &ft_tiles)
{
    // Parameters
    const int N = static_cast<int>(isqrt_exact(ft_tiles[0].get().size()));
    const std::size_t n_tiles = isqrt_exact(ft_tiles.size());
    const bool sync = variant == Variant::sync_future;
    if (variant != Variant::async_future && !sync)
    {
        std::cout << "Variant not supported.";
        return;
    }
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        // POTRF: Compute Cholesky factor L
        ft_tiles[k * n_tiles + k] = df(true, k, k, f_potrf, "cholesky_potrf", ft_tiles[k * n_tiles + k], N);
        if (sync)
        {
            ft_tiles[k * n_tiles + k].get();
        }
        submit_loop(!sync,
                    k + 1,
                    n_tiles,
                    [&](std::size_t m)
                    {
                        // TRSM:  Solve X * L^T = A
                        ft_tiles[m * n_tiles + k] =
                            df(true,
                               m,
                               k,
                               f_trsm,
                               "cholesky_trsm",
                               ft_tiles[k * n_tiles + k],
                               ft_tiles[m * n_tiles + k],
                               N,
                               N,
                               Blas_trans,
                               Blas_right);
                    });
        if (sync)
        {
            for (std::size_t m = k + 1; m < n_tiles; m++)
            {
                ft_tiles[m * n_tiles + k].get();
            }
        }
        submit_loop(
            !sync,
            k + 1,
            n_tiles,
            [&](std::size_t m)
            {
                // SYRK:  A = A - B * B^T
                ft_tiles[m * n_tiles + m] =
                    df(false, m, m, f_syrk, "cholesky_syrk", ft_tiles[m * n_tiles + m], ft_tiles[m * n_tiles + k], N);
                for (std::size_t n = k + 1; n < m; n++)
                {
                    // GEMM: C = C - A * B^T
                    ft_tiles[m * n_tiles + n] =
                        df(false,
                           m,
                           n,
                           f_gemm,
                           "cholesky_gemm",
                           ft_tiles[m * n_tiles + k],
                           ft_tiles[n * n_tiles + k],
                           ft_tiles[m * n_tiles + n],
                           N,
                           N,
                           N,
                           Blas_no_trans,
                           Blas_trans);
                }
            });
        if (sync)
        {
            for (std::size_t m = k + 1; m < n_tiles; m++)
            {
                for (std::size_t n = k + 1; n <= m; n++)
                {
                    ft_tiles[m * n_tiles + n].get();
                }
            }
        }
    }
}

void right_looking_cholesky_tiled_loop(Variant variant, Tiled_vector_matrix &tiles)
{
    // Parameters
    const int N = static_cast<int>(isqrt_exact(tiles[0].size()));
    const std::size_t n_tiles = isqrt_exact(tiles.size());
    // The trailing-update outer loop has triangular work; this parallel
    // execution policy adds dynamic scheduling for load balance.
    auto par_dyn = hpx::execution::par.with(hpx::execution::experimental::dynamic_chunk_size(1));

    // Variants
    switch (variant)
    {
        case Variant::loop_one:
            for (std::size_t k = 0; k < n_tiles; k++)
            {
                // POTRF: Compute Cholesky factor L
                potrf(tiles[k * n_tiles + k], N);

                hpx::experimental::for_loop(
                    hpx::execution::par,
                    k + 1,
                    n_tiles,
                    [&](std::size_t m)
                    {
                        // TRSM:  Solve X * L^T = A
                        trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                    });

                hpx::experimental::for_loop(
                    par_dyn,
                    k + 1,
                    n_tiles,
                    [&](std::size_t m)
                    {
                        hpx::experimental::for_loop(
                            hpx::execution::seq,
                            k + 1,
                            m + 1,
                            [&](std::size_t n)
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
                            });
                    });
            }
            break;
        case Variant::loop_two:
            for (std::size_t k = 0; k < n_tiles; k++)
            {
                // POTRF: Compute Cholesky factor L
                potrf(tiles[k * n_tiles + k], N);

                hpx::experimental::for_loop(
                    hpx::execution::par,
                    k + 1,
                    n_tiles,
                    [&](std::size_t m)
                    {
                        // TRSM:  Solve X * L^T = A
                        trsm(tiles[k * n_tiles + k], tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
                    });

                // Collapsed triangular (m, n) iteration space, scheduled dynamically
                const std::size_t r = n_tiles - k - 1;
                const std::size_t P = r * (r + 1) / 2;
                hpx::experimental::for_loop(
                    par_dyn,
                    std::size_t(0),
                    P,
                    [&](std::size_t p)
                    {
                        std::size_t t = static_cast<std::size_t>((std::sqrt(8.0 * double(p) + 1.0) - 1.0) / 2.0);
                        while (t * (t + 1) / 2 > p)
                        {
                            --t;
                        }
                        while ((t + 1) * (t + 2) / 2 <= p)
                        {
                            ++t;
                        }
                        const std::size_t m = k + 1 + t;
                        const std::size_t n = k + 1 + (p - t * (t + 1) / 2);
                        if (n == m)
                        {
                            syrk(tiles[m * n_tiles + m], tiles[m * n_tiles + k], N);
                        }
                        else
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
                    });
            }
            break;
        default:
            // clang-format off
            std::cout << "Variant not supported."; break;
            // clang-format on
    }
}

void right_looking_cholesky_tiled_void(Tiled_vector_matrix &tiles, Tiled_void_matrix &dep_tiles)
{
    // Tile parameters
    const int N = static_cast<int>(isqrt_exact(tiles[0].size()));
    const std::size_t n_tiles = isqrt_exact(tiles.size());

    for (std::size_t k = 0; k < n_tiles; k++)
    {
        // POTRF: Compute Cholesky factor L
        dep_tiles[k * n_tiles + k] =
            df(true, k, k, potrf_f, "cholesky_potrf", dep_tiles[k * n_tiles + k], std::ref(tiles[k * n_tiles + k]), N);

        submit_loop(false,
                    k + 1,
                    n_tiles,
                    [&](std::size_t m)
                    {
                        // TRSM: Solve X * L^T = A
                        dep_tiles[m * n_tiles + k] =
                            df(true,
                               m,
                               k,
                               trsm_f,
                               "cholesky_trsm",
                               dep_tiles[k * n_tiles + k],  // dep on L
                               dep_tiles[m * n_tiles + k],  // dep on A
                               std::ref(tiles[k * n_tiles + k]),
                               std::ref(tiles[m * n_tiles + k]),
                               N,
                               N,
                               Blas_trans,
                               Blas_right);
                    });

        submit_loop(false,
                    k + 1,
                    n_tiles,
                    [&](std::size_t m)
                    {
                        // SYRK: A = A - B * B^T
                        dep_tiles[m * n_tiles + m] =
                            df(false,
                               m,
                               m,
                               syrk_f,
                               "cholesky_syrk",
                               dep_tiles[m * n_tiles + m],  // dep on A
                               dep_tiles[m * n_tiles + k],  // dep on B
                               std::ref(tiles[m * n_tiles + m]),
                               std::cref(tiles[m * n_tiles + k]),
                               N);

                        for (std::size_t n = k + 1; n < m; n++)
                        {
                            // GEMM: C[m,n] = C[m,n] - A[m,k] * B[n,k]^T
                            dep_tiles[m * n_tiles + n] = df(
                                false,
                                m,
                                n,
                                gemm_f,
                                "cholesky_gemm",
                                dep_tiles[m * n_tiles + k],  // dep on A
                                dep_tiles[n * n_tiles + k],  // dep on B
                                dep_tiles[m * n_tiles + n],  // dep on C
                                std::cref(tiles[m * n_tiles + k]),
                                std::cref(tiles[n * n_tiles + k]),
                                std::ref(tiles[m * n_tiles + n]),
                                N,
                                N,
                                N,
                                Blas_no_trans,
                                Blas_trans);
                        }
                    });
    }
}

}  // end of namespace cpu
