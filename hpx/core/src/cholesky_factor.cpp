#include "cholesky_factor.hpp"

#include "adapter_cblas_fp64.hpp"
#include <hpx/algorithm.hpp>
#include <hpx/functional.hpp>
#include <hpx/future.hpp>

namespace cpu
{

// Tiled Cholesky Algorithm
void right_looking_cholesky_tiled(Variant variant, Tiled_future_matrix &ft_tiles)
{
    // Parameters
    int N = std::sqrt(ft_tiles[0].get().size());
    std::size_t n_tiles = std::sqrt(ft_tiles.size());
    // Variants
    switch (variant)
    {
            // Asynchronous variants
        case Variant::async_future:
            for (std::size_t k = 0; k < n_tiles; k++)
            {
                // POTRF: Compute Cholesky factor L
                ft_tiles[k * n_tiles + k] =
                    hpx::dataflow(hpx::annotated_function(f_potrf, "cholesky_potrf"), ft_tiles[k * n_tiles + k], N);
                for (std::size_t m = k + 1; m < n_tiles; m++)
                {
                    // TRSM:  Solve X * L^T = A
                    ft_tiles[m * n_tiles + k] = hpx::dataflow(
                        hpx::annotated_function(f_trsm, "cholesky_trsm"),
                        ft_tiles[k * n_tiles + k],
                        ft_tiles[m * n_tiles + k],
                        N,
                        N,
                        Blas_trans,
                        Blas_right);
                }
                for (std::size_t m = k + 1; m < n_tiles; m++)
                {
                    // SYRK:  A = A - B * B^T
                    ft_tiles[m * n_tiles + m] = hpx::dataflow(
                        hpx::annotated_function(f_syrk, "cholesky_syrk"),
                        ft_tiles[m * n_tiles + m],
                        ft_tiles[m * n_tiles + k],
                        N);
                    for (std::size_t n = k + 1; n < m; n++)
                    {
                        // GEMM: C = C - A * B^T
                        ft_tiles[m * n_tiles + n] = hpx::dataflow(
                            hpx::annotated_function(f_gemm, "cholesky_gemm"),
                            ft_tiles[m * n_tiles + k],
                            ft_tiles[n * n_tiles + k],
                            ft_tiles[m * n_tiles + n],
                            N,
                            N,
                            N,
                            Blas_no_trans,
                            Blas_trans);
                    }
                }
            }
            break;
            // Synchronous variants
        case Variant::sync_future:
            for (std::size_t k = 0; k < n_tiles; k++)
            {
                // POTRF: Compute Cholesky factor L
                ft_tiles[k * n_tiles + k] =
                    hpx::dataflow(hpx::annotated_function(f_potrf, "cholesky_potrf"), ft_tiles[k * n_tiles + k], N);
                // Synchronize
                ft_tiles[k * n_tiles + k].get();

                for (std::size_t m = k + 1; m < n_tiles; m++)
                {
                    // TRSM:  Solve X * L^T = A
                    ft_tiles[m * n_tiles + k] = hpx::dataflow(
                        hpx::annotated_function(f_trsm, "cholesky_trsm"),
                        ft_tiles[k * n_tiles + k],
                        ft_tiles[m * n_tiles + k],
                        N,
                        N,
                        Blas_trans,
                        Blas_right);
                }
                // Synchronize
                for (std::size_t m = k + 1; m < n_tiles; m++)
                {
                    ft_tiles[m * n_tiles + k].get();
                }

                for (std::size_t m = k + 1; m < n_tiles; m++)
                {
                    // SYRK:  A = A - B * B^T
                    ft_tiles[m * n_tiles + m] = hpx::dataflow(
                        hpx::annotated_function(f_syrk, "cholesky_syrk"),
                        ft_tiles[m * n_tiles + m],
                        ft_tiles[m * n_tiles + k],
                        N);
                    for (std::size_t n = k + 1; n < m; n++)
                    {
                        // GEMM: C = C - A * B^T
                        ft_tiles[m * n_tiles + n] = hpx::dataflow(
                            hpx::annotated_function(f_gemm, "cholesky_gemm"),
                            ft_tiles[m * n_tiles + k],
                            ft_tiles[n * n_tiles + k],
                            ft_tiles[m * n_tiles + n],
                            N,
                            N,
                            N,
                            Blas_no_trans,
                            Blas_trans);
                    }
                }
                // Synchronize
                for (std::size_t m = k + 1; m < n_tiles; m++)
                {
                    for (std::size_t n = k + 1; n <= m; n++)
                    {
                        ft_tiles[m * n_tiles + n].get();
                    }
                }
            }
            break;
        default: std::cout << "Variant not supported.\n"; break;
    }
}

void right_looking_cholesky_tiled_loop(Variant variant, Tiled_vector_matrix &tiles)
{
    // Parameters
    int N = std::sqrt(tiles[0].size());
    std::size_t n_tiles = std::sqrt(tiles.size());
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
                    hpx::execution::par,
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

                hpx::experimental::for_loop(
                    hpx::execution::par,
                    k + 1,
                    n_tiles,
                    [&](std::size_t m)
                    {
                        hpx::experimental::for_loop(
                            hpx::execution::par,
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
        default: std::cout << "Variant not supported.\n"; break;
    }
}

void right_looking_cholesky_tiled_void(Tiled_vector_matrix &tiles,
                                       Tiled_void_matrix &dep_tiles,
                                       std::size_t n_tiles)
{
    // Tile size derived from the first tile
    int N = static_cast<int>(std::sqrt(tiles[0].size()));

    for (std::size_t k = 0; k < n_tiles; k++)
    {
        // POTRF: Cholesky decomposition of diagonal tile [k,k]
        // Depends only on the previous operation that last wrote [k,k]
        dep_tiles[k * n_tiles + k] = hpx::dataflow(
            hpx::annotated_function(potrf_f, "cholesky_potrf"),
            dep_tiles[k * n_tiles + k],
            std::ref(tiles[k * n_tiles + k]),
            N);

        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // TRSM: Solve X * L^T = A for panel tile [m,k]
            // Depends on [k,k] being factored and [m,k] not being written by anyone else
            dep_tiles[m * n_tiles + k] = hpx::dataflow(
                hpx::annotated_function(trsm_f, "cholesky_trsm"),
                dep_tiles[k * n_tiles + k],  // dep on L
                dep_tiles[m * n_tiles + k],  // dep on A
                std::ref(tiles[k * n_tiles + k]),
                std::ref(tiles[m * n_tiles + k]),
                N,
                N,
                Blas_trans,
                Blas_right);
        }

        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // SYRK: A[m,m] = A[m,m] - B[m,k] * B[m,k]^T
            dep_tiles[m * n_tiles + m] = hpx::dataflow(
                hpx::annotated_function(syrk_f, "cholesky_syrk"),
                dep_tiles[m * n_tiles + m],  // dep on A
                dep_tiles[m * n_tiles + k],  // dep on B
                std::ref(tiles[m * n_tiles + m]),
                std::cref(tiles[m * n_tiles + k]),
                N);

            for (std::size_t n = k + 1; n < m; n++)
            {
                // GEMM: C[m,n] = C[m,n] - A[m,k] * B[n,k]^T
                dep_tiles[m * n_tiles + n] = hpx::dataflow(
                    hpx::annotated_function(gemm_f, "cholesky_gemm"),
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
        }
    }
}

}  // end of namespace cpu
