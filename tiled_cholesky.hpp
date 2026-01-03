#pragma once
#include <vector>
#include <random>

#include "mkl_adapter.hpp"
#include "omp.h"

void right_looking_cholesky_tiled(std::vector<std::vector<double>> &tiles, int N, std::size_t n_tiles)
{
    for (std::size_t k = 0; k < n_tiles; ++k)
    {
        // POTRF: Compute Cholesky factor L
        potrf(tiles[k * n_tiles + k], N);

        // TRSM over the panel below k
        #pragma omp parallel for schedule(static)
        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            trsm(tiles[k * n_tiles + k],
                 tiles[m * n_tiles + k],
                 N,
                 N,
                 Blas_trans,
                 Blas_right);
        }

        // Trailing matrix update
        // #pragma omp parallel for schedule(static)
        // for (std::size_t m = k + 1; m < n_tiles; ++m)
        // {
        //     for (std::size_t n = k + 1; n < m + 1; ++n)
        //     {
        // #pragma omp parallel for collapse(2)
        // for (std::size_t m = k + 1; m < n_tiles; ++m)
        // {
        //   for (std::size_t n = k + 1; n < m + 1; ++n)
        //   {
        //         if (n == m)
        //         {
        //             // SYRK: A = A - B * B^T
        //             syrk(tiles[m * n_tiles + m],
        //                  tiles[m * n_tiles + k],
        //                  N);
        //         }
        //         else
        //         {
        //             // GEMM: C = C - A * B^T
        //             gemm(tiles[m * n_tiles + k],
        //                  tiles[n * n_tiles + k],
        //                  tiles[m * n_tiles + n],
        //                  N, N, N,
        //                  Blas_no_trans,
        //                  Blas_trans);
        //         }
        //   }
        // }
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
        gemm(tiles[m * n_tiles + k], tiles[n * n_tiles + k], tiles[m * n_tiles + n], N, N, N, Blas_no_trans, Blas_trans);
    }
}

    }
 }


// void right_looking_cholesky_tiled(std::vector<std::vector<double>> &tiles, int N, std::size_t n_tiles)
// {
//     #pragma omp parallel
//     {
//         #pragma omp single
//         for (std::size_t k = 0; k < n_tiles; ++k)
//         {
//             // POTRF: Cholesky on diagonal tile
//             potrf(tiles[k * n_tiles + k], N);
//
//             // TRSM: Solve tiles below the diagonal
//             for (std::size_t m = k + 1; m < n_tiles; ++m)
//             {
//                 #pragma omp task firstprivate(m)
//                 trsm(tiles[k * n_tiles + k],
//                      tiles[m * n_tiles + k],
//                      N,
//                      N,
//                      Blas_trans,
//                      Blas_right);
//             }
//
//             // Trailing matrix update
//             for (std::size_t m = k + 1; m < n_tiles; ++m)
//             {
//                 for (std::size_t n = k + 1; n <= m; ++n)
//                 {
//                     #pragma omp task firstprivate(m,n)
//                     if (m == n)
//                     {
//                         // SYRK: Update diagonal tile
//                         syrk(tiles[m * n_tiles + m],
//                              tiles[m * n_tiles + k],
//                              N);
//                     }
//                     else
//                     {
//                         // GEMM: Update off-diagonal tile
//                         gemm(tiles[m * n_tiles + k],
//                              tiles[n * n_tiles + k],
//                              tiles[m * n_tiles + n],
//                              N, N, N,
//                              Blas_no_trans,
//                              Blas_trans);
//                     }
//                 }
//             }
//
//             // Wait for all tasks of this panel to finish before moving to next k
//             #pragma omp taskwait
//         }
//     }
// }


// void right_looking_cholesky_tiled(std::vector<std::vector<double>> &tiles, int N, std::size_t n_tiles)
// {
//     #pragma omp parallel
//     #pragma omp single
//     {
//         for (std::size_t k = 0; k < n_tiles; ++k)
//         {
//             // Diagonal tile
//             auto &Akk = tiles[k * n_tiles + k];
//             #pragma omp task depend(inout: Akk)
//             {
//                 potrf(Akk, N);   // Lkk = Cholesky factor
//             }
//
//             // Tiles below the diagonal (TRSM)
//             for (std::size_t m = k + 1; m < n_tiles; ++m)
//             {
//                 auto &Amk = tiles[m * n_tiles + k];
//                 #pragma omp task depend(in: Akk) depend(inout: Amk)
//                 {
//                     trsm(Akk, Amk, N, N, Blas_trans, Blas_right);
//                 }
//             }
//
//             // Trailing matrix update (SYRK and GEMM)
//             for (std::size_t m = k + 1; m < n_tiles; ++m)
//             {
//                 auto &Amk = tiles[m * n_tiles + k];
//                 for (std::size_t n = k + 1; n <= m; ++n)
//                 {
//                     auto &Ank = tiles[n * n_tiles + k];
//                     auto &Amn = tiles[m * n_tiles + n];
//
//                     if (m == n)
//                     {
//                         // SYRK on diagonal tile
//                         #pragma omp task depend(in: Amk) depend(inout: Amn)
//                         {
//                             syrk(Amn, Amk, N);
//                         }
//                     }
//                     else
//                     {
//                         // GEMM on off-diagonal tile
//                         #pragma omp task depend(in: Amk, Ank) depend(inout: Amn)
//                         {
//                             gemm(Amk, Ank, Amn, N, N, N, Blas_no_trans, Blas_trans);
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// void right_looking_cholesky_tiled(//)_plasma_style(
//     std::vector<std::vector<double>> &tiles,
//     int N,                   // tile size
//     std::size_t n_tiles)     // number of tiles
// {
// #pragma omp parallel
// //#pragma omp master
// #pragma omp single
//     for (std::size_t k = 0; k < n_tiles; ++k)
//     {
//         // -------------------------
//         // POTRF on diagonal
//         // -------------------------
//         auto &Akk = tiles[k * n_tiles + k];
//
//         #pragma omp task depend(inout: Akk)
//         {
//             potrf(Akk, N);   // Lkk = Cholesky factor
//         }
//
//         // -------------------------
//         // TRSM on panel below diagonal
//         // -------------------------
//         for (std::size_t m = k + 1; m < n_tiles; ++m)
//         {
//             auto &Amk = tiles[m * n_tiles + k];
//
//             #pragma omp task depend(in: Akk) depend(inout: Amk)
//             {
//                 trsm(Akk, Amk, N, N, Blas_trans, Blas_right);
//             }
//         }
//
//         // -------------------------
//         // Trailing matrix update
//         // -------------------------
//         for (std::size_t m = k + 1; m < n_tiles; ++m)
//         {
//             auto &Amk = tiles[m * n_tiles + k];
//
//             // SYRK diagonal update
//             auto &Amm = tiles[m * n_tiles + m];
//             #pragma omp task depend(in: Amk) depend(inout: Amm)
//             {
//                 syrk(Amm, Amk, N);
//             }
//
//             // GEMM off-diagonal updates
//             for (std::size_t n = k + 1; n < m; ++n)
//             {
//                 auto &Ank = tiles[n * n_tiles + k];
//                 auto &Amn = tiles[m * n_tiles + n];
//
//                 #pragma omp task depend(in: Amk, Ank) depend(inout: Amn)
//                 {
//                     gemm(Amk, Ank, Amn, N, N, N, Blas_no_trans, Blas_trans);
//                 }
//             }
//         }
//     }
//
// #pragma omp taskwait
// }

// You need these functions defined elsewhere:
// void potrf(std::vector<double> &tile, int N);
// void trsm(const std::vector<double> &A, std::vector<double> &B, int N, int M, ...);
// void syrk(std::vector<double> &C, const std::vector<double> &A, int N);
// void gemm(const std::vector<double> &A, const std::vector<double> &B, std::vector<double> &C, int M, int N, int K, ...);

// void right_looking_cholesky_tiled(std::vector<std::vector<double>> &tiles,
//                                   int N,
//                                   std::size_t n_tiles)
// {
//     #pragma omp parallel
// #pragma omp single
// for (std::size_t k = 0; k < n_tiles; ++k)
// {
//     auto &diag_tile = tiles[k * n_tiles + k];
//     #pragma omp task depend(inout: diag_tile)
//     {
//         potrf(diag_tile, N);
//     }
//
//     for (std::size_t m = k + 1; m < n_tiles; ++m)
//     {
//         auto &panel_tile = tiles[m * n_tiles + k];
//         #pragma omp task depend(in: diag_tile) depend(inout: panel_tile)
//         {
//             trsm(diag_tile, panel_tile, N, N, Blas_trans, Blas_right);
//         }
//     }
//
//     for (std::size_t m = k + 1; m < n_tiles; ++m)
//     {
//         for (std::size_t n = k + 1; n <= m; ++n)
//         {
//             auto &tile_mn = tiles[m * n_tiles + n];
//             if (n == m)
//             {
//                 auto &tile_mm = tiles[m * n_tiles + m];
//                 #pragma omp task depend(in: tile_mn) depend(inout: tile_mm)
//                 {
//                     syrk(tile_mm, tile_mn, N);
//                 }
//             }
//             else
//             {
//                 auto &tile_mk = tiles[m * n_tiles + k];
//                 auto &tile_nk = tiles[n * n_tiles + k];
//                 #pragma omp task depend(in: tile_mk, tile_nk) depend(inout: tile_mn)
//                 {
//                     gemm(tile_mk, tile_nk, tile_mn, N, N, N, Blas_no_trans, Blas_trans);
//                 }
//             }
//         }
//     }
// }
//
// #pragma omp taskwait
//
// }
//
// void right_looking_cholesky_tiled(
//     std::vector<std::vector<double>> &tiles,
//     int N,
//     std::size_t n_tiles)
// {
// #pragma omp parallel
// #pragma omp single
// for (std::size_t k = 0; k < n_tiles; ++k)
// {
// #pragma omp taskgroup
// {
//     // ---- POTRF (diagonal) ----
//     auto &Akk = tiles[k*n_tiles + k];
//
// #pragma omp task depend(inout:Akk)
//     {
//         potrf(Akk, N);
//     }
//
//     // ---- TRSM (panel below diagonal) ----
//     for (std::size_t m = k+1; m < n_tiles; ++m)
//     {
//         auto &Amk = tiles[m*n_tiles + k];
//
// #pragma omp task depend(in:Akk) depend(inout:Amk)
//         {
//             trsm(Akk, Amk, N, N, Blas_trans, Blas_right);
//         }
//     }
//
//     // ---- Trailing matrix update ----
//     for (std::size_t m = k+1; m < n_tiles; ++m)
//     {
//         auto &Lmk = tiles[m*n_tiles + k];
//
//         for (std::size_t n = k+1; n <= m; ++n)
//         {
//             auto &Amn = tiles[m*n_tiles + n];
//
//             if (n == m)
//             {
//                 // SYRK on diagonal tile
// #pragma omp task depend(in:Lmk) depend(inout:Amn)
//                 {
//                     syrk(Amn, Lmk, N);
//                 }
//             }
//             else
//             {
//                 auto &Lnk = tiles[n*n_tiles + k];
//
// #pragma omp task depend(in:Lmk, Lnk) depend(inout:Amn)
//                 {
//                     gemm(Lmk, Lnk, Amn,
//                          N, N, N,
//                          Blas_no_trans, Blas_trans);
//                 }
//             }
//         }
//     }
// }
// }
//
// #pragma omp taskwait
// }
//
//
// void right_looking_cholesky_tiled(
//     std::vector<std::vector<double>> &tiles,
//     int N,
//     std::size_t n_tiles)
// {
// #pragma omp parallel
// #pragma omp single
// for (std::size_t k = 0; k < n_tiles; ++k)
// {
// #pragma omp taskgroup
// {
//     // ---- POTRF on diagonal ----
//     auto &Akk = tiles[k*n_tiles + k];
//
// #pragma omp task depend(inout:Akk)
//     {
//         potrf(Akk, N);   // Akk = Lkk
//     }
//
//     // ---- TRSM on panel below diagonal ----
//     for (std::size_t m = k+1; m < n_tiles; ++m)
//     {
//         auto &Amk = tiles[m*n_tiles + k];   // below-diagonal tile
//
// #pragma omp task depend(in:Akk) depend(inout:Amk)
//         {
//             // Amk = Amk * inv(Lkk^T)
//             trsm(Akk, Amk, N, N, Blas_trans, Blas_right);
//         }
//     }
//
//     // ---- Trailing matrix update (lower triangle only) ----
//     for (std::size_t m = k+1; m < n_tiles; ++m)
//     {
//         auto &Lmk = tiles[m*n_tiles + k];   // panel tile (input to syrk/gemm)
//
//         for (std::size_t n = k+1; n <= m; ++n)
//         {
//             auto &Amn = tiles[m*n_tiles + n];   // target of update (always n<=m)
//
//             if (n == m)
//             {
//                 // Diagonal update: Amm -= Lmk * Lmk^T
// #pragma omp task depend(in:Lmk) depend(inout:Amn)
//                 {
//                     syrk(Amn, Lmk, N);
//                 }
//             }
//             else
//             {
//                 // Off-diagonal update: Amn -= Lmk * Lnk^T
//                 auto &Lnk = tiles[n*n_tiles + k];
//
// #pragma omp task depend(in:Lmk, Lnk) depend(inout:Amn)
//                 {
//                     gemm(Lmk, Lnk, Amn,
//                          N, N, N,
//                          Blas_no_trans, Blas_trans);
//                 }
//             }
//         }
//     }
// }
// }
//
// #pragma omp taskwait
// }


std::vector<double> gen_tile(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_tiles)
{
    std::size_t i_global, j_global;
    double random_value;
    // Create random generator
    size_t seed = row * col;
    std::mt19937 generator ( seed );
    std::uniform_real_distribution<double> distribute( 0, 1 );
    // Preallocate required memory
    std::vector<double> tile;
    tile.reserve(N * N);
    // Compute entries
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        for (std::size_t j = 0; j < N; j++)
        {
            j_global = N * col + j;
            // compute covariance function
            random_value =distribute(generator);
            if (i_global == j_global)
            {
                // noise variance on diagonal
                random_value += N * n_tiles;
            }
            tile.push_back(random_value);
        }
    }
    return tile;
}

std::vector<std::vector<double>> gen_tiled_matrix(
    std::size_t N,
    std::size_t n_tiles)
{
    // Tiled data structure
    std::vector<std::vector<double>> tiled_matrix;
    // Preallocate memory
    tiled_matrix.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // No reserve because of triangular structure

    ///////////////////////////////////////////////////////////////////////////
    // Launch synchronous assembly
    #pragma omp parallel for collapse(2)
    for (std::size_t i = 0; i < n_tiles; ++i)
    for (std::size_t j = 0; j < i + 1; ++j)
    {
      tiled_matrix[i * n_tiles + j] =
          gen_tile(i, j, N, n_tiles);
    }

    return tiled_matrix;
}
