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
        #pragma omp parallel for collapse(2)
        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
          for (std::size_t n = k + 1; n < m + 1; ++n)
          {
                if (n == m)
                {
                    // SYRK: A = A - B * B^T
                    syrk(tiles[m * n_tiles + m],
                         tiles[m * n_tiles + k],
                         N);
                }
                else
                {
                    // GEMM: C = C - A * B^T
                    gemm(tiles[m * n_tiles + k],
                         tiles[n * n_tiles + k],
                         tiles[m * n_tiles + n],
                         N, N, N,
                         Blas_no_trans,
                         Blas_trans);
                }
          }
        }
    }
}

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
