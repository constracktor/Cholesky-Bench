#include "tile_generation.hpp"

#ifndef DISABLE_COMPUTATION
#include <random>
#endif

std::vector<double> gen_tile(std::size_t row, std::size_t col, std::size_t N, std::size_t n_tiles)
{
#ifdef DISABLE_COMPUTATION
    // No-op path for task-overhead measurements: skip the random fill and
    // return an empty tile. Task dependency addresses work on the vector
    // object itself, which remains distinct per slot in gen_tiled_matrix.
    (void)row;
    (void)col;
    (void)N;
    (void)n_tiles;
    return {};
#else
    std::size_t i_global, j_global;
    double random_value;
    // Create random generator
    size_t seed = row * col;
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribute(0, 1);
    // Preallocate required memory
    std::vector<double> tile;
    tile.resize(N * N);
    // Compute entries
    // Check for diagonal tile
    if (row == col)
    {
        for (std::size_t i = 0; i < N; i++)
        {
            i_global = N * row + i;
            for (std::size_t j = 0; j <= i; j++)
            {
                j_global = N * col + j;
                // compute covariance function
                random_value = distribute(generator);

                if (i_global == j_global)
                {
                    random_value += N * n_tiles;
                }
                tile[i * N + j] = random_value;
                tile[j * N + i] = random_value;
            }
        }
    }
    else
    {
        for (std::size_t i = 0; i < N; i++)
        {
            for (std::size_t j = 0; j < N; j++)
            {
                random_value = distribute(generator);
                tile[i * N + j] = random_value;
            }
        }
    }
    // // print tile
    // std::cout << "(" << row << "," << col << ")\n";
    // for (std::size_t i = 0; i < N; i++)
    // {
    //     for (std::size_t j = 0; j < N; j++)
    //     {
    //         std::cout << tile[i*N +j] << " ";
    //     }
    //     std::cout << "\n";
    // }
    // std::cout << "\n";
    return tile;
#endif
}

Tiled_vector_matrix gen_tiled_matrix(std::size_t problem_size, std::size_t n_tiles)
{
    // Tiled data structure
    Tiled_vector_matrix tiled_matrix;
    // Preallocate memory
    tiled_matrix.resize(n_tiles * n_tiles);  // No reserve because of triangular structure

#ifdef DISABLE_COMPUTATION
    // No-op path: leave all inner vectors empty. Each slot is still a
    // distinct std::vector<double> object, which is all the OpenMP
    // depend() clauses need to identify dependencies.
    (void)problem_size;
#else
    std::size_t tile_size = problem_size / n_tiles;
///////////////////////////////////////////////////////////////////////////
// Launch synchronous assembly
#pragma omp parallel for collapse(2)
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j < i + 1; ++j)
        {
            tiled_matrix[i * n_tiles + j] = gen_tile(i, j, tile_size, n_tiles);
        }
    }
#endif

    return tiled_matrix;
}
