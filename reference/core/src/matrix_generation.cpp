#include "matrix_generation.hpp"

#include <random>
#include <vector>

std::vector<double> gen_matrix(std::size_t N)
{
    // Identical to gen_tile(row=0, col=0, N, n_tiles=1).
    const std::size_t row = 0, col = 0, n_tiles = 1;
    std::size_t i_global, j_global;
    double random_value;
    size_t seed = row * n_tiles + col;
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribute(0, 1);
    std::vector<double> A;
    A.resize(N * N);
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        for (std::size_t j = 0; j <= i; j++)
        {
            j_global = N * col + j;
            random_value = distribute(generator);
            if (i_global == j_global)
            {
                random_value += N * n_tiles;
            }
            A[i * N + j] = random_value;
            A[j * N + i] = random_value;
        }
    }
    return A;
}
