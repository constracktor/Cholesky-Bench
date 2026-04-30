#include "matrix_generation.hpp"

#include <random>
#include <vector>

std::vector<double> gen_matrix(std::size_t N)
{
    std::vector<double> A(N * N);

    // The matrix is built row by row in parallel. Each row uses its own RNG
    // seeded by the row index, so the matrix is deterministic and
    // reproducible regardless of the number of threads.
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i)
    {
        std::mt19937 generator(static_cast<std::mt19937::result_type>(i + 1));
        std::uniform_real_distribution<double> distribute(0.0, 1.0);
        for (std::size_t j = 0; j <= i; ++j)
        {
            const double v = distribute(generator);
            A[i * N + j] = v;
            A[j * N + i] = v;
        }
        A[i * N + i] += static_cast<double>(N);
    }

    return A;
}
