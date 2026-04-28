#include "validate.hpp"

#include "tile_generation.hpp"

#ifdef GPRAT_ENABLE_MKL
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

#include <algorithm>
#include <cmath>
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>

namespace cpu
{

namespace
{

// Zero the strictly upper triangle of an N x N row-major matrix in place.
// Used so diagonal L tiles can participate in a plain dgemm.
inline void zero_strict_upper(std::vector<double> &T, int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = i + 1; j < N; ++j)
        {
            T[static_cast<std::size_t>(i) * N + j] = 0.0;
        }
    }
}

}  // namespace

double cholesky_residual(std::size_t problem_size, std::size_t n_tiles, const Tiled_vector_matrix &L)
{
    const int N = static_cast<int>(problem_size / n_tiles);

    // Make lower-triangular copies of every diagonal L tile up front.
    std::vector<std::vector<double>> L_diag(n_tiles);
    for (std::size_t k = 0; k < n_tiles; ++k)
    {
        L_diag[k] = L[k * n_tiles + k];
        zero_strict_upper(L_diag[k], N);
    }

    auto Ltile = [&](std::size_t m, std::size_t n) -> const std::vector<double> &
    { return (m == n) ? L_diag[m] : L[m * n_tiles + n]; };

    // Per-row partial sums; merged sequentially after the parallel loop.
    std::vector<double> r_row(n_tiles, 0.0);
    std::vector<double> a_row(n_tiles, 0.0);

    // Outer loop work is triangular; small dynamic chunks keep every worker busy.
    auto par_dyn = hpx::execution::par.with(hpx::execution::experimental::dynamic_chunk_size(1));

    hpx::experimental::for_loop(
        par_dyn,
        std::size_t{ 0 },
        n_tiles,
        [&](std::size_t m)
        {
            std::vector<double> C(static_cast<std::size_t>(N) * static_cast<std::size_t>(N));

            double row_r = 0.0;
            double row_a = 0.0;
            for (std::size_t n = 0; n <= m; ++n)
            {
                // Reconstruct (L L^T)_{m,n} = sum_{k=0}^{n} L_{m,k} * L_{n,k}^T.
                std::fill(C.begin(), C.end(), 0.0);
                for (std::size_t k = 0; k <= n; ++k)
                {
                    const auto &Lmk = Ltile(m, k);
                    const auto &Lnk = Ltile(n, k);
                    cblas_dgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasTrans,
                        N,
                        N,
                        N,
                        1.0,
                        Lmk.data(),
                        N,
                        Lnk.data(),
                        N,
                        1.0,
                        C.data(),
                        N);
                }

                // Regenerate the original A tile deterministically
                const std::vector<double> A_tile = gen_tile(m, n, static_cast<std::size_t>(N), n_tiles);

                double tile_r = 0.0;
                double tile_a = 0.0;
                for (std::size_t idx = 0; idx < A_tile.size(); ++idx)
                {
                    const double d = A_tile[idx] - C[idx];
                    tile_r += d * d;
                    tile_a += A_tile[idx] * A_tile[idx];
                }

                // Only the lower triangular tile region is stored. A diagonal
                // tile already carries full symmetric data (gen_tile fills
                // both halves), so its squared-Frobenius contribution is
                // counted once. An off-diagonal tile stands in for itself
                // and its unstored transpose, so it counts twice.
                const double weight = (m == n) ? 1.0 : 2.0;
                row_r += weight * tile_r;
                row_a += weight * tile_a;
            }
            r_row[m] = row_r;
            a_row[m] = row_a;
        });

    double r_norm_sq = 0.0;
    double a_norm_sq = 0.0;
    for (std::size_t m = 0; m < n_tiles; ++m)
    {
        r_norm_sq += r_row[m];
        a_norm_sq += a_row[m];
    }

    if (a_norm_sq == 0.0)
    {
        return 0.0;
    }
    return std::sqrt(r_norm_sq / a_norm_sq);
}

}  // namespace cpu
