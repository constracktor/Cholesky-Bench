#include "validate.hpp"

#include "tile_generation.hpp"

#ifdef GPRAT_ENABLE_MKL
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

#include <algorithm>
#include <cmath>

namespace cpu
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

    double r_norm_sq = 0.0;
    double a_norm_sq = 0.0;

    std::vector<double> C(static_cast<std::size_t>(N) * static_cast<std::size_t>(N));

    // Walk the stored lower-triangular tile region: m >= n.
    for (std::size_t m = 0; m < n_tiles; ++m)
    {
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
            r_norm_sq += weight * tile_r;
            a_norm_sq += weight * tile_a;
        }
    }

    if (a_norm_sq == 0.0)
    {
        return 0.0;
    }
    return std::sqrt(r_norm_sq / a_norm_sq);
}

}  // namespace cpu
