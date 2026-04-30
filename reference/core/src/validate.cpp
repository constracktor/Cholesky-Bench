#include "validate.hpp"

#include "matrix_generation.hpp"

#ifdef ENABLE_MKL
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

#include <cmath>
#include <cstddef>
#include <vector>

namespace cpu
{

double cholesky_residual(std::size_t N, const std::vector<double> &L)
{
    // Build a working copy of L with its strictly upper triangle zeroed out.
    std::vector<double> Lwork(L);
    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = i + 1; j < N; ++j)
        {
            Lwork[i * N + j] = 0.0;
        }
    }

    // Compute LLt = L * L^T (full N x N) with a single dgemm.
    std::vector<double> LLt(N * N, 0.0);
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        static_cast<int>(N),
        static_cast<int>(N),
        static_cast<int>(N),
        1.0,
        Lwork.data(),
        static_cast<int>(N),
        Lwork.data(),
        static_cast<int>(N),
        0.0,
        LLt.data(),
        static_cast<int>(N));

    // Regenerate the original matrix A deterministically and accumulate Frobenius
    // norms of (A - LLt) and A.
    const std::vector<double> A = gen_matrix(N);

    double r_norm_sq = 0.0;
    double a_norm_sq = 0.0;
    for (std::size_t idx = 0; idx < A.size(); ++idx)
    {
        const double d = A[idx] - LLt[idx];
        r_norm_sq += d * d;
        a_norm_sq += A[idx] * A[idx];
    }

    if (a_norm_sq == 0.0)
    {
        return 0.0;
    }
    return std::sqrt(r_norm_sq / a_norm_sq);
}

}  // namespace cpu
