#ifndef CPU_VALIDATE_H
#define CPU_VALIDATE_H

#pragma once

#include <cstddef>
#include <vector>

namespace cpu
{

/**
 * @brief Compute the relative Cholesky residual ||A - L * L^T||_F / ||A||_F
 *        for the dense, row-major reference factorization.
 *
 * The original A is regenerated on the fly with the same deterministic seed
 * used by gen_matrix, so no extra storage is needed.
 *
 * @param N matrix dimension (must match the factorization)
 * @param L row-major buffer of length N*N holding the factor returned by
 *          LAPACKE_dpotrf with uplo='L' (only the lower triangle is read)
 * @return relative Frobenius residual
 */
double cholesky_residual(std::size_t N, const std::vector<double> &L);

}  // namespace cpu

#endif  // end of CPU_VALIDATE_H
