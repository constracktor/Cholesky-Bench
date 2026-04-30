#ifndef CPU_CHOLESKY_FACTOR_H
#define CPU_CHOLESKY_FACTOR_H

#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace cpu
{

/**
 * @brief Reference Cholesky variants.
 *
 *   - reference : threaded LAPACKE_dpotrf2 call
 *   - plasma    : plasma_dpotrf call (PLASMA's high-level
 *                 synchronous Cholesky over the OpenMP runtime).
 */
enum class Variant { reference, plasma };

inline Variant to_variant(const std::string &s)
{
    if (s == "reference")
    {
        return Variant::reference;
    }
    if (s == "plasma")
    {
        return Variant::plasma;
    }
    throw std::invalid_argument("Unknown Variant: " + s);
}

/**
 * @brief Run the requested reference variant on the full row-major N x N
 *        matrix. Factorization is in place; @p matrix holds the lower
 *        triangular factor L on return.
 */
void parallel_cholesky(Variant variant, std::vector<double> &matrix, int N);

}  // end of namespace cpu
#endif  // end of CPU_CHOLESKY_FACTOR_H
