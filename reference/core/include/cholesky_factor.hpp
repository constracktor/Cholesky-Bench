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
 *   - reference   : single threaded LAPACKE_dpotrf2 call (no tiling;
 *                   parallelism lives entirely inside the threaded BLAS).
 *   - plasma      : single plasma_dpotrf call (PLASMA's high-level
 *                   synchronous Cholesky over the OpenMP runtime).
 *   - plasma_tile : plasma_omp_dpotrf called over a manually-built tile
 *                   descriptor (PLASMA's asynchronous tile interface).
 */
enum class Variant { reference, plasma, plasma_tile };

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
    if (s == "plasma_tile")
    {
        return Variant::plasma_tile;
    }
    throw std::invalid_argument("Unknown Variant: " + s);
}

/**
 * @brief Run the requested reference variant on the full row-major N x N
 *        matrix @p A. Factorisation is in place; @p A holds the lower
 *        triangular factor L on return.
 */
void parallel_blas_cholesky(Variant variant, std::vector<double> &A, int N);

}  // end of namespace cpu
#endif  // end of CPU_CHOLESKY_FACTOR_H
