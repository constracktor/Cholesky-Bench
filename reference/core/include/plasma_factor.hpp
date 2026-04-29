#ifndef CPU_PLASMA_FACTOR_H
#define CPU_PLASMA_FACTOR_H

#pragma once

#include <vector>

namespace cpu
{

/**
 * @brief PLASMA tiled Cholesky on a row-major N x N buffer using the
 *        high-level synchronous API (plasma_dpotrf).
 *
 * PLASMA's high-level API is column-major, so we ask for @c PlasmaUpper:
 * the upper triangle in PLASMA's column-major view aliases the lower
 * triangle in our row-major view, which is the layout the validation
 * routine expects (and which matches the LAPACKE_dpotrf2 reference).
 *
 * Caller is responsible for having invoked plasma_init() at startup; that
 * cost is intentionally amortised over all timed calls and stays out of the
 * timed region.
 *
 * Throws @c std::runtime_error before calling PLASMA if the descriptor
 * size computation inside plasma_desc_triangular_create() would overflow
 * int32 (PLASMA 24.8.7 still does this multiplication in @c int). This
 * keeps PLASMA's own multi-line error spam off stderr when the surrounding
 * sweep walks past N=65280.
 */
void plasma_cholesky(std::vector<double> &A, int N);

/**
 * @brief PLASMA tiled Cholesky on a row-major N x N buffer using the
 *        asynchronous tile interface (plasma_omp_dpotrf).
 *
 * Allocates the tile-layout backing store ourselves with size_t
 * arithmetic, then wraps it in a @c plasma_desc_t via
 * plasma_desc_general_init -- which performs no malloc and therefore
 * sidesteps PLASMA 24.8.7's int32 overflow inside the create routines.
 * This means the tile path is expected to keep working past N>65280
 * where the high-level @c plasma_cholesky aborts.
 *
 * After the descriptor is set up, PLASMA's tile-API routines translate
 * our row-major buffer into tile layout (plasma_omp_dge2desc), run the
 * tiled factorisation (plasma_omp_dpotrf with PlasmaUpper), and
 * translate back (plasma_omp_ddesc2ge). The output layout matches the
 * high-level path: row-major lower triangle holds L.
 */
void plasma_tile_cholesky(std::vector<double> &A, int N);

}  // end of namespace cpu
#endif  // end of CPU_PLASMA_FACTOR_H
