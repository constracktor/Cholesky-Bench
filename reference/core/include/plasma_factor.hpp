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
 * Allocates an *uninitialised* general (full N x N) tile-layout backing
 * store ourselves and wraps it in a @c plasma_desc_t via
 * plasma_desc_general_init -- which performs no malloc and therefore
 * sidesteps PLASMA 24.8.7's int32 overflow inside the create routines.
 *
 * Leaving the buffer uninitialised lets plasma_omp_dge2desc first-touch
 * each tile from its consuming core, so pages land on the right NUMA
 * node instead of all on the main thread's. That is the optimisation
 * that closes part of the runtime gap with @c plasma_cholesky; the
 * remainder of the gap is the wider working-set of the general
 * descriptor (full N*N tile area vs the high-level path's triangular
 * mt*(mt+1)/2 area), which would only be recovered by switching to
 * @c plasma_desc_triangular_init -- attempted but found incompatible
 * with the dge2desc/ddesc2ge translation routines in PLASMA 24.8.7.
 *
 * Note: PLASMA does int32 tile-offset arithmetic during execution as
 * well, so the tile path is also bounded by an int32 overflow guard
 * (general formula). Past the bound this function throws and
 * @c main.cpp's catch handler records @c nan rather than letting PLASMA
 * segfault.
 */
void plasma_tile_cholesky(std::vector<double> &A, int N);

}  // end of namespace cpu
#endif  // end of CPU_PLASMA_FACTOR_H
