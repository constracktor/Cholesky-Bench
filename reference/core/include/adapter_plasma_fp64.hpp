#ifndef CPU_ADAPTER_PLASMA_FP64_H
#define CPU_ADAPTER_PLASMA_FP64_H

#pragma once

#include <vector>

namespace cpu
{

/**
 * @brief PLASMA tiled Cholesky on a row-major N x N buffer using the
 *        high-level synchronous API (plasma_dpotrf).
 *
 * Throws @c std::runtime_error before calling PLASMA when the descriptor
 * size computation inside plasma_desc_*_create() would overflow int32.
 *
 */
void plasma_potrf(std::vector<double> &A, int N);

}  // end of namespace cpu
#endif  // end of CPU_ADAPTER_PLASMA_FP64_H
