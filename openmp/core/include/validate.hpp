#ifndef CPU_VALIDATE_H
#define CPU_VALIDATE_H

#pragma once

#include <cstddef>
#include <vector>

using Tiled_vector_matrix = std::vector<std::vector<double>>;

namespace cpu
{

/**
 * @brief Compute the relative Cholesky residual
 *        ||A - L * L^T||_F / ||A||_F
 *
 * A is reconstructed on the fly by calling @c gen_tile with the same
 * parameters used at matrix generation, so the caller does not need to
 * hold a second full copy of the symmetric input matrix. This matters at
 * large @p problem_size where a deep copy would not fit in memory.
 *
 * @param problem_size full matrix dimension (must match the factorization)
 * @param n_tiles      number of tiles per dimension (must match)
 * @param L            factorized matrix in lower-triangular tile storage;
 *                     diagonal tiles contain a lower-triangular factor
 *                     with undefined upper triangle, and off-diagonal
 *                     tiles contain the corresponding panel blocks.
 */
double cholesky_residual(std::size_t problem_size, std::size_t n_tiles, const Tiled_vector_matrix &L);

}  // namespace cpu

#endif  // end of CPU_VALIDATE_H
