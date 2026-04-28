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
 * parameters used at matrix generation.
 *
 * @param problem_size full matrix dimension (must match the factorization)
 * @param n_tiles      number of tiles per dimension (must match)
 * @param L            factorized matrix in lower-triangular tile storage
 *
 * @return relative Frobenius residual
 **/
double cholesky_residual(std::size_t problem_size, std::size_t n_tiles, const Tiled_vector_matrix &L);

}  // namespace cpu

#endif  // end of CPU_VALIDATE_H
