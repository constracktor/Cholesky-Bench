#ifndef MATRIX_GENERATION_H
#define MATRIX_GENERATION_H

#pragma once

#include <cstddef>
#include <vector>

/**
 * @brief Generate a deterministic, dense, row-major SPD matrix of size N x N.
 *
 * Entries are uniform on [0, 1) using a per-row seed; the diagonal is shifted
 * by +N to guarantee strict diagonal dominance and therefore symmetric
 * positive definiteness. The result is stored as a single contiguous
 * std::vector<double> of length N*N in row-major order.
 *
 * @param N matrix dimension
 * @return owning row-major buffer of length N*N
 */
std::vector<double> gen_matrix(std::size_t N);

#endif
