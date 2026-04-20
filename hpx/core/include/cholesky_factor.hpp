#ifndef CPU_CHOLESKY_FACTOR_H
#define CPU_CHOLESKY_FACTOR_H

#pragma once

#include <hpx/future.hpp>

using Tiled_vector_matrix = std::vector<std::vector<double>>;
using Tiled_future_matrix = std::vector<hpx::shared_future<std::vector<double>>>;
using Tiled_void_matrix   = std::vector<hpx::shared_future<void>>;

namespace cpu
{
enum class Variant { async_future, sync_future, loop_one, loop_two, async_void };

inline Variant to_variant(std::string s)
{
    if (s == "async_future")
    {
        return Variant::async_future;
    }
    if (s == "sync_future")
    {
        return Variant::sync_future;
    }
    if (s == "loop_one")
    {
        return Variant::loop_one;
    }
    if (s == "loop_two")
    {
        return Variant::loop_two;
    }
    if (s == "async_void")
    {
        return Variant::async_void;
    }

    throw std::invalid_argument("Unknown Variant: " + std::string(s));
}

void right_looking_cholesky_tiled(Variant variant, Tiled_future_matrix &ft_tiles);

void right_looking_cholesky_tiled_loop(Variant variant, Tiled_vector_matrix &tiles);

/**
 * @brief Right-looking tiled Cholesky using void futures for dependency tracking.
 *        Tile data lives in @p tiles (no copies); @p dep_tiles carries only completion signals.
 * @param tiles     flat lower-triangular tile data (mutated in-place)
 * @param dep_tiles matching void futures (updated to track each tile's latest operation)
 * @param n_tiles   tiles per dimension
 */
void right_looking_cholesky_tiled_void(Tiled_vector_matrix &tiles,
                                       Tiled_void_matrix &dep_tiles,
                                       std::size_t n_tiles);

}  // end of namespace cpu
#endif  // end of CPU_CHOLESKY_FACTOR_H
