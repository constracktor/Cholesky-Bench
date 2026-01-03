#ifndef CPU_CHOLESKY_FACTOR_H
#define CPU_CHOLESKY_FACTOR_H

#pragma once

#include "tile_data.hpp"
#include <hpx/future.hpp>

using Tiled_vector_matrix = std::vector<std::vector<double>>;
using Tiled_future_matrix = std::vector<hpx::shared_future<std::vector<double>>>;
using Tiled_mutable_matrix = std::vector<hpx::shared_future<mutable_tile_data<double>>>;

namespace cpu
{
enum class Variant { async_future, async_ref, async_val, sync_future, sync_ref, sync_val, loop_one, loop_two };

inline Variant to_variant(std::string s)
{
    if (s == "async_future")
    {
        return Variant::async_future;
    }
    if (s == "async_ref")
    {
        return Variant::async_ref;
    }
    if (s == "async_val")
    {
        return Variant::async_val;
    }

    if (s == "sync_future")
    {
        return Variant::sync_future;
    }
    if (s == "sync_ref")
    {
        return Variant::sync_ref;
    }
    if (s == "sync_val")
    {
        return Variant::sync_val;
    }

    if (s == "loop_one")
    {
        return Variant::loop_one;
    }
    if (s == "loop_two")
    {
        return Variant::loop_two;
    }

    throw std::invalid_argument("Unknown Variant: " + std::string(s));
}

void right_looking_cholesky_tiled(Variant variant, Tiled_future_matrix &ft_tiles);

void right_looking_cholesky_tiled_loop(Variant variant, Tiled_vector_matrix &tiles);

void right_looking_cholesky_tiled_mutable(Tiled_mutable_matrix &ft_tiles);

}  // end of namespace cpu
#endif  // end of CPU_CHOLESKY_FACTOR_H
