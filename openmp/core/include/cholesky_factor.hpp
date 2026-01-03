#ifndef CPU_CHOLESKY_FACTOR_H
#define CPU_CHOLESKY_FACTOR_H

#pragma once

#include <stdexcept>
#include <vector>

using Tiled_vector_matrix = std::vector<std::vector<double>>;

namespace cpu
{
enum class Variant { for_collapse, for_split, for_naive, sync_future, sync_ref, sync_val, loop_one, loop_two };

inline Variant to_variant(std::string s)
{
    if (s == "for_collapse")
    {
        return Variant::for_collapse;
    }
    if (s == "for_split")
    {
        return Variant::for_split;
    }
    if (s == "for_naive")
    {
        return Variant::for_naive;
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

void right_looking_cholesky_tiled(Variant variant, Tiled_vector_matrix &tiles);

}  // end of namespace cpu
#endif  // end of CPU_CHOLESKY_FACTOR_H
