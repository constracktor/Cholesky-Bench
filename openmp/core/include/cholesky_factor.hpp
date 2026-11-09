// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CPU_CHOLESKY_FACTOR_H
#define CPU_CHOLESKY_FACTOR_H

#pragma once

#include <stdexcept>
#include <string>
#include <vector>

using Tiled_vector_matrix = std::vector<std::vector<double>>;

namespace cpu
{
enum class Variant { for_collapse, for_naive, task_naive, task_depend, task_prio, task_domain };

inline Variant to_variant(const std::string &s)
{
    if (s == "for_collapse")
    {
        return Variant::for_collapse;
    }
    if (s == "for_naive")
    {
        return Variant::for_naive;
    }
    if (s == "task_naive")
    {
        return Variant::task_naive;
    }
    if (s == "task_depend")
    {
        return Variant::task_depend;
    }
    if (s == "task_prio")
    {
        return Variant::task_prio;
    }
    if (s == "task_domain")
    {
        return Variant::task_domain;
    }

    throw std::invalid_argument("Unknown Variant: " + s);
}

void right_looking_cholesky_tiled(Variant variant, Tiled_vector_matrix &tiles);

}  // end of namespace cpu
#endif  // end of CPU_CHOLESKY_FACTOR_H
