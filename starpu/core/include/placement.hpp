// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef PLACEMENT_H
#define PLACEMENT_H

#pragma once

#include <cstddef>
#include <starpu.h>

/**
 * Tile placement. Tile rows are distributed cyclically over the NUMA domains of the CPU workers, the columns over
 * the workers of a domain. Tiles are generated (first touched) by a thread bound to their owner worker, which puts
 * their pages in the owner's NUMA domain, and every task is restricted to the workers of the domain that owns the
 * tile it writes (the scheduler balances the load among them).
 */
namespace placement
{

/// Group the CPU workers by NUMA domain. Call once after starpu_init.
void init();

/// Worker that owns (generates) tile (row, col).
int owner_worker(std::size_t row, std::size_t col);

/// Restrict @p task to the workers of the NUMA domain owning tile row @p row (the row of the tile it writes).
void place_task(starpu_task *task, std::size_t row);

/// Bind the calling thread to the CPU of @p worker.
void bind_to_worker(int worker);

/// Number of CPU workers.
int n_workers();

}  // namespace placement

#endif  // end of PLACEMENT_H
