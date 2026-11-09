// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "tile_generation.hpp"

#include <vector>

#ifndef DISABLE_COMPUTATION
#include <random>
#endif

#if defined(__linux__)
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace
{
#if defined(__linux__) && defined(SYS_set_mempolicy) && !defined(DISABLE_COMPUTATION)
// Highest online NUMA node id + 1, parsed from a range list such as "0-7" or
// "0,2-3"; 0 if it cannot be determined.
unsigned long online_node_bits()
{
    std::ifstream f("/sys/devices/system/node/online");
    std::string list;
    if (!std::getline(f, list))
    {
        return 0;
    }
    unsigned long max_node = 0;
    bool found = false;
    const char *p = list.c_str();
    while (*p)
    {
        if (!std::isdigit(static_cast<unsigned char>(*p)))
        {
            ++p;
            continue;
        }
        char *end;
        unsigned long hi = std::strtoul(p, &end, 10);
        p = end;
        if (*p == '-')
        {
            hi = std::strtoul(p + 1, &end, 10);
            p = end;
        }
        max_node = hi > max_node ? hi : max_node;
        found = true;
    }
    return found ? max_node + 1 : 0;
}

// Spread the pages of the tile storage round-robin over all online NUMA nodes
// while the tiles are generated (the pages are first touched by the generating
// thread). Without this, every tile lives on the NUMA node of the generating
// thread and the memory-bound tasks of all other NUMA domains compete for one
// node's memory bandwidth. Uses the raw syscall, so no libnuma is required.
// The nodemask is sized from the online node count; if that is unknown or the
// call fails (non-NUMA or restricted systems) interleaving is silently skipped.
struct InterleavePolicy
{
    constexpr static int mpol_default = 0;
    constexpr static int mpol_interleave = 3;
    constexpr static unsigned long max_nodes = 1024;  // upper bound for a sane node count

    InterleavePolicy()
    {
        const unsigned long bits = online_node_bits();
        constexpr unsigned long word_bits = 8 * sizeof(unsigned long);
        if (bits == 0 || bits > max_nodes)
        {
            return;
        }
        std::vector<unsigned long> mask((bits + word_bits - 1) / word_bits, 0UL);
        for (unsigned long n = 0; n < bits; n++)
        {
            mask[n / word_bits] |= 1UL << (n % word_bits);
        }
        // The kernel reads maxnode - 1 bits of the mask.
        active = syscall(SYS_set_mempolicy, mpol_interleave, mask.data(), bits + 1) == 0;
    }

    ~InterleavePolicy()
    {
        if (active)
        {
            syscall(SYS_set_mempolicy, mpol_default, nullptr, 0UL);
        }
    }

    InterleavePolicy(const InterleavePolicy &) = delete;
    InterleavePolicy &operator=(const InterleavePolicy &) = delete;

    bool active = false;
};
#else
struct InterleavePolicy
{ };
#endif
}  // namespace

std::vector<double> gen_tile(std::size_t row, std::size_t col, std::size_t N, std::size_t n_tiles)
{
#ifdef DISABLE_COMPUTATION
    // No-op path for task-overhead measurements: return an empty tile.
    (void) row;
    (void) col;
    (void) N;
    (void) n_tiles;
    return {};
#else
    std::size_t i_global, j_global;
    double random_value;
    // Create random generator
    size_t seed = row * n_tiles + col;
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribute(0, 1);
    // Preallocate required memory
    std::vector<double> tile;
    tile.resize(N * N);
    // Compute entries
    // Check for diagonal tile
    if (row == col)
    {
        for (std::size_t i = 0; i < N; i++)
        {
            i_global = N * row + i;
            for (std::size_t j = 0; j <= i; j++)
            {
                j_global = N * col + j;
                random_value = distribute(generator);

                if (i_global == j_global)
                {
                    random_value += N * n_tiles;
                }
                tile[i * N + j] = random_value;
                tile[j * N + i] = random_value;
            }
        }
    }
    else
    {
        for (std::size_t i = 0; i < N; i++)
        {
            for (std::size_t j = 0; j < N; j++)
            {
                random_value = distribute(generator);
                tile[i * N + j] = random_value;
            }
        }
    }
    return tile;
#endif
}

Tiled_vector_matrix gen_tiled_matrix(std::size_t problem_size, std::size_t n_tiles)
{
    // Tiled data structure
    InterleavePolicy numa_interleave;
    Tiled_vector_matrix tiled_matrix;
    // Preallocate memory
    tiled_matrix.resize(n_tiles * n_tiles);  // No reserve because of triangular structure

#ifdef DISABLE_COMPUTATION
    // No-op path: leave all inner vectors empty.
    (void) problem_size;
#else
    std::size_t tile_size = problem_size / n_tiles;
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            tiled_matrix[i * n_tiles + j] = gen_tile(i, j, tile_size, n_tiles);
        }
    }
#endif

    return tiled_matrix;
}
