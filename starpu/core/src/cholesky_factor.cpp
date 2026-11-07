// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "cholesky_factor.hpp"

#include "adapter_cblas_fp64.hpp"
#include "placement.hpp"
#include <cmath>
#include <stdexcept>

namespace cpu
{
namespace
{
// StarPU CPU implementations. Tiles are square N x N row-major matrices; buffers[i] is the
// matrix interface of the i-th handle passed to starpu_task_insert (see modes below).
#ifndef DISABLE_COMPUTATION
inline double *tile_ptr(void *buffer) { return reinterpret_cast<double *>(STARPU_MATRIX_GET_PTR(buffer)); }

inline int tile_dim(void *buffer) { return static_cast<int>(STARPU_MATRIX_GET_NX(buffer)); }
#endif

void potrf_cpu(void *buffers[], void *)
{
#ifndef DISABLE_COMPUTATION
    potrf(tile_ptr(buffers[0]), tile_dim(buffers[0]));
#else
    (void) buffers;
#endif
}

void trsm_cpu(void *buffers[], void *)
{
#ifndef DISABLE_COMPUTATION
    const int N = tile_dim(buffers[0]);
    trsm(tile_ptr(buffers[0]), tile_ptr(buffers[1]), N, N, Blas_trans, Blas_right);
#else
    (void) buffers;
#endif
}

void syrk_cpu(void *buffers[], void *)
{
#ifndef DISABLE_COMPUTATION
    syrk(tile_ptr(buffers[1]), tile_ptr(buffers[0]), tile_dim(buffers[0]));
#else
    (void) buffers;
#endif
}

void gemm_cpu(void *buffers[], void *)
{
#ifndef DISABLE_COMPUTATION
    const int N = tile_dim(buffers[0]);
    gemm(tile_ptr(buffers[0]), tile_ptr(buffers[1]), tile_ptr(buffers[2]), N, N, N, Blas_no_trans, Blas_trans);
#else
    (void) buffers;
#endif
}

struct Codelets
{
    starpu_codelet potrf;
    starpu_codelet trsm;
    starpu_codelet syrk;
    starpu_codelet gemm;

    Codelets()
    {
        init(potrf, potrf_cpu, "cholesky_potrf", 1);
        potrf.modes[0] = STARPU_RW;

        init(trsm, trsm_cpu, "cholesky_trsm", 2);
        trsm.modes[0] = STARPU_R;
        trsm.modes[1] = STARPU_RW;

        init(syrk, syrk_cpu, "cholesky_syrk", 2);
        syrk.modes[0] = STARPU_R;
        syrk.modes[1] = STARPU_RW;

        init(gemm, gemm_cpu, "cholesky_gemm", 3);
        gemm.modes[0] = STARPU_R;
        gemm.modes[1] = STARPU_R;
        gemm.modes[2] = STARPU_RW;
    }

    static void init(starpu_codelet &cl, starpu_cpu_func_t func, const char *name, int nbuffers)
    {
        starpu_codelet_init(&cl);
        cl.where = STARPU_CPU;
        cl.cpu_funcs[0] = func;
        cl.nbuffers = nbuffers;
        cl.name = name;
    }
};

// Constructed on first use, after which the codelets are immutable.
Codelets &codelets()
{
    static Codelets cls;
    return cls;
}

std::size_t isqrt_exact(std::size_t n)
{
    const std::size_t r = static_cast<std::size_t>(std::sqrt(static_cast<double>(n)) + 0.5);
    if (r * r != n)
    {
        throw std::invalid_argument("size is not a perfect square: " + std::to_string(n));
    }
    return r;
}
}  // namespace

void register_tiles(Tiled_vector_matrix &tiles, Tile_handles &handles)
{
    const std::size_t n_tiles = isqrt_exact(tiles.size());
    handles.assign(n_tiles * n_tiles, nullptr);
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            const std::size_t idx = i * n_tiles + j;
#ifndef DISABLE_COMPUTATION
            const uint32_t N = static_cast<uint32_t>(isqrt_exact(tiles[idx].size()));
            starpu_matrix_data_register(
                &handles[idx],
                STARPU_MAIN_RAM,
                reinterpret_cast<uintptr_t>(tiles[idx].data()),
                N,
                N,
                N,
                sizeof(double));
#else
            // tiles are empty in no-op builds; keep only the dependency structure
            starpu_void_data_register(&handles[idx]);
#endif
        }
    }
}

namespace
{
// Build and submit a task directly (avoids the variadic argument parsing of starpu_task_insert).
// The task is placed on the NUMA domain owning tile row `row`, the row of the tile it writes.
inline void submit(starpu_codelet &cl,
                   std::size_t row,
                   starpu_data_handle_t h0,
                   starpu_data_handle_t h1 = nullptr,
                   starpu_data_handle_t h2 = nullptr)
{
    starpu_task *task = starpu_task_create();
    task->cl = &cl;
    placement::place_task(task, row);
    task->handles[0] = h0;
    if (h1)
    {
        task->handles[1] = h1;
    }
    if (h2)
    {
        task->handles[2] = h2;
    }
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
}
}  // namespace

void submit_right_looking_cholesky(Tile_handles &handles, std::size_t n_tiles, bool synchronize)
{
    Codelets &cl = codelets();
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        // POTRF: Compute Cholesky factor L
        submit(cl.potrf, k, handles[k * n_tiles + k]);
        if (synchronize)
        {
            starpu_task_wait_for_all();
        }

        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // TRSM: Solve X * L^T = A
            submit(cl.trsm, m, handles[k * n_tiles + k], handles[m * n_tiles + k]);
        }
        if (synchronize)
        {
            starpu_task_wait_for_all();
        }

        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // SYRK: A = A - B * B^T
            submit(cl.syrk, m, handles[m * n_tiles + k], handles[m * n_tiles + m]);

            for (std::size_t n = k + 1; n < m; n++)
            {
                // GEMM: C = C - A * B^T
                submit(cl.gemm, m, handles[m * n_tiles + k], handles[n * n_tiles + k], handles[m * n_tiles + n]);
            }
        }
        if (synchronize)
        {
            starpu_task_wait_for_all();
        }
    }
}

void unregister_tiles(Tile_handles &handles, std::size_t n_tiles)
{
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            starpu_data_unregister(handles[i * n_tiles + j]);
            handles[i * n_tiles + j] = nullptr;
        }
    }
}

}  // end of namespace cpu
