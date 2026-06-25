// Copyright (c) 2026 Alexander Strack
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "cholesky_factor.hpp"

#include "adapter_cblas_fp64.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
#include <ttg.h>
#include <ttg/serialization/std/tuple.h>
#include <ttg/serialization/std/vector.h>
#include <ttg/util/multiindex.h>
#include <tuple>
#include <utility>
#include <vector>

namespace cpu
{

namespace
{

// A tile is a row-major N×N block stored contiguously, identical to the layout
// produced by gen_tile() and consumed by the BLAS adapter wrappers.
using Tile = std::vector<double>;

// Multi-dimensional task identifiers, mirroring the reference TTG potrf example.
using Key1 = ttg::MultiIndex<1>;  // POTRF(k)
using Key2 = ttg::MultiIndex<2>;  // TRSM(m,k) / SYRK(k,m) / dispatch / result
using Key3 = ttg::MultiIndex<3>;  // GEMM(m,n,k)

// Robust integer square root for exact squares.
inline std::size_t isqrt_exact(std::size_t x)
{
    return static_cast<std::size_t>(std::lround(std::sqrt(static_cast<double>(x))));
}

// Lightweight, shared-memory matrix descriptor. It owns no tile storage; it
// only carries the tiling geometry and a pointer to the flat lower-triangular
// storage so the load/result tasks can read and write it.
struct TiledMatrix
{
    std::size_t nt = 0;  // tiles per dimension
    int nb = 0;          // tile size (rows == cols of a tile)
    Tiled_vector_matrix *store = nullptr;

    std::size_t rows() const { return nt; }

    std::size_t cols() const { return nt; }

    int tile_size() const { return nb; }

    // Lower triangle only.
    bool in_matrix(std::size_t row, std::size_t col) const { return col <= row; }
};

///////////////////////////////////////////////////////////////////////////////
// Source task: inject every lower-triangular tile of the original matrix into
// the graph on a Key2 edge, keyed by (row, col).
///////////////////////////////////////////////////////////////////////////////
auto make_load(TiledMatrix &A, ttg::Edge<Key2, Tile> &toop)
{
    auto f = [&A](std::tuple<ttg::Out<Key2, Tile>> &out)
    {
        for (std::size_t i = 0; i < A.rows(); ++i)
        {
            for (std::size_t j = 0; j < A.cols() && A.in_matrix(i, j); ++j)
            {
                // Move the original tile into the graph (the result task writes
                // the factor back into the same slot). This mirrors the
                // upstream make_load_tt, which moves its tiles in rather than
                // deep-copying them.
                ttg::send<0>(Key2{ i, j }, std::move((*A.store)[i * A.nt + j]), out);
            }
        }
    };
    return ttg::make_tt<void>(f, ttg::edges(), ttg::edges(toop), "Load Matrix", {}, { "To Dispatch" });
}

///////////////////////////////////////////////////////////////////////////////
// Dispatcher: route each incoming original tile to the first task that
// consumes it (POTRF, TRSM, SYRK or GEMM).
///////////////////////////////////////////////////////////////////////////////
auto make_dispatcher(ttg::Edge<Key2, Tile> &input,
                     ttg::Edge<Key1, Tile> &to_potrf,
                     ttg::Edge<Key2, Tile> &to_trsm,
                     ttg::Edge<Key2, Tile> &to_syrk,
                     ttg::Edge<Key3, Tile> &to_gemm)
{
    auto f =
        [=](const Key2 &key,
            Tile &&tile,
            std::tuple<ttg::Out<Key1, Tile>, ttg::Out<Key2, Tile>, ttg::Out<Key2, Tile>, ttg::Out<Key3, Tile>> &out)
    {
        if (0 == key[0] && 0 == key[1])
        {
            // First diagonal element goes straight to POTRF.
            ttg::send<0>(Key1{ key[0] }, std::move(tile), out);
            return;
        }
        if (key[0] == key[1])
        {
            // Remaining diagonal elements start a SYRK accumulation chain.
            ttg::send<2>(Key2{ 0, key[0] }, std::move(tile), out);
            return;
        }
        // Lower triangle only.
        assert(key[0] > key[1]);
        if (0 == key[1])
        {
            // First column goes to TRSM directly.
            ttg::send<1>(key, std::move(tile), out);
            return;
        }
        // Everything else starts a GEMM accumulation chain.
        ttg::send<3>(Key3{ key[0], key[1], 0 }, std::move(tile), out);
    };
    return ttg::make_tt(
        f,
        ttg::edges(input),
        ttg::edges(to_potrf, to_trsm, to_syrk, to_gemm),
        "Dispatch",
        { "Input" },
        { "POTRF", "TRSM", "SYRK", "GEMM" });
}

///////////////////////////////////////////////////////////////////////////////
// POTRF: factorize the diagonal tile, then feed its column of TRSM tasks.
///////////////////////////////////////////////////////////////////////////////
auto make_potrf(TiledMatrix &A,
                ttg::Edge<Key1, Tile> &input_disp,     // from the dispatcher
                ttg::Edge<Key1, Tile> &input,          // from the final SYRK of the column
                ttg::Edge<Key2, Tile> &output_trsm,    // to TRSM
                ttg::Edge<Key2, Tile> &output_result)  // to result writer
{
    const int N = A.tile_size();
    const std::size_t nt = A.rows();
    auto f = [=](const Key1 &key, Tile &&tile_kk, std::tuple<ttg::Out<Key2, Tile>, ttg::Out<Key2, Tile>> &out)
    {
        const std::size_t K = key[0];
        potrf(tile_kk, N);

        // Successors: TRSM(m, K) for every m below the diagonal.
        std::vector<Key2> keylist;
        keylist.reserve(nt - K);
        for (std::size_t m = K + 1; m < nt; ++m)
        {
            keylist.push_back(Key2{ m, K });
        }
        ttg::broadcast<0, 1>(std::make_tuple(Key2{ K, K }, keylist), std::move(tile_kk), out);
    };
    return ttg::make_tt(
        f,
        ttg::edges(ttg::fuse(input, input_disp)),
        ttg::edges(output_result, output_trsm),
        "POTRF",
        { "tile_kk/dispatcher" },
        { "output_result", "output_trsm" });
}

///////////////////////////////////////////////////////////////////////////////
// TRSM: triangular solve of an off-diagonal tile against the diagonal factor.
///////////////////////////////////////////////////////////////////////////////
auto make_trsm(TiledMatrix &A,
               ttg::Edge<Key2, Tile> &input_disp,     // from the dispatcher
               ttg::Edge<Key2, Tile> &input_kk,       // from POTRF
               ttg::Edge<Key2, Tile> &input_mk,       // from the previous GEMM
               ttg::Edge<Key2, Tile> &output_diag,    // to SYRK
               ttg::Edge<Key3, Tile> &output_row,     // to GEMM (as left operand)
               ttg::Edge<Key3, Tile> &output_col,     // to GEMM (as right operand)
               ttg::Edge<Key2, Tile> &output_result)  // to result writer
{
    const int N = A.tile_size();
    const std::size_t nt = A.rows();
    auto f =
        [=](const Key2 &key,
            const Tile &tile_kk,
            Tile &&tile_mk,
            std::tuple<ttg::Out<Key2, Tile>, ttg::Out<Key2, Tile>, ttg::Out<Key3, Tile>, ttg::Out<Key3, Tile>> &out)
    {
        const std::size_t M = key[0];
        const std::size_t K = key[1];

        // Solve X * L_kk^T = A_mk  (side = right, lower, transpose).
        trsm(tile_kk, tile_mk, N, N, Blas_trans, Blas_right);

        std::vector<Key3> keylist_row;  // GEMM(M, n, K) for K < n < M
        keylist_row.reserve(M - K);
        std::vector<Key3> keylist_col;  // GEMM(m, M, K) for m > M
        keylist_col.reserve(nt - M - 1);
        for (std::size_t n = K + 1; n < M; ++n)
        {
            keylist_row.push_back(Key3{ M, n, K });
        }
        for (std::size_t m = M + 1; m < nt; ++m)
        {
            keylist_col.push_back(Key3{ m, M, K });
        }

        ttg::broadcast<0, 1, 2, 3>(
            std::make_tuple(key, Key2{ K, M }, keylist_row, keylist_col), std::move(tile_mk), out);
    };
    return ttg::make_tt(
        f,
        ttg::edges(input_kk, ttg::fuse(input_mk, input_disp)),
        ttg::edges(output_result, output_diag, output_row, output_col),
        "TRSM",
        { "tile_kk", "tile_mk/dispatcher" },
        { "output_result", "output_diag", "output_row", "output_col" });
}

///////////////////////////////////////////////////////////////////////////////
// SYRK: accumulate the diagonal trailing update A_mm -= L_mk * L_mk^T.
// The diagonal tile travels down a k-chain; when fully updated it is handed to
// the next POTRF.
///////////////////////////////////////////////////////////////////////////////
auto make_syrk(TiledMatrix &A,
               ttg::Edge<Key2, Tile> &input_disp,    // from the dispatcher
               ttg::Edge<Key2, Tile> &input_mk,      // from TRSM
               ttg::Edge<Key2, Tile> &input_kk,      // from the previous SYRK
               ttg::Edge<Key1, Tile> &output_potrf,  // to POTRF
               ttg::Edge<Key2, Tile> &output_syrk)   // to the next SYRK
{
    const int N = A.tile_size();
    auto f = [=](const Key2 &key,
                 const Tile &tile_mk,
                 Tile &&tile_kk,
                 std::tuple<ttg::Out<Key1, Tile>, ttg::Out<Key2, Tile>> &out)
    {
        const std::size_t K = key[0];
        const std::size_t M = key[1];

        syrk(tile_kk, tile_mk, N);

        if (M == K + 1)
        {
            // Diagonal tile is fully updated: factorize it next.
            ttg::send<0>(Key1{ K + 1 }, std::move(tile_kk), out);
        }
        else
        {
            // Continue accumulating with the next column's TRSM result.
            ttg::send<1>(Key2{ K + 1, M }, std::move(tile_kk), out);
        }
    };
    return ttg::make_tt(
        f,
        ttg::edges(input_mk, ttg::fuse(input_kk, input_disp)),
        ttg::edges(output_potrf, output_syrk),
        "SYRK",
        { "tile_mk", "tile_kk/dispatcher" },
        { "output_potrf", "output_syrk" });
}

///////////////////////////////////////////////////////////////////////////////
// GEMM: accumulate the off-diagonal trailing update A_mn -= L_mk * L_nk^T.
// The tile travels down a k-chain; when fully updated for its column it is
// handed to the TRSM that finalizes it.
///////////////////////////////////////////////////////////////////////////////
auto make_gemm(TiledMatrix &A,
               ttg::Edge<Key3, Tile> &input_disp,   // from the dispatcher
               ttg::Edge<Key3, Tile> &input_mk,     // from TRSM (row)
               ttg::Edge<Key3, Tile> &input_nk,     // from TRSM (col)
               ttg::Edge<Key3, Tile> &input_mn,     // from the previous GEMM
               ttg::Edge<Key2, Tile> &output_trsm,  // to TRSM
               ttg::Edge<Key3, Tile> &output_gemm)  // to the next GEMM
{
    const int N = A.tile_size();
    auto f = [=](const Key3 &key,
                 const Tile &tile_mk,
                 const Tile &tile_nk,
                 Tile &&tile_mn,
                 std::tuple<ttg::Out<Key2, Tile>, ttg::Out<Key3, Tile>> &out)
    {
        const std::size_t M = key[0];
        const std::size_t Ncol = key[1];
        const std::size_t K = key[2];
        assert(M != Ncol && M > K && Ncol > K);

        gemm(tile_mk, tile_nk, tile_mn, N, N, N, Blas_no_trans, Blas_trans);

        if (Ncol == K + 1)
        {
            // Fully updated for its column: hand to TRSM.
            ttg::send<0>(Key2{ M, Ncol }, std::move(tile_mn), out);
        }
        else
        {
            // Continue accumulating across the next k.
            ttg::send<1>(Key3{ M, Ncol, K + 1 }, std::move(tile_mn), out);
        }
    };
    return ttg::make_tt(
        f,
        ttg::edges(input_mk, input_nk, ttg::fuse(input_disp, input_mn)),
        ttg::edges(output_trsm, output_gemm),
        "GEMM",
        { "input_mk", "input_nk", "input_mn/dispatcher" },
        { "output_trsm", "output_gemm" });
}

///////////////////////////////////////////////////////////////////////////////
// Sink task: write each finalized tile back into the flat storage so the
// residual check can read it.
///////////////////////////////////////////////////////////////////////////////
auto make_result(TiledMatrix &A, ttg::Edge<Key2, Tile> &result)
{
    auto f = [&A](const Key2 &key, Tile &&tile, std::tuple<> &out)
    {
        (void) out;
        const std::size_t I = key[0];
        const std::size_t J = key[1];
        (*A.store)[I * A.nt + J] = std::move(tile);
    };
    return ttg::make_tt(f, ttg::edges(result), ttg::edges(), "Result", { "result" }, {});
}

}  // namespace

double right_looking_cholesky_tiled_ttg(Tiled_vector_matrix &tiles)
{
    const std::size_t n_tiles = isqrt_exact(tiles.size());
    const int nb = static_cast<int>(isqrt_exact(tiles[0].size()));

    TiledMatrix A{ n_tiles, nb, &tiles };

    // Edges (named like the reference TTG potrf example).
    ttg::Edge<Key1, Tile> syrk_potrf("syrk_potrf"), disp_potrf("disp_potrf");
    ttg::Edge<Key2, Tile> startup("startup"), result("result");
    ttg::Edge<Key2, Tile> potrf_trsm("potrf_trsm"), trsm_syrk("trsm_syrk"), gemm_trsm("gemm_trsm"),
        syrk_syrk("syrk_syrk"), disp_trsm("disp_trsm"), disp_syrk("disp_syrk");
    ttg::Edge<Key3, Tile> gemm_gemm("gemm_gemm"), trsm_gemm_row("trsm_gemm_row"), trsm_gemm_col("trsm_gemm_col"),
        disp_gemm("disp_gemm");

    auto op_load = make_load(A, startup);
    auto op_disp = make_dispatcher(startup, disp_potrf, disp_trsm, disp_syrk, disp_gemm);
    auto op_potrf = make_potrf(A, disp_potrf, syrk_potrf, potrf_trsm, result);
    auto op_trsm = make_trsm(A, disp_trsm, potrf_trsm, gemm_trsm, trsm_syrk, trsm_gemm_row, trsm_gemm_col, result);
    auto op_syrk = make_syrk(A, disp_syrk, trsm_syrk, syrk_syrk, syrk_potrf, syrk_syrk);
    auto op_gemm = make_gemm(A, disp_gemm, trsm_gemm_row, trsm_gemm_col, gemm_gemm, gemm_trsm, gemm_gemm);
    auto op_result = make_result(A, result);

    auto connected = ttg::make_graph_executable(op_load.get());
    assert(connected);
    (void) connected;

    // Time only the asynchronous factorization (invoke -> fence), excluding the
    // one-time graph construction above. This matches both the upstream TTG
    // potrf benchmark and the factorization-only timing of the OpenMP/HPX
    // variants, so the runtimes are directly comparable.
    auto start = std::chrono::high_resolution_clock::now();
    ttg::execute();
    if (ttg::default_execution_context().rank() == 0)
    {
        op_load->invoke();
    }
    ttg::fence();
    auto stop = std::chrono::high_resolution_clock::now();
    return (stop - start).count() / 1e9;
}

}  // end of namespace cpu
