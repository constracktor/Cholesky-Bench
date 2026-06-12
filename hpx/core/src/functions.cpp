#include "functions.hpp"

#include "cholesky_factor.hpp"
#include "tile_generation.hpp"
#include <cmath>
#include <hpx/future.hpp>

namespace cpu
{

double cholesky_future(Tiled_future_matrix &tiled_matrix, const std::string &variant)
{
    auto start = std::chrono::high_resolution_clock::now();
    ///////////////////////////////////////////////////////////////////////////
    // Launch Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(to_variant(variant), tiled_matrix);
    // Synchronize: only the lower-triangle futures are initialized; waiting on
    // the upper-triangle slots (default-constructed, invalid) is undefined.
    const std::size_t n_tiles = static_cast<std::size_t>(std::sqrt(tiled_matrix.size()));
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            tiled_matrix[i * n_tiles + j].wait();
        }
    }
    ///////////////////////////////////////////////////////////////////////////
    auto stop = std::chrono::high_resolution_clock::now();
    return (stop - start).count() / 1e9;
}

double cholesky_loop(Tiled_vector_matrix &tiled_matrix, const std::string &variant)
{
    auto start = std::chrono::high_resolution_clock::now();
    ///////////////////////////////////////////////////////////////////////////
    // Launch Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled_loop(to_variant(variant), tiled_matrix);
    ///////////////////////////////////////////////////////////////////////////
    auto stop = std::chrono::high_resolution_clock::now();
    return (stop - start).count() / 1e9;
}

double cholesky_void(Tiled_vector_matrix &tiled_matrix, Tiled_void_matrix &dep_tiles)
{
    auto start = std::chrono::high_resolution_clock::now();
    ///////////////////////////////////////////////////////////////////////////
    // Launch Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled_void(tiled_matrix, dep_tiles);
    // Synchronize: only the lower-triangle futures are initialized; waiting on
    // the upper-triangle slots (default-constructed, invalid) is undefined.
    const std::size_t n_tiles = static_cast<std::size_t>(std::sqrt(dep_tiles.size()));
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            dep_tiles[i * n_tiles + j].wait();
        }
    }
    ///////////////////////////////////////////////////////////////////////////
    auto stop = std::chrono::high_resolution_clock::now();
    return (stop - start).count() / 1e9;
}

}  // end of namespace cpu
