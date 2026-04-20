#include "functions.hpp"

#include "cholesky_factor.hpp"
#include "tile_generation.hpp"
#include <hpx/future.hpp>

namespace cpu
{

double cholesky_future(Tiled_future_matrix &tiled_matrix, std::string variant)
{
    auto start = std::chrono::high_resolution_clock::now();
    ///////////////////////////////////////////////////////////////////////////
    // Launch Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(to_variant(variant), tiled_matrix);
    // Synchronize
    hpx::wait_all(tiled_matrix);
    ///////////////////////////////////////////////////////////////////////////
    auto stop = std::chrono::high_resolution_clock::now();
    return (stop - start).count() / 1e9;
}

double cholesky_loop(Tiled_vector_matrix &tiled_matrix, std::string variant)
{
    auto start = std::chrono::high_resolution_clock::now();
    ///////////////////////////////////////////////////////////////////////////
    // Launch Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled_loop(to_variant(variant), tiled_matrix);
    ///////////////////////////////////////////////////////////////////////////
    auto stop = std::chrono::high_resolution_clock::now();
    return (stop - start).count() / 1e9;
}

double cholesky_void(std::size_t problem_size, std::size_t n_tiles)
{
    Tiled_vector_matrix tiles;
    Tiled_void_matrix dep_tiles;
    gen_void_tiled_matrix(tiles, dep_tiles, problem_size, n_tiles);

    auto start = std::chrono::high_resolution_clock::now();
    ///////////////////////////////////////////////////////////////////////////
    // Launch Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled_void(tiles, dep_tiles, n_tiles);
    // Synchronize: wait on all void futures in the lower triangle
    hpx::wait_all(dep_tiles);
    ///////////////////////////////////////////////////////////////////////////
    auto stop = std::chrono::high_resolution_clock::now();
    return (stop - start).count() / 1e9;
}

}  // end of namespace cpu
