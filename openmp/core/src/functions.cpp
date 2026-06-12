#include "functions.hpp"

#include "cholesky_factor.hpp"
#include <chrono>

namespace cpu
{

double cholesky(Tiled_vector_matrix &tiled_matrix, const std::string &variant)
{
    auto start = std::chrono::high_resolution_clock::now();
    ///////////////////////////////////////////////////////////////////////////
    // Launch Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(to_variant(variant), tiled_matrix);
    ///////////////////////////////////////////////////////////////////////////
    auto stop = std::chrono::high_resolution_clock::now();
    return (stop - start).count() / 1e9;
}

}  // end of namespace cpu
