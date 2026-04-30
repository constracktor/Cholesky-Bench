#include "functions.hpp"

#include "cholesky_factor.hpp"
#include <chrono>

namespace cpu
{

double cholesky(std::vector<double> &matrix, std::size_t N, const std::string &variant)
{
    const Variant v = to_variant(variant);
    auto start = std::chrono::high_resolution_clock::now();
    ///////////////////////////////////////////////////////////////////////////
    // Launch Cholesky decomposition: A = L * L^T
    parallel_cholesky(v, matrix, static_cast<int>(N));
    ///////////////////////////////////////////////////////////////////////////
    auto stop = std::chrono::high_resolution_clock::now();
    return (stop - start).count() / 1e9;
}

}  // end of namespace cpu
