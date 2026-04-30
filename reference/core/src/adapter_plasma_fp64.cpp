#include "adapter_plasma_fp64.hpp"

#include <plasma.h>
#include <stdexcept>
#include <string>

namespace cpu
{

void plasma_potrf(std::vector<double> &A, int N)
{
    constexpr int k_plasma_max_n = 65'280;
    if (N > k_plasma_max_n)
    {
        throw std::runtime_error(
            "plasma_dpotrf: skipped to avoid PLASMA descriptor int32 overflow at N=" + std::to_string(N)
            + " (max supported with default nb=256: " + std::to_string(k_plasma_max_n) + ")");
    }

    // PLASMA is column-major. Our buffer is row-major and the matrix is
    // symmetric, so we can pass it through unchanged and ask PLASMA to write
    // its result into the upper triangle of its column-major view
    const int info = plasma_dpotrf(PlasmaUpper, N, A.data(), N);
    if (info != 0)
    {
        throw std::runtime_error("plasma_dpotrf failed with info=" + std::to_string(info));
    }
}

}  // end of namespace cpu
