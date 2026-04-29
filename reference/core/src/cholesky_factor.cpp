#include "cholesky_factor.hpp"

#include "adapter_cblas_fp64.hpp"
#ifdef ENABLE_PLASMA
#include "plasma_factor.hpp"
#endif

#include <stdexcept>

namespace cpu
{

void parallel_blas_cholesky(Variant variant, std::vector<double> &A, int N)
{
    switch (variant)
    {
        case Variant::reference:
            // Single threaded LAPACKE call on the full matrix; the BLAS
            // library dispatches work across the available threads.
            potrf(A, N);
            return;

        case Variant::plasma:
#ifdef ENABLE_PLASMA
            plasma_cholesky(A, N);
            return;
#else
            throw std::invalid_argument(
                "Variant 'plasma' requested but the binary was built without ENABLE_PLASMA=ON");
#endif

        case Variant::plasma_tile:
#ifdef ENABLE_PLASMA
            plasma_tile_cholesky(A, N);
            return;
#else
            throw std::invalid_argument(
                "Variant 'plasma_tile' requested but the binary was built without ENABLE_PLASMA=ON");
#endif
    }
}

}  // end of namespace cpu
