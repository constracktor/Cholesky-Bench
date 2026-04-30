#include "cholesky_factor.hpp"

#include "adapter_cblas_fp64.hpp"
#ifdef ENABLE_PLASMA
#include "adapter_plasma_fp64.hpp"
#endif

#include <stdexcept>

namespace cpu
{

void parallel_cholesky(Variant variant, std::vector<double> &matrix, int N)
{
    switch (variant)
    {
        case Variant::reference: lapacke_potrf(matrix, N); return;

        case Variant::plasma:
#ifdef ENABLE_PLASMA
            plasma_potrf(matrix, N);
            return;
#else
            throw std::invalid_argument("Variant 'plasma' requested but the binary was built without ENABLE_PLASMA=ON");
#endif
    }
}

}  // end of namespace cpu
