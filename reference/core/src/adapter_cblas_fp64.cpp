#include "adapter_cblas_fp64.hpp"

#ifdef ENABLE_MKL
// MKL CBLAS / LAPACKE
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

void lapacke_potrf(vector &A, const int N) { LAPACKE_dpotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N); }
