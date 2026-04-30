#include "plasma_factor.hpp"

#include <plasma.h>

#include <climits>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cpu
{
namespace
{

// PLASMA's default tile size for fp64 (typical 24.x default). We hardcode
// this rather than calling plasma_get(PlasmaNb, ...) so the overflow guard
// below stays portable across PLASMA versions. If you tune via
// plasma_set(PlasmaNb, ...) at startup, keep this matching.
constexpr int kPlasmaDefaultNb = 256;

// Pre-flight: would PLASMA's int32 multiplication for descriptor sizing
// overflow? PLASMA 24.8.7's plasma_desc_*_create routines compute the
// total tile-layout backing-store size as int*int and then cast to size_t,
// so the malloc gets a sign-extended-negative argument and fails for any
// padded total >= INT32_MAX. We replicate the math here and throw before
// invoking PLASMA, which avoids the multi-line PLASMA ERROR diagnostic on
// stderr and keeps the surrounding sweep clean.
//
// Used for both paths. The high-level path needs it because of the malloc
// inside _create; the tile path needs it because PLASMA also does int32
// tile-offset arithmetic *during execution* (segfaults at N>~46080 with the
// general descriptor and default nb), even though we allocate the buffer
// ourselves and bypass _create entirely.
void guard_descriptor_overflow(int N, int nb, bool triangular, const char *which)
{
    const long long mt = (N + nb - 1) / nb;
    const long long padded =
        triangular ? (mt * (mt + 1) / 2) * static_cast<long long>(nb) * nb
                   : mt * mt * static_cast<long long>(nb) * nb;
    if (padded > static_cast<long long>(INT_MAX))
    {
        throw std::runtime_error(
            std::string(which) + ": skipped to avoid PLASMA descriptor int32 overflow at N=" + std::to_string(N)
            + " (nb=" + std::to_string(nb) + ", mt=" + std::to_string(mt)
            + ", padded elements=" + std::to_string(padded) + " > INT32_MAX)");
    }
}

}  // anonymous namespace

void plasma_cholesky(std::vector<double> &A, int N)
{
    // High-level plasma_dpotrf allocates a triangular tile descriptor
    // internally; overflow check uses the triangular size formula.
    guard_descriptor_overflow(N, kPlasmaDefaultNb, /*triangular=*/true, "plasma_dpotrf");

    // PLASMA is column-major. Our buffer is row-major and the matrix is
    // symmetric, so we can pass it through unchanged and ask PLASMA to write
    // its result into the upper triangle of its column-major view -- that
    // upper triangle aliases the lower triangle of our row-major view, which
    // is the layout the validator (and the LAPACKE reference path) expects.
    const int info = plasma_dpotrf(PlasmaUpper, N, A.data(), N);
    if (info != 0)
    {
        throw std::runtime_error("plasma_dpotrf failed with info=" + std::to_string(info));
    }
}

void plasma_tile_cholesky(std::vector<double> &A, int N)
{
    // Pre-flight: PLASMA does int32 tile-offset arithmetic during execution
    // (not just inside _create), so the general descriptor still hits an
    // overflow ceiling at N>~46080 with the default nb. Without this guard
    // plasma_omp_dpotrf segfaults rather than failing cleanly.
    guard_descriptor_overflow(N, kPlasmaDefaultNb, /*triangular=*/false, "plasma_omp_dpotrf");

    // The tile path bypasses PLASMA's _create allocator (which has the
    // int32-multiply malloc bug) by allocating the tile-layout backing
    // store ourselves and wrapping it with plasma_desc_general_init. _init
    // performs no malloc, so the buggy multiplication is never reached.
    //
    // The buffer is *uninitialised* (new double[N], not value-initialised
    // with std::vector). Two reasons: (1) skips a multi-GB zero-init pass
    // run on the main thread, and (2) lets plasma_omp_dge2desc first-touch
    // each tile from its consuming core, so pages land on the right NUMA
    // node instead of all on the main thread's node. That's what shaves
    // time off the general-descriptor tile path here.

    const int nb = kPlasmaDefaultNb;
    const long long mt_ll = (N + nb - 1) / nb;
    const int mt = static_cast<int>(mt_ll);
    const int lm = mt * nb;  // padded leading dimension; fits int32 even for huge N

    const std::size_t tile_buf_elements = static_cast<std::size_t>(lm) * static_cast<std::size_t>(lm);

    std::unique_ptr<double[]> tile_buf(new double[tile_buf_elements]);

    plasma_desc_t descA;
    int retval =
        plasma_desc_general_init(PlasmaRealDouble, tile_buf.get(), nb, nb, lm, lm, 0, 0, N, N, &descA);
    if (retval != PlasmaSuccess)
    {
        throw std::runtime_error("plasma_desc_general_init failed with retval=" + std::to_string(retval));
    }

    // PLASMA 24.8.7's tile interface uses stack-allocated sequence/request
    // structs. Zero-init lands status=0=PlasmaSuccess, the expected
    // pre-call state.
    plasma_sequence_t sequence{};
    plasma_request_t request{};

    // Translate row-major buffer -> tile descriptor, factor in place on the
    // descriptor, translate back. Same PlasmaUpper convention as the
    // high-level path, so the resulting layout (row-major lower triangle = L)
    // matches what the validator expects.
#pragma omp parallel
#pragma omp master
    {
        plasma_omp_dge2desc(A.data(), N, descA, &sequence, &request);
        plasma_omp_dpotrf(PlasmaUpper, descA, &sequence, &request);
        plasma_omp_ddesc2ge(descA, A.data(), N, &sequence, &request);
    }

    if (sequence.status != PlasmaSuccess)
    {
        throw std::runtime_error("plasma tile sequence failed with status=" + std::to_string(sequence.status));
    }
}

}  // end of namespace cpu
