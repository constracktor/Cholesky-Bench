# Reference

A non-tiled parallel baseline for the tiled runtimes.

## Variants

| Mode | Description |
|------|-------------|
| `lapacke` | Single threaded `LAPACKE_dpotrf2` call on the full matrix; no tiling. Parallelism is delegated entirely to a threaded BLAS (OpenBLAS built with `threads=openmp`, or threaded Intel oneMKL via `ENABLE_MKL=ON`). Enabled by default; disable with `ENABLE_LAPACKE=OFF`. |
| `plasma` | Single `plasma_dpotrf` call on the full matrix (PLASMA's high-level synchronous API). PLASMA does its own tiled, OpenMP-task-based parallel Cholesky internally; tile size is left at PLASMA's built-in default. Built only when `ENABLE_PLASMA=ON`. |

The `lapacke` mode isolates the contribution of vendor-provided dense-LA parallelism, and the `plasma` mode adds a tiled-parallel competitor that uses the same OpenMP runtime as the in-house variants.

## Dependencies

Common dependencies are listed in the [top-level README](../README.md#dependencies). This directory links against a *threaded* BLAS.

| Dependency | |
|---|---|
| PLASMA 24.8.7 | optional (`ENABLE_PLASMA=ON`) |

## Build

```bash
./compile.sh   # always gcc
```

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_PLASMA` | `OFF` | Also build the PLASMA `plasma_dpotrf` variant. Adds a `plasma` column alongside `lapacke` in the runtime output. |
| `ENABLE_LAPACKE` | `ON` | Run the `lapacke` mode at runtime. Set `OFF` to skip it (e.g. when only `plasma` is wanted). Linking is unchanged either way — PLASMA and validation still need cblas/lapacke symbols. |

When `ENABLE_LAPACKE=OFF`, `adapter_cblas_fp64.cpp` and `validate.cpp` are still compiled and linked (they share cblas/lapacke symbols with PLASMA's BLAS dependency); only the runtime dispatch of the `lapacke` mode is skipped.

```bash
# Threaded MKL baseline
ENABLE_MKL=ON ./compile.sh

# Also build the PLASMA tiled-Cholesky variant
ENABLE_PLASMA=ON ./compile.sh

# PLASMA only, skip the LAPACKE_dpotrf column at runtime
ENABLE_LAPACKE=OFF ENABLE_PLASMA=ON ./compile.sh
```

## Run

```bash
OMP_NUM_THREADS=128 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./build/cholesky_reference \
  --loop 1 --size_start 1024 --size_stop 65536
```

```bash
sbatch run.sh   # defaults to N=65280 (see PLASMA boundary note)
```

## PLASMA descriptor int32 overflow

PLASMA 24.8.7's `plasma_desc_*_create()` routines compute their tile-storage size as `int * int` before casting to `size_t`, which silently overflows once the padded triangular tile-area exceeds `INT32_MAX`. With the default `nb=256`, the boundary is at `N=65280` (`mt=255`).

The benchmark handles this transparently:

- For sweep sizes `N` in `(65280, 65536]`, **only `plasma` is silently clamped down to 65280** for that iteration; `lapacke` runs at the full `N`. The `problem_size` column reports the original `N`, so `plasma`'s timing in this range corresponds to the 65280 compute even though the row is labelled with the input size.
- For `N > 65536` `plasma` records `nan`. `lapacke` is unaffected by the int32 ceiling and continues normally.
