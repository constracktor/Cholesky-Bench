# Cholesky-Bench
<img align="right" width="20%" src="/images/cholesky_bench_logo.jpeg">

Cholesky-Bench benchmarks right-looking tiled Cholesky factorization across a spectrum of parallelization strategies, from classical fork-join to fully asynchronous task graphs, covering multiple parallelism models including state-of-the-art asynchronous many-task runtimes. Currently OpenMP and [HPX](https://github.com/TheHPXProject/hpx) are supported, each with five parallel implementations. A non-tiled parallel reference using LAPACKE and optionally [PLASMA](https://github.com/icl-utk-edu/plasma) is also included as a baseline.

## Variants

### OpenMP (`openmp/`)

| Mode | Description |
|------|-------------|
| `for_naive` | Naive parallel for loop with dynamic schedule for trailing-update |
| `for_collapse` | Collapsed parallel for loop (optional dynamic schedule for trailing-update with LLVM) |
| `task_naive` | OpenMP tasking without dependencies (OpenMP 3.0) |
| `task_depend` | OpenMP tasking with `depend` clauses (OpenMP 4.0) |
| `task_prio` | OpenMP tasking with `priority` clauses (OpenMP 4.5) (requires `OMP_MAX_TASK_PRIORITY` to be set at runtime) |

### HPX (`hpx/`)

| Mode | Description |
|------|-------------|
| `async_future` | Fully asynchronous tasking with dataflow using `hpx::shared_future<vector>` |
| `sync_future` | Manually synchronized tasking with dataflow using `hpx::shared_future<vector>` |
| `loop_one` | Naive fork-join with dynamic schedule for trailing-update |
| `loop_two` | Collapsed fork-join with dynamic schedule for trailing-update |
| `async_void` |  Fully asynchronous tasking with dataflow using `hpx::shared_future<void>` |

### Reference (`reference/`)

| Mode | Description |
|------|-------------|
| `lapacke` | Single threaded `LAPACKE_dpotrf` call on the full matrix; no tiling. Parallelism is delegated entirely to a threaded BLAS (OpenBLAS built with `threads=openmp`, or threaded Intel oneMKL via `ENABLE_MKL=ON`). Enabled by default; disable with `ENABLE_LAPACKE=OFF`. |
| `plasma` | Single `plasma_dpotrf` call on the full matrix (PLASMA's high-level synchronous API). PLASMA does its own tiled, OpenMP-task-based parallel Cholesky internally; tile size is left at PLASMA's built-in default. Built only when `ENABLE_PLASMA=ON`. |

This directory is the natural baseline for the OpenMP and HPX tiled implementations: the `lapacke` mode isolates the contribution of vendor-provided dense-LA parallelism, and the `plasma` mode adds a tiled-parallel competitor that uses the same OpenMP runtime as the in-house variants.

#### PLASMA descriptor int32 overflow

PLASMA 24.8.7's `plasma_desc_*_create()` routines compute their tile-storage size as `int * int` before casting to `size_t`, which silently overflows once the padded triangular tile-area exceeds `INT32_MAX`. With the default `nb=256`, the boundary is at `N=65280` (`mt=255`).

The benchmark handles this transparently:

- For sweep sizes `N` in `(65280, 65536]`, **only `plasma` is silently clamped down to 65280** for that iteration; `lapacke` runs at the full `N`. The `problem_size` column reports the original `N`, so `plasma`'s timing in this range corresponds to the 65280 compute even though the row is labelled with the input size.
- For `N > 65536` `plasma` records `nan`. `lapacke` is unaffected by the int32 ceiling and continues normally.

## Dependencies

All three implementations are built with CMake (≥ 3.23) and C++20. The OpenMP and HPX directories link against a *sequential* BLAS (parallelism is at the tile level); the `reference/` directory links against a *threaded* BLAS instead.

| Dependency | OpenMP | HPX | Reference |
|---|---|---|---|
| OpenBLAS 0.3.28 (sequential) | ✓ (default) | ✓ (default) | — |
| OpenBLAS 0.3.28 (`threads=openmp`) | — | — | ✓ (default) |
| Intel oneMKL (sequential) | optional (`ENABLE_MKL=ON`) | optional (`ENABLE_MKL=ON`) | — |
| Intel oneMKL (`intel_thread`) | — | — | optional (`ENABLE_MKL=ON`) |
| PLASMA 24.8.7 | — | — | optional (`ENABLE_PLASMA=ON`) |
| HPX 1.11.0 + jemalloc | — | ✓ | — |
| GCC 14.2.0 | ✓ | ✓ | ✓ |
| LLVM/Clang 22.1.2 | optional | — | — |

Dependencies are managed via [Spack](https://spack.io/).

## Build

From within the `openmp/`, `hpx/`, or `reference/` directory, run:

```bash
./compile.sh [gcc|llvm]   # OpenMP:    gcc (default) or llvm
./compile.sh              # HPX:       always gcc
./compile.sh              # Reference: always gcc
```

The script clears and recreates the `build/` directory, then runs CMake in Release mode followed by a parallel make.

### CMake options

These can be set as environment variables before calling `compile.sh`:

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_VALIDATION` | `OFF` | After each factorization, compute the relative residual ‖A − LL^T‖_F / ‖A‖_F and warn if it exceeds 1e-10. In `openmp/` and `hpx/`, mutually exclusive with `DISABLE_COMPUTATION`. |
| `DISABLE_COMPUTATION` | `OFF` | *(`openmp/` and `hpx/` only)* Replace all BLAS/tile-generation calls with no-ops. The task graph and loops remain intact, so scheduling overhead can be measured in isolation. |
| `ENABLE_DYNAMIC_SCHEDULE` | `OFF` | *(`openmp/` only)* Use `schedule(dynamic,1)` on the trailing-update worksharing loops in `for_collapse`. Requires the LLVM toolchain; rejected at compile time with GCC. |
| `ENABLE_MKL` | `OFF` | Link against Intel oneMKL instead of OpenBLAS. In `openmp/` and `hpx/` this is the *sequential* MKL; in `reference/` it is the *threaded* MKL. |
| `ENABLE_PLASMA` | `OFF` | *(`reference/` only)* Also build the PLASMA `plasma_dpotrf` variant. Adds a `plasma` column alongside `lapacke` in the runtime output. |
| `ENABLE_LAPACKE` | `ON` | *(`reference/` only)* Run the `lapacke` mode at runtime. Set `OFF` to skip it (e.g. when only `plasma` is wanted). Linking is unchanged either way — PLASMA and validation still need cblas/lapacke symbols. |

**Examples:**

```bash
# OpenMP: GCC build with validation enabled
ENABLE_VALIDATION=ON ./compile.sh gcc

# OpenMP: LLVM build with dynamic scheduling
ENABLE_DYNAMIC_SCHEDULE=ON ./compile.sh llvm

# HPX: measure pure scheduling overhead
DISABLE_COMPUTATION=ON ./compile.sh

# Reference: threaded MKL baseline
ENABLE_MKL=ON ./compile.sh

# Reference: also build the PLASMA tiled-Cholesky variant
ENABLE_PLASMA=ON ./compile.sh

# Reference: PLASMA only, skip the LAPACKE_dpotrf column at runtime
ENABLE_LAPACKE=OFF ENABLE_PLASMA=ON ./compile.sh
```

## Run

### Directly

```bash
# OpenMP
OMP_NUM_THREADS=128 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./build/cholesky_openmp \
  --loop 1 --size_start 1024 --size_stop 65536 \
  --tiles_start 64 --tiles_stop 64

# HPX
./build/cholesky_hpx \
  --hpx:threads=128 \
  --loop=1 --size_start=1024 --size_stop=65536 \
  --tiles_start=64 --tiles_stop=64

# Reference (parallel BLAS, no tiling)
OMP_NUM_THREADS=128 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./build/cholesky_reference \
  --loop 1 --size_start 1024 --size_stop 65536
```

### Via SLURM

All three directories contain a `run.sh` that is a ready-to-submit SLURM batch script (128 CPUs, exclusive node, 144-hour wall time):

```bash
sbatch openmp/run.sh             # gcc runtime (default)
sbatch openmp/run.sh llvm        # llvm runtime
sbatch hpx/run.sh
sbatch reference/run.sh          # gcc runtime; defaults to N=65280 (see PLASMA boundary note)
```

### Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--loop` / `--loop=` | 1 | Number of timed repetitions per configuration |
| `--size_start` / `--size_stop` | 32 / 128 | Problem size range (doubled each step) |
| `--tiles_start` / `--tiles_stop` | 16 / 32 | Tile count range (doubled each step). Accepted but ignored by the `reference/` binary, which has no tiling axis. |

## Output

Results are appended to a text file in the working directory:

```
runtimes_openmp_cholesky_<suffix>.txt
runtimes_hpx_cholesky_<suffix>.txt
runtimes_reference_cholesky_<suffix>.txt
```

The suffix encodes which dimension is swept: `tile_` if tiles vary, `size_` if size varies, followed by the loop count. The file uses `;`-separated columns:

The `reference/` binary reports a `lapacke` column (suppressed by `ENABLE_LAPACKE=OFF`) plus a `plasma` column when built with `ENABLE_PLASMA=ON`, with `tile_size = problem_size` and `n_tiles = 1`, so its runtime files merge cleanly with the tiled benchmarks on the `problem_size` key:

## Repository structure

```
.
├── .clang-format           # repo-wide style; governs all three subtrees
├── CMakeLists.txt          # top-level coordinator (formatting only; LANGUAGES NONE)
├── openmp/
│   ├── CMakeLists.txt
│   ├── compile.sh          # build script (gcc or llvm)
│   ├── run.sh              # SLURM job script
│   ├── main.cpp
│   └── core/
│       ├── include/
│       │   ├── cholesky_factor.hpp
│       │   ├── functions.hpp
│       │   ├── tile_generation.hpp
│       │   ├── validate.hpp
│       │   └── adapter_cblas_fp64.hpp
│       └── src/
│           ├── cholesky_factor.cpp
│           ├── functions.cpp
│           ├── tile_generation.cpp
│           ├── validate.cpp
│           └── adapter_cblas_fp64.cpp
├── hpx/
│   ├── CMakeLists.txt
│   ├── compile.sh          # build script (gcc only)
│   ├── run.sh              # SLURM job script
│   ├── main.cpp
│   └── core/
│       ├── include/
│       │   ├── cholesky_factor.hpp
│       │   ├── functions.hpp
│       │   ├── tile_generation.hpp
│       │   ├── validate.hpp
│       │   └── adapter_cblas_fp64.hpp
│       └── src/
│           ├── cholesky_factor.cpp
│           ├── functions.cpp
│           ├── tile_generation.cpp
│           ├── validate.cpp
│           └── adapter_cblas_fp64.cpp
└── reference/
    ├── CMakeLists.txt
    ├── compile.sh          # build script (gcc only)
    ├── run.sh              # SLURM job script
    ├── main.cpp
    └── core/
        ├── include/
        │   ├── cholesky_factor.hpp
        │   ├── functions.hpp
        │   ├── matrix_generation.hpp
        │   ├── adapter_plasma_fp64.hpp  # only used when ENABLE_PLASMA=ON
        │   ├── validate.hpp
        │   └── adapter_cblas_fp64.hpp
        └── src/
            ├── cholesky_factor.cpp
            ├── functions.cpp
            ├── matrix_generation.cpp
            ├── adapter_plasma_fp64.cpp  # only built when ENABLE_PLASMA=ON
            ├── validate.cpp
            └── adapter_cblas_fp64.cpp
```

When `ENABLE_LAPACKE=OFF`, `adapter_cblas_fp64.cpp` and `validate.cpp` are still compiled and linked (they share cblas/lapacke symbols with PLASMA's BLAS dependency); only the runtime dispatch of the `lapacke` mode is skipped.

## Formatting

A repository-wide [`.clang-format`](.clang-format) governs all subtrees. The top-level [`CMakeLists.txt`](CMakeLists.txt) wires up `clang-format` and `cmake-format` targets via [Format.cmake](https://github.com/TheLartians/Format.cmake); configure once from the repo root and use the targets:

```bash
cmake -B build-fmt
cmake --build build-fmt --target check-clang-format   # CI-style check
cmake --build build-fmt --target fix-clang-format     # apply formatting
```

Each subproject (`openmp/`, `hpx/`, `reference/`) is its own standalone CMake project with its own dependencies, so the top-level `CMakeLists.txt` only handles formatting. The actual builds still happen from inside each subdirectory via its `compile.sh`.

## Contributing

We would be happy to expand Cholesky-Bench to additional asynchronous many-task (AMT) runtimes. If you would like to add an implementation, feel free to open a pull request.

## How to cite

If you use Cholesky-Bench in your research, please cite:

```bibtex
@misc{coming_soon}
```
