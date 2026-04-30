# Cholesky-Bench

Cholesky-Bench benchmarks right-looking tiled Cholesky factorization from fork-join to asynchronous tasks across several parallelism models, currently comparing OpenMP and HPX implementations side by side. A non-tiled parallel-BLAS reference is also included as a baseline.

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
| `reference` | Single threaded `LAPACKE_dpotrf` call on the full matrix; no tiling. Parallelism is delegated entirely to a threaded BLAS (OpenBLAS built with `threads=openmp`, or threaded Intel oneMKL via `ENABLE_MKL=ON`). |
| `plasma` | Single `plasma_dpotrf` call on the full matrix (PLASMA's high-level synchronous API). PLASMA does its own tiled, OpenMP-task-based parallel Cholesky internally; tile size is left at PLASMA's built-in default. Built only when `ENABLE_PLASMA=ON`. |
| `plasma_tile` | `plasma_omp_dpotrf` over a manually-built `plasma_desc_t` (PLASMA's asynchronous tile interface). Allocates an *uninitialised* general (full N×N) tile-layout backing store in user code and wraps it via `plasma_desc_general_init` — PLASMA's `_create` routines never run. Leaving the buffer uninitialised lets `plasma_omp_dge2desc` first-touch each tile from its consuming core, so pages land on the right NUMA node and a chunk of the runtime gap with `plasma` closes. Built only when `ENABLE_PLASMA=ON`. |

This directory is the natural baseline for the OpenMP and HPX tiled implementations: the `reference` mode isolates the contribution of vendor-provided dense-LA parallelism, and the two `plasma*` modes add true tiled-parallel competitors that use the same OpenMP runtime as the in-house variants.

#### PLASMA descriptor int32 overflow

PLASMA 24.8.7's `plasma_desc_*_create()` routines compute their tile-storage size as `int * int` before casting to `size_t`, which silently overflows once the padded tile-area exceeds `INT32_MAX`:

| Path | Boundary (default `nb=256`) | Behaviour past the boundary |
|------|------------------------------|------------------------------|
| `plasma` (high-level, triangular descriptor) | `N > 65280` | Skipped before invoking PLASMA. Records `nan` instead of triggering PLASMA's multi-line `malloc() failed` diagnostic. |
| `plasma_tile` (tile API, user-allocated general buffer) | `N > 46080` | Skipped before invoking PLASMA. The user-allocated buffer avoids `_create`'s malloc-overflow, but PLASMA does additional int32 tile-offset arithmetic *during execution* of `plasma_omp_dpotrf`, which segfaults past this boundary. The guard makes the failure clean. |

Patching `(size_t)` casts into `control/descriptor.c` and the tile-offset code in the spack PLASMA package removes both ceilings, and the guards become no-ops.

## Dependencies

All three implementations are built with CMake (≥ 3.23) and C++20. The OpenMP and HPX directories link against a *sequential* BLAS (parallelism is at the tile level); the `reference/` directory links against a *threaded* BLAS instead.

| Dependency | OpenMP | HPX | Reference |
|---|---|---|---|
| OpenBLAS 0.3.28 (sequential) | ✓ (default) | ✓ (default) | — |
| OpenBLAS 0.3.28 (`threads=openmp`) | — | — | ✓ (default) |
| Intel oneMKL (sequential) | optional (`ENABLE_MKL=ON`) | optional (`ENABLE_MKL=ON`) | — |
| Intel oneMKL (`intel_thread`) | — | — | optional (`ENABLE_MKL=ON`) |
| PLASMA | — | — | optional (`ENABLE_PLASMA=ON`) |
| HPX 1.11.0 + jemalloc | — | ✓ | — |
| GCC 14.2.0 | ✓ | ✓ | ✓ |
| LLVM/Clang 22.1.2 | optional | — | — |

Dependencies are managed via [Spack](https://spack.io/). The compile scripts auto-detect the host system and load the correct Spack environment.

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
| `ENABLE_PLASMA` | `OFF` | *(`reference/` only)* Also build the PLASMA tiled-Cholesky variant. Adds a `plasma` column alongside `reference` in the runtime output. |

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
sbatch reference/run.sh
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

```
threads;problem_size;tile_size;n_tiles;for_collapse;for_naive;task_naive;task_depend
128;65536;1024;64;3.14;3.21;2.98;2.87
```

The `reference/` binary reports a `reference` column (and `plasma` + `plasma_tile` columns when built with `ENABLE_PLASMA=ON`), with `tile_size = problem_size` and `n_tiles = 1`, so its runtime files merge cleanly with the tiled benchmarks on the `problem_size` key:

```
threads;problem_size;tile_size;n_tiles;reference;plasma;plasma_tile
128;65280;65280;1;2.71;68.12;71.30
```

The same lines are also printed to stdout.

## Repository structure

```
.
├── openmp/
│   ├── CMakeLists.txt
│   ├── CMakePresets.json
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
│   ├── CMakePresets.json
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
    ├── CMakePresets.json
    ├── compile.sh          # build script (gcc only)
    ├── run.sh              # SLURM job script
    ├── main.cpp
    └── core/
        ├── include/
        │   ├── cholesky_factor.hpp
        │   ├── functions.hpp
        │   ├── matrix_generation.hpp
        │   ├── plasma_factor.hpp     # only used when ENABLE_PLASMA=ON
        │   ├── validate.hpp
        │   └── adapter_cblas_fp64.hpp
        └── src/
            ├── cholesky_factor.cpp
            ├── functions.cpp
            ├── matrix_generation.cpp
            ├── plasma_factor.cpp     # only built when ENABLE_PLASMA=ON
            ├── validate.cpp
            └── adapter_cblas_fp64.cpp
```

## Contributing

We would be happy to expand Cholesky-Bench to additional asynchronous many-task (AMT) runtimes. If you have an implementation you would like to add, feel free to open a pull request.

## How to cite

If you use Cholesky-Bench in your research, please cite:

```bibtex
@misc{coming_soon}
```
