# Cholesky-Bench

<img align="right" width="20%" src="/images/cholesky_bench_logo.jpeg">

Cholesky-Bench benchmarks right-looking tiled Cholesky factorization across a spectrum of parallelization strategies, from classical fork-join to fully asynchronous task graphs, covering multiple parallelism models including state-of-the-art asynchronous many-task runtimes. A non-tiled parallel reference using LAPACKE and optionally [PLASMA](https://github.com/icl-utk-edu/plasma) is also included as a baseline.

## Runtimes

Each runtime lives in its own directory and is a standalone CMake project. Its `README.md` lists its variants, its extra dependencies, and runtime-specific options and run settings.

| Directory | Runtime |
|---|---|
| `openmp/` | OpenMP (fork-join and tasking) |
| `hpx/` | [HPX](https://github.com/TheHPXProject/hpx) |
| `cpp_standard/` | Standard C++ (custom thread pool) |
| `starpu/` | [StarPU](https://starpu.gitlabpages.inria.fr/) |
| `ttg/` | [TTG](https://github.com/TESSEorg/ttg) (PaRSEC backend) |
| `reference/` | Non-tiled LAPACKE / PLASMA baseline |

Only the directories present in your checkout are available; the runtimes are added to the repository one by one.

## Dependencies

All runtimes are built with CMake (≥ 3.23) and C++20 and managed via [Spack](https://spack.io/). The tiled runtimes link against a *sequential* BLAS (parallelism is at the tile level); the `reference/` directory links against a *threaded* BLAS instead.

| Dependency | Tiled runtimes | Reference |
|---|---|---|
| OpenBLAS 0.3.28 (sequential) | ✓ (default) | — |
| OpenBLAS 0.3.28 (`threads=openmp`) | — | ✓ (default) |
| Intel oneMKL (sequential) | optional (`ENABLE_MKL=ON`) | — |
| Intel oneMKL (`intel_thread`) | — | optional (`ENABLE_MKL=ON`) |
| GCC 14.2.0 | ✓ | ✓ |

Runtime-specific dependencies (HPX, StarPU, TTG, LLVM, PLASMA, ...) are listed in each runtime's README.

## Build

From within a runtime directory, run:

```bash
./compile.sh
```

The script clears and recreates the `build/` directory, then runs CMake in Release mode followed by a parallel make. Some runtimes accept extra arguments (for example `openmp/` takes `gcc` or `llvm`); see the runtime's README.

### CMake options

These can be set as environment variables before calling `compile.sh`:

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_VALIDATION` | `OFF` | After each factorization, compute the relative residual ‖A − LL^T‖_F / ‖A‖_F and warn if it exceeds 1e-10. In the tiled runtimes, mutually exclusive with `DISABLE_COMPUTATION`. |
| `DISABLE_COMPUTATION` | `OFF` | *(tiled runtimes only)* Replace all BLAS/tile-generation calls with no-ops. The task graph and loops remain intact, so scheduling overhead can be measured in isolation. |
| `ENABLE_MKL` | `OFF` | Link against Intel oneMKL instead of OpenBLAS. In the tiled runtimes this is the *sequential* MKL; in `reference/` it is the *threaded* MKL. |

Further options are described in the runtime READMEs.

```bash
# GCC build with validation enabled
ENABLE_VALIDATION=ON ./compile.sh

# Measure pure scheduling overhead
DISABLE_COMPUTATION=ON ./compile.sh
```

## Run

Every runtime directory contains a `run.sh`, a ready-to-submit SLURM batch script (128 CPUs, exclusive node, 144-hour wall time):

```bash
sbatch <runtime>/run.sh
```

The binaries can also be started directly from the runtime directory (`./build/cholesky_<runtime>`). The thread count and pinning are set differently per runtime, so the exact commands are given in each runtime's README.

### Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--loop` / `--loop=` | 1 | Number of timed repetitions per configuration |
| `--size_start` / `--size_stop` | 32 / 128 | Problem size range (doubled each step) |
| `--tiles_start` / `--tiles_stop` | 16 / 32 | Tile count range (doubled each step). Accepted but ignored by the `reference/` binary, which has no tiling axis. |

## Output

Results are appended to a text file in the working directory:

```
runtimes_<runtime>_cholesky_<suffix>.txt
```

The suffix encodes which dimension is swept: `tile_` if tiles vary, `size_` if size varies, followed by the loop count. The file uses `;`-separated columns: `threads;problem_size;tile_size;n_tiles` followed by one runtime column per mode.

The `reference/` binary reports a `lapacke` column plus a `plasma` column when built with `ENABLE_PLASMA=ON`, with `tile_size = problem_size` and `n_tiles = 1`, so its runtime files merge cleanly with the tiled benchmarks on the `problem_size` key.

## Formatting

A repository-wide [`.clang-format`](.clang-format) governs all subtrees. The top-level [`CMakeLists.txt`](CMakeLists.txt) wires up `clang-format` and `cmake-format` targets via [Format.cmake](https://github.com/TheLartians/Format.cmake); configure once from the repo root and use the targets:

```bash
cmake -B build-fmt
cmake --build build-fmt --target check-clang-format   # CI-style check
cmake --build build-fmt --target fix-clang-format     # apply formatting
```

Each runtime directory is its own standalone CMake project with its own dependencies, so the top-level `CMakeLists.txt` only handles formatting. The actual builds still happen from inside each subdirectory via its `compile.sh`.

## Contributing

We would be happy to expand Cholesky-Bench to additional asynchronous many-task (AMT) runtimes. If you would like to add an implementation, feel free to open a pull request.

## How to cite

If you use Cholesky-Bench in your research, please cite:

```bibtex
@misc{Strack2026_choleskybench,
      title={From Fork-Join to Asynchronous Tasks: Parallelizing Tiled Cholesky Decomposition with OpenMP and HPX},
      author={Alexander Strack and Alexander Van Craen and Dirk Pflüger},
      year={2026},
      eprint={2606.11937},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2606.11937},
}
```
