# OpenMP

## Variants

| Mode | Description |
|------|-------------|
| `for_naive` | Naive parallel for loop with dynamic schedule for trailing-update |
| `for_collapse` | Collapsed parallel for loop (optional dynamic schedule for trailing-update with LLVM) |
| `task_naive` | OpenMP tasking without dependencies (OpenMP 3.0) |
| `task_depend` | OpenMP tasking with `depend` clauses (OpenMP 4.0) |
| `task_prio` | OpenMP tasking with `priority` clauses (OpenMP 4.5) (requires `OMP_MAX_TASK_PRIORITY` to be set at runtime) |
| `task_domain` | One nested team per NUMA domain (8 domains, needs >= 16 threads, otherwise a single domain); each team runs depend-tasks for its rows, cross-domain panel tiles awaited via atomic flags |

## Dependencies

Common dependencies are listed in the [top-level README](../README.md#dependencies).

| Dependency | |
|---|---|
| LLVM/Clang 22.1.2 | optional (`./compile.sh llvm`) |

## Build

```bash
./compile.sh [gcc|llvm]   # gcc (default) or llvm
```

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_DYNAMIC_SCHEDULE` | `OFF` | Use `schedule(dynamic,1)` on the trailing-update worksharing loops in `for_collapse`. Requires the LLVM toolchain; rejected at compile time with GCC. |

```bash
# GCC build with validation enabled
ENABLE_VALIDATION=ON ./compile.sh gcc

# LLVM build with dynamic scheduling
ENABLE_DYNAMIC_SCHEDULE=ON ./compile.sh llvm
```

## Run

```bash
OMP_NUM_THREADS=128 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./build/cholesky_openmp \
  --loop 1 --size_start 1024 --size_stop 65536 \
  --tiles_start 64 --tiles_stop 64
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--modes a,b` | all | Comma-separated list of modes to run, e.g. `--modes task_naive,task_domain` |

```bash
sbatch run.sh          # gcc runtime (default)
sbatch run.sh llvm     # llvm runtime (run.sh sets OMP_PROC_BIND=spread,close, OMP_PLACES=cores)
```
