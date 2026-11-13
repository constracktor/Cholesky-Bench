# StarPU

## Variants

| Mode | Description |
|------|-------------|
| `stf` | Fully asynchronous tiled factorization using [StarPU](https://starpu.gitlabpages.inria.fr/) in sequential-task-flow style: tasks are submitted in program order and dependencies are inferred from the tile access modes (R/RW). The scheduler is selected at runtime via `STARPU_SCHED` (default `lws`). |
| `stf_sync` | Barrier-synchronized variant of `stf`: the same tasks, but `starpu_task_wait_for_all()` follows each of the POTRF, TRSM and SYRK/GEMM phases of every step (analogue of HPX `sync_future`) |

## Dependencies

Common dependencies are listed in the [top-level README](../README.md#dependencies).

| Dependency | |
|---|---|
| StarPU 1.4.7 (CPU only, no MPI) | ✓ |

## Build

```bash
./compile.sh   # always gcc
```

## Run

```bash
STARPU_SCHED=lws ./build/cholesky_starpu \
  --threads 128 \
  --loop 1 --size_start 1024 --size_stop 65536 \
  --tiles_start 64 --tiles_stop 64
```

The binary takes its scheduler from the `STARPU_SCHED` environment variable (default `lws`) and prints `[starpu] scheduler=<name> workers=<n>` once at startup; `--threads` sets the number of CPU workers (`STARPU_NCPU` overrides it, and the `threads` column always reports the actual worker count).

For `n_tiles >= 512` the number of submitted tasks (~n³/6) becomes huge; bound memory by throttling submission with `STARPU_LIMIT_MAX_SUBMITTED_TASKS` and `STARPU_LIMIT_MIN_SUBMITTED_TASKS` (see `run.sh`). With 8 or more threads the main thread reserves a core for task submission; set `STARPU_MAIN_THREAD_BIND=0` to disable this.

```bash
sbatch run.sh
```
