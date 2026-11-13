# Standard C++

A pure standard-library port of the HPX implementation, built for an apples-to-apples comparison: tasking runs on a custom work-stealing thread pool and fork-join loops run on the same pool. There is no runtime dependency beyond a C++20 toolchain. The five modes line up one-to-one with the HPX variants.

The pool has a **fixed number of worker threads**, mirroring HPX's `--hpx:threads=N` model: one thread per CPU in the process's affinity mask (e.g. as set by `srun`/`taskset`), overridable with the environment variable `CHOLESKY_THREADS=N`. Workers are pinned to CPUs and the whole task graph is multiplexed onto them; no thread is created per task. `hpx::dataflow` is continuation based: a task counts its unsatisfied inputs and is queued by whichever thread satisfies the last one, so workers never block on futures. Each worker owns two deques (normal and high priority). Tasks and tiles are placed on the worker that owns the tile they write (rows are spread over NUMA domains, columns over the workers of a domain). A worker pops its own newest task and, when idle, steals the oldest task of another worker, trying workers of its own NUMA domain first; idle workers spin briefly and then sleep. The fork-join modes use the same pool with dynamic chunk claiming, so tasking and fork-join share one thread budget.

## Variants

| Mode | Description |
|------|-------------|
| `async_future` | Fully asynchronous tasking (pool + continuation dataflow on `std_backend::future<vector>`) |
| `sync_future` | Manually synchronized tasking (pool + continuation dataflow on `std_backend::future<vector>`) |
| `loop_one` | Naive fork-join: `parallel_for` over rows, sequential trailing-update sweep |
| `loop_two` | Collapsed fork-join: nested `parallel_for` over the trailing-update |
| `async_void` | Fully asynchronous tasking (pool + continuation dataflow on `std_backend::future<void>`) |

> Note on scheduling: like HPX, the std pool never blocks a worker on an unready dependency (continuations are used instead), and it steals work between workers. Thread *count* and memory footprint are therefore directly comparable to HPX; remaining differences come from the runtimes themselves (e.g. HPX's lightweight thread suspension and its scheduler policies).

## Dependencies

No dependencies beyond the [common ones](../README.md#dependencies).

## Build

```bash
./compile.sh   # always gcc
```

## Run

```bash
srun --cpu-bind=cores ./build/cholesky_std \
  --loop=1 --size_start=1024 --size_stop=65536 \
  --tiles_start=64 --tiles_stop=64
```

The worker count follows the CPU affinity mask; set `CHOLESKY_THREADS=N` to override it.

```bash
sbatch run.sh
```
