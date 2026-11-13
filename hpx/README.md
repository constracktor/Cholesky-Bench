# HPX

## Variants

| Mode | Description |
|------|-------------|
| `async_future` | Fully asynchronous tasking with dataflow using `hpx::shared_future<vector>` |
| `sync_future` | Manually synchronized tasking with dataflow using `hpx::shared_future<vector>` |
| `loop_one` | Naive fork-join with dynamic schedule for trailing-update |
| `loop_two` | Collapsed fork-join with dynamic schedule for trailing-update |
| `async_void` | Fully asynchronous tasking with dataflow using `hpx::shared_future<void>` |

## Dependencies

Common dependencies are listed in the [top-level README](../README.md#dependencies).

| Dependency | |
|---|---|
| HPX 2.0.0 + jemalloc | ✓ (not yet in upstream Spack; `.github/spack/add-hpx-2.0.0.sh` adds the version to the local recipe) |

## Build

```bash
./compile.sh   # always gcc
```

```bash
# Measure pure scheduling overhead
DISABLE_COMPUTATION=ON ./compile.sh
```

## Run

```bash
./build/cholesky_hpx \
  --hpx:threads=128 \
  --loop=1 --size_start=1024 --size_stop=65536 \
  --tiles_start=64 --tiles_stop=64
```

```bash
sbatch run.sh
```
