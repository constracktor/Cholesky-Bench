# TTG

## Variants

| Mode | Description |
|------|-------------|
| `flow` | Fully asynchronous tiled factorization expressed as a [TTG](https://github.com/TESSEorg/ttg) task graph (dispatcher → POTRF → TRSM → SYRK/GEMM), an equivalent shared-memory CPU port of the upstream [`examples/potrf`](https://github.com/TESSEorg/ttg/tree/master/examples/potrf) (device and distributed paths stripped) |
| `flow_sync` | Barrier-synchronized variant: edge-less TTG tasks operate directly on the shared tile storage, and a `ttg::fence()` follows each of the POTRF, TRSM and SYRK/GEMM phases of every step (analogue of HPX `sync_future`) |

`flow` assigns DPLASMA task priorities (favoring the critical path; saturated at `INT_MAX` for very large tile counts). Both modes interleave the tile pages round-robin over all online NUMA nodes during matrix generation (Linux only, via `set_mempolicy`; silently skipped if unavailable) so memory bandwidth is not limited to one node.

## Dependencies

Common dependencies are listed in the [top-level README](../README.md#dependencies).

| Dependency | |
|---|---|
| TTG (PaRSEC backend) | ✓ |
| Open MPI, Boost 1.86.0 (`+serialization`) | ✓ (loaded by `compile.sh`) |

## Build

```bash
./compile.sh   # always gcc
```

## Run

```bash
./build/cholesky_ttg \
  --threads=128 \
  --loop=1 --size_start=1024 --size_stop=65536 \
  --tiles_start=64 --tiles_stop=64
```

```bash
sbatch run.sh
```
