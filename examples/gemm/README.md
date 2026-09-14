# A complete tiled GEMM

`kernel.jl` implements the complete global → shared → registers → MMA →
global path using Tylo. It uses `cp.async` and `mma.sync.m16n8k16`, so it can
execute on compute capability 8.0 and later, including GB10. It does not use
Hopper WGMMA or datacenter Blackwell tcgen05 MMA.

The reusable pieces are memory layouts/views, copy plans, MMA atoms,
per-warp/CTA ownership, and accumulator arithmetic. The example owns the
launch grid, shared allocation and copy pipeline. No reference kernel is
imported and no pre-existing GEMM implementation is called.

The central operations are:

```julia
copy_async!(config.ac, shared_a, global_a_window, tid)
copy_async!(config.bc, shared_b, global_b_window, tid)
commit_copies()
wait_copies(Val(1)) # for a two-stage pipeline with the next stage still pending
sync_threads()
acc = mma(config.plan, shared_a, shared_b, acc, tid)
sync_threads()    # release shared storage for reuse
result = map(x -> max(alpha*x, 0f0), acc)
store!(config.plan, output_window, result, tid)
```

The actual loop drains the last copy group with `wait_copies(Val(0))`.

## Run

From the repository root, with the GPU test environment instantiated:

```sh
julia --project=test examples/gemm/run.jl
julia --project=test test/runtests.jl
compute-sanitizer --tool racecheck --error-exitcode 86 \
  julia --project=test test/tools/sanitize.jl
```

Use a Compute Sanitizer version compatible with the CUDA compiler/runtime
selected by CUDACore. `sanitize.jl` exercises executable tile kernels, including GEMM, TMA and
row operations, and requires a GPU; it cannot pass by skipping runtime execution.

## Scope and evidence

Inputs are typed BF16 or FP16 arrays, with FP32 accumulation/output. A is
physically K-contiguous and B uses ordinary column-major `(K,N)` storage.
The logical API always computes `A * B`. Tests exercise padded leading
strides, rectangular grids, K tiles of 16/32/64, different warp arrangements,
one- and two-stage pipelines, one iteration through repeated buffer reuse,
and a fused scaling/ReLU epilogue. Output padding is checked for corruption.

The aligned path requires full tiles; the bounded path below handles arbitrary
positive dimensions. Split-K and scheduling autotuning remain outside the
example. The measurements do not claim parity with cuBLAS throughput.
Timing output excludes compilation and allocation and reports the median
CUDA-graph replay time. All configurations compile before measurement;
measurement order alternates after six warmup rounds. Compare configurations only on the same GPU/session.

## Boundary tiles

Run `julia --project=test examples/gemm/run.jl 65 97 73` for a ragged case.
The demo selects bounded copies and stores automatically. Direct kernel callers
use `gemm_config(...; bounds=true)` and ceiling-divided launch dimensions.
See [the bounds contract](../../docs/src/boundaries.md) for partial vectors,
zero-fill, shared-memory synchronization, and compact/padded timing comparisons.
