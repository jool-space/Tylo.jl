# Tylo

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jool-space.github.io/Tylo.jl/dev/)
[![Build Status](https://github.com/jool-space/Tylo.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jool-space/Tylo.jl/actions/workflows/CI.yml)

Composable tile programming for NVIDIA GPUs in Julia.

Tylo provides logical memory views, distributed register values, and specific
copy and matrix operations that connect them. PTX.jl supplies instruction
bindings. The kernel author controls the launch, work assignment, allocation
and pipeline. The API is experimental.

## Read the design and code

Start with the [manual](docs/src/index.md). For a focused reading path:

1. [Design and boundaries](docs/src/design.md): storage, ownership, operation
   plans, completion, and the tradeoffs still worth challenging.
2. [Layouts](docs/src/layouts.md) and [fragments](docs/src/rows.md): static
   notation, logical axes, register distribution, scalar broadcast and reductions.
3. [Walk through GEMM](docs/src/gemm.md): one complete consumer from host planning
   through shared copies and matrix instructions to the epilogue.
4. [Read the implementation](docs/src/codebase.md): source files, specialization,
   dispatch paths and the tests that establish their contracts.
5. [Current status](docs/src/validation.md): supported combinations, hardware
   coverage and performance limits.

The [API reference](docs/src/api.md) and [dated validation history](docs/src/validation-history.md)
are separate from the conceptual explanation. Build this working tree's manual
with the instructions below; the published documentation follows deployed commits.

## The core distinction

`Fragment(values, ownership)` separates each thread's local values from their
logical coordinates. Memory layouts separately map those coordinates to storage.
A thread's values can form a row, a scattered patch, or part of an MMA result.

Inside a kernel, ready register fragments support ordinary scalar composition:

```julia
m = maximum(f; dims=2)  # requires a supported collective for this ownership
w = exp.(f .- m)       # generic scalar broadcast, for finite input here
y = w ./ sum(w; dims=2)
z = ptx"fma.rn.f32".(y, 2f0, 1f0)
```

Elementwise operations preserve ownership. Reductions need an implemented
communication recipe. Logical `permutedims` changes coordinates; producing a
different instruction's register arrangement may require an explicit conversion.
Memory tiles and pending results are not implicitly loaded or waited on by
broadcast. See [the fragment support table](docs/src/rows.md).

## Worked consumers

- [Complete warp-MMA GEMM](examples/gemm/README.md): BF16/FP16 inputs,
  shared layouts, asynchronous copies, bounded tiles and composed epilogues.
- [Softmax](examples/softmax/README.md): lane-local, warp-striped and MMA
  reductions, with explicit masking and fully masked-group behavior.
- [Streaming attention](examples/streaming_attention/README.md): a complete
  D=64 BF16 forward kernel with online statistics and chained warp MMA on GB10.
- [TMA/WGMMA GEMM](examples/hopper/README.md): producer/consumer pipeline and
  the primitives also used by Megakernels' `HopperProjection`.
- [Datacenter FlashAttention experiment](examples/flash_attention/README.md):
  TMEM correction and epilogue replacements in a raw PTX kernel.

Warp GEMM, TMA and register operations have GB10 runtime coverage. TMEM
transfers and the datacenter attention comparison have run on a B200. WGMMA
has assembly coverage and prepared hardware tests; H100/H200 execution remains
unvalidated. The library does not yet have general CuTe/ThunderKittens
coverage, including tcgen05 MMA, arbitrary redistribution or allocation
management.

`Tylo.Layouts` contains the pure coordinate mathematics. Megakernels consumes
Tylo operations while owning task scheduling and buffer reuse. Standalone
kernels use Tylo without Megakernels; cuTile uses a separate compiler-driven
approach. The [design](docs/src/design.md) explains those boundaries.

## Development

The test project is a workspace member, so one instantiate covers the
package, its tests and its docs (Julia 1.10+; CUDACore and PTX are pulled in
by the test project):

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(; workspace=true)'
julia --project=test test/runtests.jl --jobs=4        # host and GPU tiers
julia --project=test test/runtests.jl host            # host only
julia --project=test test/runtests.jl gpu/gemm gpu/atoms
julia --project=test examples/gemm/run.jl 65 97 73
```

Tests run in parallel with `ParallelTestRunner`. `host/` needs no GPU.
`gpu/` files need the CUDA compiler; their assembly checks always run and
their runtime sections run when the device satisfies the file's
`# TEST_TARGET: cc>=8.0`-style banner (`test/targets.jl`). Set
`TYLO_REQUIRE_GPU_RUNTIME=true` to fail instead of skipping without a GPU
and `TYLO_EVIDENCE=<dir>` to save PTX and cubins.

Build the manual locally, including its host doctests:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Open `docs/build/index.html`. GPU excerpts in the manual identify their required
surrounding setup; the complete programs live in `examples/`.
