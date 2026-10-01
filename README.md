# Tylo

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.jool.space/Tylo.jl/dev/)
[![Build Status](https://github.com/jool-space/Tylo.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jool-space/Tylo.jl/actions/workflows/CI.yml)

Tile programming for NVIDIA GPUs in Julia, at no cost over writing the
instructions by hand.

A kernel describes **where** a tile lives (a memory tile: a pointer and a
layout), **who** holds each value (an `Ownership`, and a `Fragment` of
values in it) and **what** the hardware does (an `Atom`). Everything else,
from vector widths and `ldmatrix` groupings to reduction shuffles, `cp.async`
zero-fill and descriptors, is derived from their layouts while the kernel
compiles.

```julia
using Tylo, CUDACore, BFloat16s

# Lane t holds 8 consecutive values of a 256-wide row: one 16-byte load.
const ROW = distribute((1, 256), 32; vector=8, axis=2)
# A column-major 256×n device matrix as n contiguous rows (device arrays are 16-byte aligned).
@inline rows(a) = @inbounds permutedims(GlobalTile(a, 16))

function softmax_rows!(y, x)
    lane, row = Int32(threadIdx().x) - Int32(1), Int32(blockIdx().x) - Int32(1)
    @inbounds begin                                 # one warp per row, by the launch
        f = Float32.(load_fragment(ROW, tile(rows(x), (1, 256), (row, 0)), lane))
        w = exp.(f .- maximum(f; dims=2))           # a butterfly across the warp
        store!(tile(rows(y), (1, 256), (row, 0)), Fragment{BFloat16}(w ./ sum(w; dims=2)), lane)
    end
    return
end

x = CuArray(BFloat16.(randn(Float32, 256, 1000)))
y = similar(x)
@cuda threads=32 blocks=size(x, 2) softmax_rows!(y, x)
```

That kernel compiles to one 16-byte load, two five-step shuffle butterflies,
paired conversions and one 16-byte store per lane. Without `@inbounds`,
every access is checked: tile extents, alignment, thread indices.

Tylo leaves the launch, work assignment, allocation, pipelining and every
barrier to the kernel, so the same primitives serve standalone kernels and
[Megakernels](https://github.com/jool-space/Megakernels.jl)' task bodies.
The API is experimental.

## What's there

- Layout algebra (compose, divide, product, complement, inverses, swizzles)
  over mixed static and runtime extents.
- Loads, stores and `cp.async` copies derived from ownership and layout:
  `ldmatrix`/`stmatrix`, vectors, zero-filling bounded tiles.
- Fragment broadcasting, reductions across lanes, coordinates for masking,
  conversions to BF16/FP16/FP8.
- Warp MMA (16-bit, TF32, FP8, INT8, block-scaled MXFP8), TMA and bulk
  copies, Hopper WGMMA, Blackwell tcgen05 and tensor memory.
- Pipelines without bookkeeping: a kernel's shared memory declared once
  (barriers, scalars, aligned buffers), rings of stages that compute every
  barrier's phase, numbered named barriers, arrival tokens, and one
  configuration argument for a kernel's constants.

See the [documentation](docs/src/index.md) and the [status page](docs/src/status.md)
for what has run on which hardware. The [examples](examples/) include a
complete GEMM, softmax, streaming attention, a TMA/WGMMA pipeline and
datacenter FlashAttention pieces.

## Development

The test project is a workspace member, so one instantiate covers the
package, its tests and its docs (Julia 1.12+):

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(; workspace=true)'
julia --project=test test/runtests.jl --jobs=4        # host and GPU tiers
julia --project=test test/runtests.jl host            # host only
julia --project=test examples/gemm/run.jl 65 97 73
```

`gpu/` test files always check their kernels' assembly and run them when the
device allows. `TYLO_EVIDENCE=<dir>` saves every test kernel's PTX and cubin,
to compare machine code before and after a change.

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl                     # builds docs/build and runs the doctests
```
