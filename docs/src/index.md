```@meta
CurrentModule = Tylo
DocTestSetup = :(using Tylo)
```

# Tylo

Tylo writes NVIDIA GPU kernels in Julia from three kinds of value:

| Value | What it says |
|:--|:--|
| [`Ownership`](@ref Layouts.Ownership) | which thread holds which element of a logical tile |
| [`GlobalTile`](@ref), `SharedTile`, [`TmemTile`](@ref) | where a logical tile lives in memory |
| [`Atom`](@ref) | one hardware instruction, described by the ownerships of its operands |

A [`Fragment`](@ref) is one thread's values together with their ownership.
Every operation connects these values, and everything it needs to know is
derived from their layouts while the kernel compiles:

- `load_fragment(ownership, tile, thread)` and `store!(tile, fragment, thread)`
  move values between registers and memory, as `ldmatrix`/`stmatrix`,
  vectors or scalars, whichever the layouts allow;
- `copy_async!(ownership, shared, global, thread)` copies without passing
  through registers, as `cp.async` vectors;
- broadcasting, `sum`/`maximum`/`minimum`, `window` and `relayout` work on
  fragments, with shuffles where values span lanes;
- `mma`, `mma_async`, `tma_load!` and `load_async` issue the hardware's
  matrix, bulk-copy and tensor-memory instructions.

```jldoctest
julia> rows = distribute((4, 8), 4; vector=8, axis=2)   # thread t holds row t
Ownership((4, 8), @Layout (4, 8) (1, 4))

julia> f = Fragment(ntuple(Float32, 8), Ownership((1, 8)));   # one thread's whole 1×8 tile

julia> f .- maximum(f; dims=2)
Fragment((-7.0f0, -6.0f0, -5.0f0, -4.0f0, -3.0f0, -2.0f0, -1.0f0, 0.0f0), Ownership((1, 8), @Layout (1, 8) (0, 1)))

julia> c = zero_accumulator(MMAAtom((16, 8, 16), Tylo.BFloat16));

julia> coordinates(c, 5).data   # the (row, col) of thread 5's four accumulator values
((1, 2), (1, 3), (9, 2), (9, 3))
```

## What Tylo leaves to the kernel

Tylo describes data and the operations on it. The kernel keeps the launch,
the assignment of work to blocks and warps, allocation, pipelining and every
barrier: nothing waits or synchronizes unless the kernel says so. That is
what lets the same primitives serve a standalone GEMM and a persistent
megakernel's task bodies.

## Reading order

1. [Layouts and ownership](layouts.md): the coordinate algebra everything
   else is written in.
2. [Memory tiles](tiles.md): storage, alignment, sub-tiles and bounds.
3. [Fragments](fragments.md): registers, arithmetic, reductions, positions.
4. [Hardware operations](operations.md): copies, matrix multiplies, TMA,
   WGMMA, tcgen05 and tensor memory.
5. [A complete kernel](kernel.md): the GEMM example, from host setup to the
   epilogue.
6. [Design](design.md) and [Status](status.md): the principles, the checks,
   and what has run on which hardware.
