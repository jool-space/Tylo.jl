```@meta
CurrentModule = Tylo
DocTestSetup = :(using Tylo)
```

# Tylo

Tylo writes NVIDIA GPU kernels in Julia from three kinds of value. Everything
else derives from them by [layout algebra](layouts.md).

| Value | What it says |
|:--|:--|
| [`Ownership`](@ref Layouts.Ownership) | Which thread holds which element of a logical tile |
| `GlobalTile`, `SharedTile`, `TmemTile` | Where a logical tile lives in memory |
| [`Atom`](@ref) | One hardware instruction, as the ownerships of its operands |

A [`Fragment`](@ref) is one thread's values together with their ownership.
The verbs connect the three: `load_fragment` and `store!` move a fragment
between registers and a tile; `mma`, `copy_async!` and `tma_load!` issue an
atom's instruction; `window`, `permutedims`, broadcasting and `sum`/`maximum`
work on fragments without leaving the registers. Every size, shape and axis
argument is a plain constant, never a `Val`.

```jldoctest
julia> atom = MMAAtom((16,8,16), Tylo.BFloat16);

julia> c = zero_accumulator(atom);

julia> [coordinate(c.ownership, 5, slot) for slot in 0:3]   # thread 5's four values
4-element Vector{Tuple{Int64, Int64}}:
 (1, 2)
 (1, 3)
 (9, 2)
 (9, 3)

julia> (2f0 .* (c .+ 1f0)).data
(2.0f0, 2.0f0, 2.0f0, 2.0f0)
```

Read on in this order: [Design](design.md), [Layouts](layouts.md),
[Fragments](rows.md), then one complete kernel, [GEMM](gemm.md). The hardware
paths are [TMA and WGMMA](hopper.md), [TMEM](tmem.md),
[streaming attention](streaming.md) and [boundary tiles](boundaries.md).
[Read the implementation](codebase.md) maps files to tests, and
[Current status](validation.md) says what runs on which device and how to run
the suites.
