```@meta
CurrentModule = Tylo
DocTestSetup = :(using Tylo)
```

# Fragments

A [`Fragment`](@ref) is one thread's share of a logical tile: a tuple of
values and the ownership that says which coordinate each holds. It is
immutable, lives in registers, and is the same type whatever its origin: a
load, an MMA accumulator, a reduction, a TMEM transfer.

```jldoctest frag
julia> f = Fragment((1f0, 5f0, 2f0, 7f0), Ownership((2, 2)))
Fragment((1.0f0, 5.0f0, 2.0f0, 7.0f0), Ownership((2, 2), @Layout (1, 4) (0, 1)))

julia> f.data, size(f)
((1.0f0, 5.0f0, 2.0f0, 7.0f0), (2, 2))
```

`fill(x, ownership)` and `zeros(T, ownership)` make fragments;
`load_fragment(ownership, tile, thread)` reads one from memory and
`store!(tile, fragment, thread)` writes one back (the thread index may be
omitted for an ownership a single thread holds). `f.data` is always the
logical values. 8- and 16-bit values are stored packed in the 32-bit
register words instructions consume, so a BF16 operand stays in its words
through a loop.

## Arithmetic

Broadcasting applies any scalar function slot by slot and keeps the
ownership: `exp.(f)`, a user function, or a scalar PTX instruction. A
fragment with a singleton axis, such as a reduction's result, broadcasts
along it.

```jldoctest frag
julia> m = maximum(f; dims=2)
Fragment((2.0f0, 7.0f0), Ownership((2, 1), @Layout (1, 2) (0, 1)))

julia> exp.(f .- m)
Fragment((0.36787945f0, 0.13533528f0, 1.0f0, 1.0f0), Ownership((2, 2), @Layout (1, 4) (0, 1)))
```

Fragments of different ownerships do not mix: that would need values to
move between threads, which Tylo does only when asked.

## Reductions

`sum`, `maximum` and `minimum` reduce over `dims=1`, `dims=2` or the whole
tile, optionally after a function and from an initial value. The recipe
comes from the ownership: a tree over the values a thread holds, then xor
shuffles over the lanes that share the kept coordinate. A value is never
combined with its replicas. `dims` must be a constant, since it decides the
result's type.

```jldoctest frag
julia> sum(abs2, f; dims=1)
Fragment((26.0f0, 53.0f0), Ownership((1, 2), @Layout (1, 2) (0, 1)))

julia> sum(f)
Fragment((15.0f0,), Ownership((1, 1), @Layout (1, 1) (0, 0)))
```

With `init`, a thread's own values fold sequentially from it, so a running
statistic over a loop costs one operation per value
(`s = sum(abs2, x; dims=1, init=s)` is one fused multiply-add each). Values
combined across lanes must be 32 bits wide; within a thread any type
reduces. An axis whose values span warps has no shuffle recipe: reduce each
warp, then combine through shared memory, where
`Fragment(f, ownership)` re-owns a fragment's registers (the per-warp
results as rows of a matrix to reduce).

## Positions

`coordinates(f, thread)` is a fragment of each value's `(row, col)`, so
masking is broadcasting:

```jldoctest frag
julia> c = coordinates(f, 0);

julia> ifelse.((p -> p[1] == p[2]).(c), f, 0f0)   # keep the diagonal
Fragment((1.0f0, 0.0f0, 0.0f0, 7.0f0), Ownership((2, 2), @Layout (1, 4) (0, 1)))
```

## Changing the view, not the values

| Operation | Result |
|:--|:--|
| `window(f, origin, shape)` | the values inside a box, when every thread's are the same slots |
| `permutedims(f)` | the transposed tile, the same registers |
| `relayout(ownership, f[, thread])` | the same values in another slot order, when each thread already holds what it needs |
| `Fragment{T}(f)` | converted values (paired `cvt` instructions on the device; FP8 rounds to nearest even and saturates) |
| `reinterpret(T, f)` | the same bits as another type of the same size |
| `Fragment(f, ownership)` | the same registers under another ownership with as many slots |

```jldoctest frag
julia> relayout(Ownership((2, 2), @Layout((1, (2, 2)), (0, (2, 1)))), f)   # row-major slots
Fragment((1.0f0, 2.0f0, 5.0f0, 7.0f0), Ownership((2, 2), @Layout (1, (2, 2)) (0, (2, 1))))
```

Anything that needs values to cross threads, other than a reduction, is
rejected with an error rather than silently communicating.

## Online softmax

[`SoftmaxState`](@ref) keeps running maxima and sums in a reduced
ownership; `softmax_update`, `softmax_merge` and `softmax_normalize`
implement the streaming softmax of attention in those terms. The
[streaming attention example](https://github.com/jool-space/Tylo.jl/tree/main/examples/streaming_attention)
uses them with MMA accumulators.
