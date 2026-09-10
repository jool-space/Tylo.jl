```@meta
CurrentModule = Tylo
DocTestSetup = :(using Tylo)
```

# Streaming attention and online state

This part of the library has two jobs: preserve a stable row summary while tiles
arrive, and convert a matrix result into the next matrix instruction's operand.
Neither job chooses buffers, CTA roles, scheduling, or barrier placement.

## Online statistics

`SoftmaxState` stores an FP32 maximum and unnormalized sum in an existing
`RowValues` distribution. This algorithm helper currently assumes the original
lane-local, warp-striped or warp-MMA row distributions and reduces dimension 2.
It has no `dims` argument and does not accept every generic or permuted fragment.
That is an unfinished API generalization, not a hardware requirement that
online statistics must be called rows. An update returns a new state, unnormalized weights,
and the factor that rescales a previous weighted numerator:

```jldoctest
julia> f = RowFragment((0f0, 0f0)); s = SoftmaxState(f);

julia> u = softmax_update(s, f);

julia> (only(u.state.maximum), only(u.state.sum), only(u.rescale))
(0.0f0, 2.0f0, 0.0f0)

julia> softmax_normalize(u.weights, u.state).data
(0.5f0, 0.5f0)
```

With previous summary `(m,l)` and new scores `x`, the update uses
`m′=max.(m,maximum(x;dims=2))`, `α=exp(m-m′)`, `p=exp(x-m′)`, and
`l′=α.*l.+sum(p;dims=2)`. Empty state is `(-Inf,0)`; explicit branches give empty
updates zero weights and avoid undefined differences and divisions.
Valid scores are finite; masking supplies `-Inf32`. Entirely masked rows
normalize to zero and have log-sum-exp `-Inf32`.

The numerator update is `o′=α*o+p*V`. `softmax_normalize(o,state)` belongs after
the final tile. `softmax_merge` combines independent summaries and exposes both
numerator rescale factors. Floating-point chunking/merge order is not associative.

The state and values must agree on the original row-result ownership. A 32-column MMA score accumulator and 64-column output
accumulator can share the same state when their M decomposition and one-N-warp
distribution agree. A lane-local summary cannot silently become a warp summary.
All lanes still participate in distributed reductions, including masked lanes.

The [streaming softmax](https://github.com/jool-space/Tylo.jl/blob/main/examples/softmax/streaming.jl)
uses four values per lane regardless of row width. It first streams statistics,
then rereads logits to write final probabilities. This bounds registers but
requires a second read and additional per-chunk reductions. The benchmark
compares it with full-row register fragments and a scalar three-pass warp loop.

## Same-lane conversion

For `mma.sync.m16n8k16`, two adjacent 16×8 C atoms supply a 16×16 A operand in
the same lanes. Their eight FP32 values become four packed BF16/FP16 words.
`pack_operand_a(atom,left,right)` performs rounding and packing. It does not
reinterpret FP32 bits or communicate between lanes.

For a tiled accumulator, `pack_operand_a(atom,acc,Val(m),Val(k))` selects a
zero-based M repetition and 16-column pair. It rejects multiple N warps and
unpaired N atoms. Coordinate tests enumerate every lane/value and repeated M/N
atom against an independent oracle; GPU tests compare with shared store/load
and consume the result in a second MMA.

The conversion uses round-to-nearest-even. Signed zeros and infinities survive;
NaNs remain NaNs, with unspecified payload/sign. There is no finite saturation
or flush-to-zero modifier. Communication-requiring layout conversions still need
an explicit implementation. This small correspondence does not supply a general
register redistribution engine.

## Attention as a consumer

The [worked attention kernel](https://github.com/jool-space/Tylo.jl/tree/main/examples/streaming_attention)
composes QK, the online update, same-lane BF16 conversion, PV and final
normalization. It fixes D=64 and a 64×32 score tile, while sequence lengths,
bounds and masks remain runtime data. Its one-stage copy schedule uses 16 KiB
shared memory and keeps score/probability tiles off global memory. The Boolean
input mask is still M×N. Probability rounding happens before PV; the denominator
remains FP32. Independent high-precision and same-boundary references make this
distinction testable.

CuTe's Ampere attention example expresses the same adjacent-atom correspondence
through layout division and reshaping; ThunderKittens expresses the row update
and conversion as register-tile operations. Tylo now has the narrow primitives
needed by this concrete dataflow. More general layouts should grow from further
proven mappings rather than assuming every reshape is a local operation.
