```@meta
CurrentModule = Tylo
DocTestSetup = :(using Tylo)
```

# Register fragments and logical axes

A `Fragment(values, ownership)` is **one thread's share of a logical tile**.
Its values need not form a row or be contiguous. Two separate mappings matter:

- Ownership: `(thread, local value slot) → logical tile coordinate`.
- Memory layout: `logical tile coordinate → memory offset`.

Changing the storage strides does not change which thread owns each value.
Conversely, changing ownership can require moving values between threads even
when the logical tile shape stays the same. An MMA lane, for example, holds
values spanning two logical rows, with other lanes supplying the missing
columns. A fragment is therefore not an ordinary array of its local registers.

The public value type is `Fragment`. Two compact ownership patterns cover the
common local and interleaved cases:

```julia
Fragment(values, Tylo.Layouts.LocalOwnership{N,2}())   # N values per lane along axis 2
Fragment(values, Tylo.Layouts.StripedOwnership{N,2}()) # N values per lane interleaved along axis 2
```

Use axis `1` for the corresponding transposed logical interpretation. These
patterns have shapes `(32,N)` and `(1,32N)` respectively at axis `2`. They
specify ownership, independently of memory order. `Layouts.Ownership` accepts
an explicit thread/value mapping when neither pattern fits.

Here each lane instead owns a 2×2 patch:

```jldoctest
julia> using Tylo.Layouts: @Layout, coordinate

julia> ownership = Tylo.Layouts.Ownership(Val((16,8)),
           @Layout(((8,4),(2,2)), ((2,32),(1,16))));

julia> f = Fragment((1f0, -2f0, 3f0, -4f0), ownership);

julia> [coordinate(ownership, 0, Val(e)) for e in 0:3]
4-element Vector{Tuple{Int64, Int64}}:
 (0, 0)
 (1, 0)
 (0, 1)
 (1, 1)

julia> ifelse.(f .> 0f0, f, 0f0).data
(1.0f0, 0.0f0, 3.0f0, 0.0f0)
```

## Julia arithmetic and reductions

Use `exp.(f)`, `f .* scale`, `ifelse.(f .> 0f0, f, 0f0)`, and
`maximum(f; dims=2)`. Nested dotted expressions fuse into scalar expressions
for each local value. Comparisons produce Bool fragments, conversions such as
`Float16.(f)` change element type, and scalar promotion follows Julia's rules:
`f .+ 1.0` promotes FP32 values to FP64. Use `1f0` when FP32 is intended.

`map(op, a, b)` combines corresponding values with matching ownership.
Broadcasting additionally expands an implemented reduction's singleton axis.
The reduced fragment records where results are replicated between lanes;
applying them again requires no additional communication. Matching logical
shapes alone are insufficient: incompatible ownerships are rejected.

```jldoctest
julia> f = Fragment((1f0, 2f0, 3f0), Tylo.Layouts.LocalOwnership{3,2}());

julia> only(sum(f; dims=2))
6.0f0

julia> (f .- maximum(f; dims=2)).data
(-2.0f0, -1.0f0, 0.0f0)

julia> size(Tylo.Layouts.layout(maximum(f; dims=2)))
(32, 1)
```

`only` on a reduced fragment extracts this thread's scalar
result; it does not assert that the complete logical tile has one element.
There is no general fragment indexing, iteration, or mutable array interface.
The `.data` tuple describes local payload, while `Layouts.layout(f)` describes
the logical tile and ownership. Do not use local tuple reductions as a
substitute for logical reductions.

## Scalar functions, including PTX

`exp` has no special tile implementation in Tylo. For each local value,
broadcast calls the supplied scalar function and rebuilds a fragment with the
same ownership. A GPU-compilable user function participates in exactly this
way. Nested dotted expressions fuse over each static local slot.

```jldoctest
julia> f = Fragment((-2f0, 0f0, 3f0), Tylo.Layouts.LocalOwnership{3,2}());

julia> affine_relu(x, a, b) = max(muladd(a, x, b), 0f0);

julia> affine_relu.(f, 2f0, 1f0).data
(0.0f0, 1.0f0, 7.0f0)
```

Inside a GPU kernel with `using PTX`, scalar instruction callables work directly:

```julia
# f is a ready Float32 register fragment.
y = ptx"fma.rn.f32".(f, 2f0, 1f0)
z = ptx"ex2.approx.f32".(f)

@inline approximate_exp(x::Float32) =
    ptx"ex2.approx.f32"(x * 1.442695f0)
w = approximate_exp.(f .- m)
```

`ex2.approx.f32` computes an approximate base-two exponential. Ordinary
`exp(::Float32)` uses the GPU backend's scalar math implementation; requesting
an approximate PTX instruction explicitly chooses different numerical behavior.
Broadcast itself chooses neither approximation policy nor a memory transfer.
See [the PTX instruction specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2).

This mechanism supports independent scalar computations. A shuffle, barrier or
matrix instruction still has its own participation and operand contract; placing
it inside a scalar function does not establish a valid tile collective. A memory
tile must first be loaded through a supported operation. Pending values cannot
participate in fragment arithmetic.

## Current representation support

| Representation | Scalar arithmetic | Reductions | Views and conversion |
|:--|:--|:--|:--|
| `Fragment(values, ownership)` | `map`, fused broadcast, type-changing results | Only implemented ownership recipes | Logical axis permutation; static `window` for lane-local/TMEM-transfer ownership |
| Ready warp-MMA accumulator | `map`, broadcast; `Fragment(acc)` exposes local values | Supported atom/tiled recipes below | Axis permutation preserves local slots; MMA operands still require specific packing |
| Completed WGMMA result | Ordinary `Fragment`: fused broadcast, `map`, conversions | Two results per thread, replicated across four lanes | `finish_mma` combines partials; logical axis permutation and ownership-based `store!` |
| `PackedFragment{T}` | `unpack` first; `T` is the logical element type | `unpack` and convert to FP32 first | BF16/FP16 packing, axis permutation, pair-aligned windows, typed global/TMEM stores |
| Pending TMEM/WGMMA result | Wait first | Wait first | No arithmetic completion implied by its Julia type |

`SoftmaxState(f; dims)` uses reduced fragments and accepts either implemented
logical axis, including WGMMA and permuted ownership. See
[Streaming attention and online state](streaming.md).

## Axes are independent of memory order

A row fixes logical coordinate 1 and varies coordinate 2. This says nothing
about which coordinate has unit memory stride. `dims=2` reduces that second
coordinate. Reversing a memory layout's shape, strides, **and coordinates**
preserves addresses; reversing only the strides or only the reduction axis
does not describe the same computation. With hierarchical shapes, exchange
the top-level modes together rather than reversing all leaves recursively.

`permutedims(f, (2,1))` exchanges a fragment's logical axes by changing its
ownership metadata. The values stay in the same threads and local slots.
Its corresponding reduction is therefore `dims=1`:

```jldoctest
julia> f = Fragment((1f0, 2f0, 3f0), Tylo.Layouts.LocalOwnership{3,2}());

julia> g = permutedims(f);

julia> size(Tylo.Layouts.layout(g))
(3, 32)

julia> size(Tylo.Layouts.layout(maximum(g; dims=1)))
(1, 32)

julia> parent(g .- maximum(g; dims=1)) === f .- maximum(f; dims=2)
true
```

This view does not redistribute values into a different MMA operand format.
`Fragment(accumulator)` exposes ready warp-MMA accumulator values and their
ownership; packed operands and pending asynchronous results retain their
separate representations and completion requirements.

## Collective implementations and participation

A reduction recipe is derived from the ownership rather than written per
ownership type. While generating code, Tylo enumerates the ownership's
`(thread, slot) → coordinate` table and finds, for the requested axis, the
slots of a thread that share a kept coordinate and the lane-index bits whose
flip preserves it. The recipe is a local tree over those slots followed by
one xor shuffle per such bit. The result ownership, with its replication
across lanes, is fitted from the same table. For the ownerships used so far
this reproduces the hand-written recipes exactly:

| Ownership | Derived communication | Result per thread |
|:--|:--|:--|
| `Layouts.LocalOwnership`, axis along the slots | Local register arithmetic | One result |
| `Layouts.StripedOwnership` | Five butterfly shuffle rounds | One result, replicated across the warp |
| Warp MMA or completed WGMMA accumulator, `dims=2` | Two shuffle rounds per row after local reduction | Two results per M repetition, replicated across four lanes |
| TMEM-transfer fragment | Local register arithmetic | One result per thread |

Either logical axis may be reduced when its values live within one warp; the
2×2-patch ownership above reduces its rows with two shuffles, and a
lane-local row reduces across lanes with five. An axis whose values span
warps, such as an MMA tiling with several N warps, has no shuffle recipe and
raises an error rather than reducing only this thread's values. Full
reductions are not provided. Lower-precision values must be converted
explicitly with `Float32.(f)` before reducing. Floating-point reduction order
need not match a sequential sum. The result type depends on the axis, so a
kernel must supply a constant; `dims=Val(2)` makes that explicit.

All 32 lanes of each warp execute distributed reductions, including the
four-lane MMA reductions. Bounds masks select data and neutral values; they
must not make participation conditional. Use zero for a sum and `-Inf32` for
a maximum over padded entries. There is no implicit CTA barrier.

Tiled MMA currently requires **one warp along N** for this reduction.
Multiple warps along M own independent groups. With multiple N warps a
complete reduction needs communication between warps, which remains explicit.
Megakernels owns that shared scratch and worker synchronization.

## Worked softmax

The [softmax example](https://github.com/jool-space/Tylo.jl/tree/main/examples/softmax)
contains lane-local and warp-striped kernels and an epilogue over actual MMA
output. Its arithmetic is:

```julia
shift_logit(x, m) = m == -Inf32 ? -Inf32 : x - m
normalize_weight(x, s) = s == 0f0 ? 0f0 : x / s

m = maximum(f; dims=2)
weights = exp.(shift_logit.(f, m))
result = normalize_weight.(weights, sum(weights; dims=2))
```

Valid logits must be finite; explicit masks supply `-Inf32`. For an entirely
masked group, ordinary subtraction would evaluate `-Inf32 - -Inf32` to NaN.
The first helper instead leaves those logits at `-Inf32`, whose exponentials
are zero. The second helper returns zeros when their sum is zero. This is an
explicit all-masked softmax convention, not a general rule for subtraction.
Other NaN or infinite valid logits are outside the example's contract.

The standalone kernels use physical `(columns, rows)` matrices, giving
adjacent warp lanes adjacent addresses. This storage choice is separate from
the fragment's logical axes. The MMA example normalizes one complete output
tile, not groups spanning independently scheduled tiles.

## Logical types and register packing

`BFloat16.(f)` and `Float16.(f)` perform ordinary numerical conversion and return
fragments of those element types. Some instructions consume two 16-bit values
in each 32-bit register operand. `pack(T, f)` performs conversion and constructs
that packed representation; `pack(f)` preserves every bit of already typed
16-bit values. `unpack` returns an ordinary `Fragment{T}` without conversion.

```jldoctest
julia> f = Fragment((1f0, -2f0), Tylo.Layouts.LocalOwnership{2,2}());

julia> p = pack(Float16, f);

julia> eltype(p), p.data
(Float16, (0xc0003c00,))

julia> unpack(p).data
(Float16(1.0), Float16(-2.0))

julia> Float32.(unpack(p)).data
(1.0f0, -2.0f0)
```

Ownership still counts logical elements, and adjacent *local slots* form each
pair. Packing does not imply adjacent memory addresses or redistribute values.
A `PackedFragment(T, words, ownership)` constructor interprets supplied bits;
it does not convert a different numeric format to `T`. NaN payload preservation
is guaranteed for packing/unpacking already typed bits, not for numerical conversion.

This separation follows the useful distinction between element type, storage
and mapping in [CuTe tensors](https://github.com/NVIDIA/cutlass/blob/main/include/cute/tensor_impl.hpp)
and [ThunderKittens register tiles](https://github.com/HazyResearch/ThunderKittens/blob/main/include/types/register/rt.cuh).
Tylo keeps scalar arithmetic and logical reductions in Julia's normal spelling;
instruction roles and completion remain explicit at their boundaries.
