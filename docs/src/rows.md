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

Every distribution is an `Ownership((rows, cols), mapping)`: a layout from
`(thread, slot)` to the column-major index of the logical coordinate held
there. Instruction operands, reduction results, windows and transposes are
all ownerships of this one type, and the constructor stores the mapping in a
canonical form, so two ownerships that distribute a tile alike have the same
type. Common cases:

```julia
Ownership((8,8))                             # one thread holds the whole tile
Ownership((1,8); threads=32)                 # every lane holds all of it
Ownership((32,n), @Layout((32,n),(1,32)))    # lane t holds row t, n values
Ownership((1,32n), @Layout((32,n),(1,32)))   # the 32 lanes interleave along one row
```

`fill(x, ownership)` and `zeros(T, ownership)` make fragments, and a tile one
thread holds loads and stores without a thread index:
`load_fragment(Ownership((8,8)), tile)`.

Here each lane instead owns a 2×2 patch:

```jldoctest
julia> ownership = Ownership((16,8), @Layout(((8,4),(2,2)), ((2,32),(1,16))));

julia> f = Fragment((1f0, -2f0, 3f0, -4f0), ownership);

julia> [coordinate(ownership, 0, slot) for slot in 0:3]
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
julia> rows = Ownership((32,3), @Layout((32,3),(1,32)));

julia> f = Fragment((1f0, 2f0, 3f0), rows);

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
julia> f = Fragment((-2f0, 0f0, 3f0), Ownership((32,3), @Layout((32,3),(1,32))));

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
| `Fragment(values, ownership)` of any element type | `map`, fused broadcast, type-changing results, `Fragment{T}(f)` conversion | FP32 values with a derived recipe | Logical axis permutation; static `window`; ownership-derived `load_fragment`/`store!` |
| Warp-MMA accumulator and operands | The same: they are ordinary fragments | Supported atom/tiled recipes below | Instructions pack 8- and 16-bit values into words when they issue |
| Completed WGMMA result | The same | Two results per thread, replicated across four lanes | `finish_mma` combines partials |
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
julia> f = Fragment((1f0, 2f0, 3f0), Ownership((32,3), @Layout((32,3),(1,32))));

julia> g = permutedims(f);

julia> size(Tylo.Layouts.layout(g))
(3, 32)

julia> size(Tylo.Layouts.layout(maximum(g; dims=1)))
(1, 32)

julia> permutedims(g .- maximum(g; dims=1)) === f .- maximum(f; dims=2)
true
```

Transposing twice restores the identical fragment, since canonical ownerships
of the same distribution are the same value. The transpose does not
redistribute values into a different MMA operand distribution; pending
asynchronous results keep their completion requirements.

## Collective implementations and participation

A reduction recipe is derived from the ownership rather than written per
ownership type. While generating code, Tylo projects the ownership's
`(thread, slot) → index` layout onto the kept axis. Slot modes with zero
stride in that projection are the slots of a thread that share a kept
coordinate; thread modes with zero stride are the lane-index bits whose flip
preserves it. The recipe is a local tree over those slots followed by one xor
shuffle per such bit, and the projection itself, with its zero strides, is the
result ownership. For the ownerships used so far this reproduces the
hand-written recipes exactly:

| Ownership | Derived communication | Result per thread |
|:--|:--|:--|
| One row per lane, reducing along the row | Local register arithmetic | One result |
| One row interleaved across the warp | Five butterfly shuffle rounds | One result, replicated across the warp |
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
kernel must supply a constant `dims`.

All 32 lanes of each warp execute distributed reductions, including the
four-lane MMA reductions. Bounds masks select data and neutral values; they
must not make participation conditional. Use zero for a sum and `-Inf32` for
a maximum over padded entries. There is no implicit CTA barrier.

Tiled MMA currently requires **one warp along N** for this reduction.
Multiple warps along M own independent groups. With multiple N warps a
complete reduction needs communication between warps, composed as below.

## Running statistics and combining warps

`sum(f, x; dims, init)`, `maximum` and `minimum` apply `f` to each value and
join an initial fragment in the result's ownership. Values a thread holds
fold sequentially from it, so a statistic carried through a loop costs one
operation per value, and `sum(abs2, x; dims=1, init=s)` one fused
multiply-add:

```julia
chunk = Ownership((8,8))              # one thread's 8 k × 8 columns
stats = Ownership((1,8))              # its per-column values
s = m = zeros(Float32, stats)
for k in lane:32:chunks-1
    x = Float32.(load_fragment(chunk, tile(input, (8,8), (k,0))))
    s = sum(abs2, x; dims=1, init=s)
    m = maximum(abs, x; dims=1, init=m)
end
```

`Fragment(f, ownership)` gives the same registers another ownership with as
many slots. Declaring each lane's statistics as one row of a warp-wide matrix
makes the lane axis reducible, with five butterfly rounds per column. The
result is replicated across the warp, and a store of a replicated fragment
is issued by one replica only. Warps combine through shared memory: each
stores its row, the caller synchronizes, and every thread loads all rows and
reduces them locally:

```julia
warp = Ownership((32,8), @Layout((32,8),(1,32)))
store!(tile(partials, (1,8), (warp_index,0)), maximum(Fragment(m, warp); dims=1), lane)
# barrier
largest = maximum(load_fragment(Ownership((8,8)), partials); dims=1)
```

`relayout(to, f, thread)` moves values into another ownership when every
thread already holds what it needs but which value depends on the thread:
each slot becomes a chain of selects on the thread's coordinate. A fragment
of eight per-column scale codes held by every lane becomes a block-scaled
MMA's B scale register, where lane quad `q` needs column `q`.
`reinterpret(T, f)` reads the same bits as another element type of one size.

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

## Element types and conversion

A fragment always holds logical values, whatever their width: a BF16
fragment holds `BFloat16` values, an FP8 fragment `Float8_E4M3FN` values.
Instructions consume two 16-bit or four 8-bit values per 32-bit register;
the binding packs them, adjacent slots into one word lowest first, when it
issues, and a load unpacks the words it reads. Both are bit operations that
LLVM folds when a load feeds an instruction, so the representation costs
nothing and no packed fragment type is needed.

`Fragment{T}(f)` converts every value to `T` and keeps the ownership. The
8-bit element types are `Int8`, `UInt8` and Microfloats' `Float8_E4M3FN` and
`Float8_E5M2`, which Tylo re-exports. Conversions into them follow
[`Tylo.convert_element`](@ref): round to nearest even and saturate to the
largest finite value, as `cvt.rn.satfinite` does, whatever overflow policy
the type declares. They go through Microfloats' conversion funnel, and FP32
to BF16, FP16 and FP8 through paired conversions whose 16-bit results are
halves of the fragment's words, so a kernel compiled for a target with the
instructions gets one native `cvt` per two values, and the same bits as the
host.

```jldoctest
julia> f = Fragment((1f0, -2f0, 1000f0, 0.1f0), Ownership((32,4), @Layout((32,4),(1,32))));

julia> Float32.(Fragment{Float8_E4M3FN}(f)).data     # saturates, rounds to nearest even
(1.0f0, -2.0f0, 448.0f0, 0.1015625f0)

julia> Float32.(Fragment{Tylo.BFloat16}(f)).data
(1.0f0, -2.0f0, 1000.0f0, 0.100097656f0)
```

This separation follows the useful distinction between element type, storage
and mapping in [CuTe tensors](https://github.com/NVIDIA/cutlass/blob/main/include/cute/tensor_impl.hpp)
and [ThunderKittens register tiles](https://github.com/HazyResearch/ThunderKittens/blob/main/include/types/register/rt.cuh).
Tylo keeps scalar arithmetic and logical reductions in Julia's normal spelling;
instruction roles and completion remain explicit at their boundaries.
