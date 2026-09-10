```@meta
CurrentModule = Tylo
DocTestSetup = :(using Tylo)
```

# Layouts, storage and ownership

Tylo now has a small layout algebra used by a complete tiled GEMM. This is
still a selected subset of CuTe's capabilities. The API is experimental.

## A layout is a function

`Layout(shape, strides)` maps zero-based logical coordinates to element
offsets. Shapes and strides may be hierarchical. Staticness belongs to each
leaf: use `static(n)` for a compile-time constant and an ordinary integer
for a runtime dimension or leading stride. Runtime strides do not become
new type parameters.

```jldoctest
julia> using Tylo.Layouts: Layout, static;

julia> l = Layout((static(16), static(32)), (static(32), static(1)));

julia> l((Int32(3), Int32(7)))
103

julia> Tylo.Layouts.tile(l, Val((8, 16)))(((3, 1), (7, 1))) == l((11, 23))
true
```

An integer coordinate decomposes first-mode-fastest. `tile` factors flat
modes into within-tile and tile-index components. `coalesce` removes unit
modes and merges contiguous modes without changing the function on that
linearized domain. Coalescing is host-side planning; mixed layouts retain
runtime representation, while wholly static inputs stay static.

`compose(f, l)` represents `f(l(c))`. It supports nonlinear offset
permutations without pretending they are affine strides. There is no
arbitrary symbolic composition simplifier or inverse solver yet.

## Swizzle phase belongs to the allocation

`Swizzle{B,M,S}` XORs two disjoint bit fields, preserving the low M bits.
The disjoint-field check is essential: overlapping fields can destroy
information instead of permuting addresses.

```jldoctest
julia> using Tylo.Layouts: Layout, static, Swizzle, compose;

julia> l = compose(Swizzle{2,3,2}(), Layout((static(32), static(32)), (static(32), static(1))));

julia> w = Tylo.Layouts.window(l, (1, 8), Val((16, 16)));

julia> w((2, 3)) == l((3, 11))
true
```

The window retains its parent's mapping and origin. Advancing a pointer
and restarting the swizzle at zero would generally describe different
storage. Nested windows combine logical origins while preserving the
allocation base. `cosize` counts storage through the largest offset,
including holes and any prefix retained by a window. For nonlinear layouts
it enumerates on the host; compute allocation sizes before launching.

Layout evaluation assumes valid coordinates. Constructors and windows check
their domains; a kernel may use `@inbounds` once its caller has validated
dimensions and its launch grid.

## Three representations, three addressing rules

- `GlobalTile` and `SharedTile` borrow typed LLVM pointers. Their layouts
  count elements; pointer access converts to bytes. Global coordinates are
  widened before offset arithmetic to support allocations larger than 4 GiB.
- `MMAFragment` and `RowFragment` hold immutable tuples in registers. They
  are values, not addressable arrays. Register indexing stays static.
- `TmemTile` and `TmemRows` retain the separate TMEM row/column encoding.
  Generic byte-pointer slicing is never applied to TMEM.

The MMA atom's ownership layout maps `(lane, logical_value)` to a matrix
coordinate. It is independent of the physical memory layout. A and B use
packed UInt32 registers; C uses FP32 registers. Their register counts come
from the instruction's operand roles, with no generic tile-area/32 rule.
B uses mathematical `(K,N)` coordinates throughout.

## Plans describe work, callers establish readiness

`CopyPlan` distributes 16-byte vectors among threads. `validate_copy` checks
both layouts for vector contiguity and alignment and checks destination
injectivity. The caller must also provide aligned pointers and keep the
allocations alive. Copy groups are per thread; `wait_copies` is not a CTA
barrier and is not a readiness proof for an independently owned object.

`MMA16x8x16` loads instruction-compatible shared tiles into typed fragments.
`TiledMMA` repeats this atom over a warp arrangement and a per-warp grid.
It reuses A/B operands across those repetitions. All participating lanes
must execute collectively with compatible views; types do not prove that
control-flow property. Shared memory must already be ready for consumption.

The GEMM example owns its one- or two-buffer pipeline. It synchronizes after
copy completion, and again after every warp has finished reading a buffer
before that buffer can be overwritten. Arithmetic on an accumulator returns
another value with the same ownership. Scaling and activation can therefore
be composed into an epilogue without a separate kernel.

## Lessons carried forward

Laythe's useful separation is pure coordinate mathematics with mixed static
and dynamic leaves. Tylo keeps that as an internal module for now. The old
Tylo prototype also identified useful memory spaces and operand roles, but
its generic register sizing and provisional byte arithmetic for TMEM did
not establish valid instruction contracts. The new implementation checks
those contracts against real generated code and executed kernels.

The TMA and Hopper WGMMA path now exercises canonical descriptor-compatible
shared layouts and explicit completion rules. Megakernels is its second
consumer. The next datacenter path can connect TMA, tcgen05 MMA and the
existing TMEM operations. See [TMA and Hopper WGMMA](@ref) for current limits.

Row reductions and broadcasts now cover lane-local, warp-striped, and tiled
MMA distributions; see [Row reductions and broadcasts](@ref). Rectangular
bounds and partial vectors are handled by [Boundary tiles](@ref).

General inverses/complements, arbitrary fragment redistribution, additional
dtypes and automatic allocation remain future work. They should arrive with
kernels that need and validate them.
