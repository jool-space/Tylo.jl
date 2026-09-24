```@meta
CurrentModule = Tylo
DocTestSetup = :(using Tylo)
```

# Layouts, storage and ownership

Tylo's layout algebra describes coordinate maps used by the instruction
implementations. Start with [Design and boundaries](design.md) for how storage
maps differ from register ownership. This page explains the mathematics and
notation; [the GEMM walkthrough](gemm.md) shows a complete consumer.

## A layout is a function

`Layout(shape, strides)` maps zero-based logical coordinates to offsets.
A memory view supplies the unit: elements for global/shared storage, and typed
TMEM slots for the TMEM adapter. Shapes and strides may be hierarchical. Staticness belongs to each
leaf: use `static(n)` for a compile-time constant and an ordinary integer
for a runtime dimension or leading stride. Runtime strides do not become
new type parameters.

```jldoctest
julia> using Tylo.Layouts: @Layout;

julia> l = @Layout (16, 32) (32, 1);

julia> l((Int32(3), Int32(7)))
103

julia> Tylo.Layouts.tile(l, Val((8, 16)))(((3, 1), (7, 1))) == l((11, 23))
true
```

For a matrix shape `(16,32)`, strides `(32,1)` make the second axis contiguous:
this is row-major storage. Strides `(1,16)` make the first axis contiguous:
ordinary column-major storage. `Layout` accepts either. There is no automatic
choice of major order in the constructor or `@Layout`.

```jldoctest
julia> using Tylo.Layouts: @Layout;

julia> l = @Layout (16, 32) (32, 1);

julia> permutedims(l)((7, 3)) == l((3, 7))
true
```

Permutation reverses the top-level shape and stride modes together. Its matching
coordinate is reversed too. With nested modes, their internal structure remains
intact. The logical index can also be a nested tuple, such as the `(within,
outer)` coordinates produced by `tile` in the preceding example.

An integer coordinate decomposes first-mode-fastest. `tile` factors flat
modes into within-tile and tile-index components; it is one use of
[the algebra](@ref "The algebra") below, which composes, divides, multiplies
and inverts layouts symbolically.

`compose(f, l)` represents `f(l(c))` for any callable `f`. When `f` is itself
a layout the result is a new layout; when it is a swizzle, the composition
stays a `Composition`, without pretending the permutation is affine.

## Static layout notation

`@Layout` defaults to static shape and stride values. Use `$` to preserve a
value's existing type: ordinary integers stay ordinary integers, while
already-static values stay static. The explicit `Layout` constructor keeps
its existing behavior.

```jldoctest
julia> using Tylo.Layouts: @Layout, shape, static;

julia> l = @Layout (16, 32) (32, 1);

julia> shape(l) === (static(16), static(32))
true

julia> n, ld = Int32(5), Int32(128);

julia> l = @Layout ((8, 4), $n) ((1, 8), $ld);

julia> shape(l) === ((static(8), static(4)), n)
true

julia> strides(l) === ((static(1), static(8)), ld)
true

julia> l(((Int32(3), Int32(2)), Int32(1)))
147
```

Tuple syntax is traversed recursively, preserving its structure. An unmarked
variable such as `subshape` becomes `static(subshape)`; `static` recursively
converts tuple values too. `$subshape` instead preserves the complete subtree,
including any mixture of static and ordinary integers. It inserts one mode,
without splicing the tuple's contents. Whole shape/stride tuples can also be
passed as `@Layout $shape_value $stride_value`.

Unmarked expressions are evaluated normally and passed to `static`, once per
occurrence. Thus a type-derived `K` works as `@Layout (64, K) (K, 1)`, but an
unknown runtime integer does not become known to inference merely because it
is unmarked. Interpolate a whole ordinary expression as `$(2*n)`.

Both arguments are required. Tuple splatting and automatic stride generation
are outside this notation; shape and stride trees obey the same constructor
checks as explicit layouts.

When generating Julia code, an enclosing quote consumes `$` before the layout
macro sees it. Insert a dollar expression explicitly to preserve the marker:

```julia
n_marker = Expr(:$, :n)
ld_marker = Expr(:$, :ld)
expr = :(@Layout (64, $n_marker) ($ld_marker, 1))
# Constructs the syntax: @Layout (64, $n) ($ld, 1)
```

The names `n` and `ld` are resolved when the generated code executes. Emitting
the explicit `Layout` constructor is also a straightforward choice for code
generators.

## Swizzle phase belongs to the allocation

`Swizzle{B,M,S}` XORs two disjoint bit fields, preserving the low M bits.
The disjoint-field check is essential: overlapping fields can destroy
information instead of permuting addresses.

```jldoctest
julia> using Tylo.Layouts: @Layout, Swizzle, compose;

julia> l = compose(Swizzle{2,3,2}(), @Layout((32, 32), (32, 1)));

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
  A third argument, `Val(align)`, declares that the pointer and every
  non-unit stride are multiples of `align` bytes; fragment loads and stores
  then vectorize up to that width, and windows keep the declaration only at
  aligned origins. The default is the element size.
- `Fragment` holds immutable values and explicit ownership. MMA operands
  retain instruction-specific packing. Neither is an addressable local array;
  register indexing stays static.
- `TmemTile` maps logical coordinates to TMEM storage; `TmemTransfer` binds
  register ownership to a supported hardware access. See [TMEM tiles and transfers](@ref).
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

`MMAAtom` describes an instruction by its operand ownerships. `CopyAtom`
describes the 8×8 `ldmatrix`/`stmatrix` matrix copies the same way, by the
row each lane addresses and the two units each lane holds. `load_fragment`
and `store!` derive the copy of any 32-lane ownership from those tables:
blocks along the shared layout's static unit-stride axis, plain or
transposed, grouped into `.x1`/`.x2`/`.x4` instructions, with the words
permuted into the fragment's slot order; `validate_copy(ownership, T,
layout)` checks the addressed rows on the host. `load_a` and `load_b` are
that load with the atom's operand ownership.
`ScaledMMAAtom` is the block-scaled `mma.sync` of consumer Blackwell
(`.kind::mxf8f6f4`): `D = (A .* scale_A) * (B .* scale_B) + C` with FP8
operands and one `Float8_E8M0FNU` power-of-two scale per row of A and per
column of B. With K = 32 one instruction multiplies one microscaling block,
so a K loop over MXFP8 operands accumulates the product without
dequantizing anything. A, B and the accumulator keep the dense atom's
ownerships. The scales are two more operands, `ScaleA` (16×4) and `ScaleB`
(4×8): a lane's register holds the four scales of its row or column for four
consecutive K blocks, and the `block` argument of `mma` selects one. These
ownerships replicate (lane bit 1 of a row scale and both lane bits of a column
scale have zero stride), which is what the instruction's thread selectors
permit; they were measured on a GB10 by giving every thread and byte its own
exponent.

`TiledMMA` repeats this atom over a warp arrangement and a per-warp grid.
It reuses A/B operands across those repetitions. All participating lanes
must execute collectively with compatible views; types do not prove that
control-flow property. Shared memory must already be ready for consumption.

The GEMM example owns its one- or two-buffer pipeline. It synchronizes after
copy completion, and again after every warp has finished reading a buffer
before that buffer can be overwritten. Arithmetic on an accumulator returns
another value with the same ownership. Scaling and activation can therefore
be composed into an epilogue without a separate kernel.

## The algebra

Layouts compose. `compose(a, b)` is the layout of `a(b(c))` on `b`'s domain,
worked out symbolically from the two stride trees, so a tile of a tile or a
thread's view of a tile is one layout with plain strides, evaluated by a
few multiplies. `b` addresses `a`'s linearized domain, first mode fastest,
which is why the tile below has stride 16 along columns. The result is exact or it throws: a mode of `b` has to step
evenly inside a mode of `a` or start on a multiple of it, and where it runs
past a mode it has to do so a whole number of times. Nothing is rounded.

```jldoctest algebra
julia> using Tylo.Layouts: @Layout, shape, compose, complement, concat, logical_divide, zipped_divide, static;

julia> l = @Layout (16, 32) (32, 1);            # row-major 16×32

julia> t = compose(l, @Layout (8, 8) (1, 16));   # the top-left 8×8 in l's own coordinates

julia> strides(t)
(static(32), static(1))

julia> t((3, 5)) == l((3, 5))
true
```

`complement(t, n)` is the layout of everything `t` leaves out of `[0, n)`:
the positions of a tile. Concatenating the two gives an injective layout
covering the whole extent, which is how `logical_divide` splits a layout
into `((within tile), (tile index))`, one tile per mode, and how
`zipped_divide` regroups that as `((tiles...), (indices...))`.

```jldoctest algebra
julia> c = complement(@Layout((8,), (1,)), static(32));   # the four positions of an 8-tile in 32

julia> shape(c), strides(c)
((static(4),), (static(8),))

julia> z = zipped_divide(l, (static(8), static(16)));

julia> shape(z), strides(z)
(((static(8), static(16)), (static(2), static(2))), ((static(32), static(1)), (static(256), static(16))))

julia> z(((3, 5), (1, 1))) == l((3 + 8, 5 + 16))
true
```

Products go the other way: `logical_product(block, tile)` repeats a block at
the positions a tile enumerates, and `blocked_product` and `raked_product`
interleave the two mode by mode. `right_inverse` and `left_inverse` invert a
static layout on its image; `compose(l, right_inverse(l))` is the identity
on the contiguous run of offsets `l` reaches.

Two rules make the algebra predictable. Staticness is decided leaf by leaf:
arithmetic between static leaves stays static, and a runtime leaf keeps its
integer type, so a global tile with an `Int32` leading dimension composed
with a static tile has an `Int32` stride where the runtime value enters and
static leaves elsewhere. Structure is decided by types alone: which modes
merge, drop or split never depends on a runtime value, so a layout
expression has one type for every value of its runtime leaves. The
structural operations are generated functions that read the types once and
emit straight-line code.

The oracle for all of this is enumeration. `agree(a, b)` compares two
layouts pointwise over their domain, and `is_injective` and `is_compact`
decide the properties the algebra does not. The host suite checks every
operation against that oracle, including that inexact cases throw.

## Limits of the algebra

Windows are affine, not linear, so they compose with callables but not
symbolically with layouts; slice or window the composed layout instead.
Complements and inverses need static strides, as their structure is a
sorting of strides. The algebra does not vectorize or redistribute registers
on its own; those derivations live with the instructions that need them.

Describing a map is separate from binding it to an instruction. For example,
TMA accepts a canonical descriptor-compatible layout, not every layout that can
be written here. A scalar fragment operation can support a new ownership before
that ownership has a reduction implementation. See [Current status](validation.md)
and [the design decisions](design.md#Decisions-still-worth-challenging).
