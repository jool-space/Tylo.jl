# Design

## Three values, everything else derived

Tylo has one type for each fact a kernel needs:

- **where** a tile lives: a memory tile, a pointer and a [layout](layouts.md);
- **who** holds each value: an [`Ownership`](@ref Layouts.Ownership), and a
  [`Fragment`](@ref) of values in it;
- **what** the hardware does: an [`Atom`](@ref), described by the ownerships
  of its operands.

Everything else is derived from these while the kernel compiles, by layout
algebra: which slots form a vector, which `ldmatrix` covers a fragment, which
lanes a reduction shuffles with, which bytes a `cp.async` reads, which
descriptor an operand needs. A new operation joins by stating its ownership
or its storage requirement, and the loads, stores, reductions and checks
that already exist apply to it.

## One spelling per idea

- Every distribution of values over threads is an `Ownership`: instruction
  operands, copy distributions, reduction results, TMEM transfers, windows
  and transposes. Its canonical form makes equal distributions equal types.
- Every register value is a `Fragment`, whatever its width; packing into
  words is how instructions consume 8- and 16-bit values, not a separate type.
- A sub-tile is `window` or `tile`, an edge is `bound`, and the same
  operations accept all three.
- A register operand is `load_fragment(atom, role, tile, thread)`; a
  shared-memory operand is `operand(atom, role, tile, origin)`.
- Shapes, axes, counts and alignments are plain constants at every call
  site; `Val` appears only inside the implementation.

## Derivations are ordinary functions

The derivations (`reduction_plan`, `vector_plan`, `copy_plan`,
`matrix_copy_plan`, `window_plan`, ...) are pure Julia functions of
ownerships and layout types. Generated functions call them while building a
kernel's code, and the host tests call them directly, checking each against
an enumeration of every `(thread, slot)` pair for the atoms' ownerships and
for random ones. A plan that cannot be derived is an error at the call, not
a silent fallback to communication.

## Checked by default, free when proven

Each runtime invariant is a `@boundscheck`: sub-tile extents, alignment
declarations, thread indices, tile shapes, bounds that would split a
gathered vector. A kernel that has established them writes `@inbounds`, as
for array indexing, and the checks disappear from the machine code. What the
types decide (an element type, an ownership no instruction implements) is an
error in every build, at no cost.

## The kernel owns time

Tylo never inserts a barrier, fence, wait or allocation. Every asynchronous
operation returns or requires an explicit completion (see
[Completion](operations.md#Completion)). This is what lets the same
primitives serve a standalone kernel and a persistent megakernel's task
bodies, whose schedules are nothing alike.

## The machine code is the specification

Tylo's claim is that its abstractions cost nothing, so its tests check
machine code: instruction counts and kinds in the generated PTX, and SASS
compared before and after every change (`TYLO_EVIDENCE=<dir>` saves each
test kernel's PTX and cubin). Megakernels' MXFP8 projection, written in
Tylo, compiles to the same SASS as the same algorithm written by hand with
pointers, vectors and inline PTX.

## Boundaries

- Tiles are two-dimensional. Batch and head dimensions are windows and tile
  indices, not layout modes of the tile.
- Values move between threads only through reductions and the hardware's
  own instructions; there is no implicit redistribution.
- There is no autotuning, split-K or tile-size search; the kernel chooses.

## Open questions

- **Collective scope.** Warp, warpgroup and block collectives are implied by
  ownerships' thread counts; an explicit scope, like ThunderKittens'
  `group<N>`, might compose better across roles.
- **WGMMA accumulators** with independent K partials are their own type
  until `finish_mma`; they could be fragments with a replicated slot mode.
- **Higher-rank tiles** would make batched and multi-head layouts first-class.
