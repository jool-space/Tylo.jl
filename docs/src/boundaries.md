# Boundary tiles

A tile has a static capacity; a source matrix has a logical extent. These are
different quantities. Tylo's bounded copy takes both without pretending the
source contains a complete tile:

```julia
copy_async!(plan, shared_destination, global_source, (row, col), thread)
commit_copies()
wait_copies(0)
# The caller synchronizes participating producers and consumers here.
```

The origin and all coordinates are zero-based. Each copy plan assigns fixed
16-byte vectors. For a vector, the implementation uses the plan's coordinate
map to check source validity, contiguity, and actual pointer alignment. A valid
aligned vector uses `cp.async`; other vectors use scalar loads and shared stores,
writing zero for every invalid element. An invalid source coordinate is checked
before forming its pointer. The fixed shared destination remains fully populated.

This supports partial vectors, leading dimensions not divisible by eight BF16
values, noncontiguous source strides, and negative crop origins. The shared
layout must still satisfy the existing vector-layout contract. The pinned PTX
wrapper currently exposes full-vector copies; the scalar path handles zero-fill
without introducing another inline PTX wrapper in Tylo.

## Bounded tiles

`bound(tile, extents)` gives a tile the extents of its valid region, with
`nothing` for an axis that needs no check. Fragment loads through it zero the
elements outside and stores skip them; the check is on the thread's
coordinate plus each slot's constant offset, and a vector counts as valid
when its first element is, so the extent along the vector axis must be a
multiple of the vector length. `window` and `tile(t, shape, index)` inherit
the bounds shifted to their origin, so a loop over sub-tiles checks each
against the same logical edge:

```julia
input = bound(GlobalTile(x, 16), (nothing, batch))   # the array as a K × batch tile
for k in lane:32:chunks-1
    v = load_fragment(Ownership((8,8)), tile(input, (8,8), (k,0)))   # absent columns read as zero
end
```

`tile(t, shape, index)` is the `index`-th tile of `shape` in a tiling of `t`,
as a tile of its own. Along a hierarchical mode the shape covers whole inner
modes, so the tile index runs over the outer ones without carries: for a K
order such as `(8,4,2,K÷64):(1,16,8,64)` the 8×8 tile `k` starts at
`16(k%4) + 8((k÷4)%2) + 64(k÷8)`, computed with masks and shifts. The tile's
declared alignment narrows to what its static outer strides keep.

`GlobalTile(A, align)` and `SharedTile(A, align)` take a device array as the
column-major matrix it is, with `Int32` extents; `SharedTile(T, pointer,
layout, align)` types a byte pointer, such as a workspace, at construction;
`permutedims(tile)` reads the same memory transposed, bounds included.

## Copy completion and buffer reuse

The mixed path does not change the caller's responsibilities. Commit and wait
for asynchronous copies, synchronize before other threads read shared memory,
and separately synchronize with those readers before reusing the buffer.
`wait_copies` completes asynchronous copies for the issuing thread; it is not a
CTA barrier and does not by itself publish scalar stores to other threads.
Every warp still participates fully in `ldmatrix` and MMA. Bounds predicates
select data and output stores, not participation in those instructions.

## A concrete edge

For an A matrix with mathematical shape `(65, 73)`, a `(64, 32)` copy at origin
`(64, 64)` has one valid row and nine valid K values. With an aligned source
address, the first eight BF16 values form one asynchronous vector. The next
vector contains one scalar load and seven zero stores; the rest of the shared
tile is zero. An unaligned source instead takes the scalar path for that first
vector too.

The destination may be a window inside a swizzled shared allocation. Its
pointer retains the original allocation base and its layout applies the window
origin before the swizzle. Zero-fill uses that same mapping, so it does not
reset the swizzle phase at the edge.

## Accumulator stores and GEMM

The corresponding bounded epilogue uses the accumulator's thread/value mapping:

```julia
store!(mma_plan, global_output, accumulator, (row, col), thread)
```

It converts FP32 accumulator values to the destination element type at the
store, suppresses invalid coordinates, and leaves out-of-bounds padding
unchanged. The output layout must assign distinct storage to the stored
elements. This form supports the existing tiled warp MMA distribution.

The GEMM example selects `gemm_config(...; bounds=true)` for boundary tiles and
uses ceiling division for its K loop and output grid. Full interior stages with
aligned leading strides take the original vector-copy path. `bounds=false` retains the
original aligned full-tile path and requires valid full tiles and aligned source
strides. The demo chooses between them from its dimensions:

```sh
julia --project=test examples/gemm/run.jl 65 97 73
julia --project=test examples/gemm/compare.jl /tmp/tylo-gemm-comparison
```

The comparison includes the original aligned kernel at a pinned Tylo revision,
the current aligned and bounded paths, and compact versus padded storage for
ragged shapes. Padding and transfer costs are excluded from kernel timing and
must be accounted for separately in an application. Boundary support is not a
claim that every irregular shape is efficient.
