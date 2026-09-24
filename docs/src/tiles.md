```@meta
CurrentModule = Tylo
DocTestSetup = :(using Tylo)
```

# Memory tiles

A memory tile is a typed pointer and a layout: the layout maps a logical
coordinate to an element offset from the pointer. [`GlobalTile`](@ref) and
`SharedTile` borrow global and shared memory; [`TmemTile`](@ref)
borrows tensor memory (see [Hardware operations](operations.md#Tensor-memory)).
Allocation, lifetime and synchronization stay with the kernel.

```julia
a = GlobalTile(A)                          # a device matrix as the column-major tile it is
a = GlobalTile(A, 16)                      # ... declaring 16-byte alignment
s = SharedTile(pointer(smem), layout, 16)  # any layout over shared memory
w = GlobalTile(Float8_E4M3FN, workspace + offset, layout, 16)   # a typed view of a byte pointer
```

## Alignment is declared

The last argument declares, in bytes, that the base address and every
runtime stride are multiples of it; the default is the element size. Loads,
stores and copies form vectors up to the declared alignment and never
beyond, so a tile declared 16-byte aligned with a static unit stride along
its rows moves 16 bytes per instruction, and the same tile declared with
the default alignment moves one element per instruction. The declaration is
checked when the tile is built, unless elided with `@inbounds` (for
example, where the host has already validated the leading dimensions).

```jldoctest tiles
julia> p = reinterpret(Core.LLVMPtr{Float32,1}, UInt64(0x10000));

julia> a = GlobalTile(p, @Layout(($(Int32(100)), 32), ($(Int32(32)), 1)), 16)
GlobalTile(Float32, 0x10000, @Layout ($100, 32) ($32, 1), 16)

julia> window(a, (Int32(0), Int32(3)), (16, 8))   # starts 12 bytes into a row
ERROR: ArgumentError: pointer, strides or window origin break the declared alignment
[...]
```

## Sub-tiles and bounds

Kernels work on tiles of static shape at runtime positions. Three
operations produce them, each a tile of its own over the same allocation:

- `window(t, origin, shape)`: the `shape` sub-tile at a coordinate.
- `tile(t, shape, index)`: the `index`-th tile of a tiling by `shape`. On a
  hierarchical layout it addresses whole inner modes, so no index is ever
  divided at run time.
- `bound(t, extents)`: the same tile, valid only below `extents` (`nothing`
  leaves an axis unchecked).

A window or tile must lie inside its parent, checked unless elided with
`@inbounds`, except along a bounded axis. There it may extend past the
parent: its bound, shifted with it, says which elements exist. Loads through
a bounded tile read zeros outside the bound, stores skip it, and copies
zero-fill it. That is how an edge tile is written: the same code as an
interior tile, through a bounded parent.

```jldoctest tiles
julia> b = bound(a, size(a))
bound(GlobalTile(Float32, 0x10000, @Layout ($100, 32) ($32, 1), 16), (100, 32))

julia> tile(b, (16, 32), (Int32(6), Int32(0)))   # rows 96 to 111, of which 4 exist
bound(GlobalTile(Float32, 0x13000, @Layout (16, 32) ($32, 1), 16), (4, 32))
```

Bounds apply a vector at a time. A vector whose elements are consecutive
along a bounded axis is cut where the bound falls: the elements before it
are read or written, the rest are zero or skipped. A vector that gathers
coordinates out of order (a permuted K order, say) must lie wholly inside or
outside the bound, which is checked.

`permutedims(t)` is the transposed tile over the same memory, bounds
included; `pointer(t, coordinate)` is an element's address.

## Checks and `@inbounds`

Every access Tylo can check, it checks: a sub-tile's extent, an alignment
declaration, a thread index against its ownership, a tile's shape against a
fragment's. Each check is a `@boundscheck`, so a kernel that has
established its invariants elides them with `@inbounds`, as it would for
array indexing, and compiles to straight-line code. Checks that the types
decide, such as an element type mismatch or an ownership no instruction
implements, are always errors and cost nothing.
