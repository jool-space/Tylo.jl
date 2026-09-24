```@meta
CurrentModule = Tylo
DocTestSetup = :(using Tylo; using Tylo.Layouts: logical_divide, complement, left_inverse, static)
```

# Layouts and ownership

## Layouts

A [`Layout`](@ref Layouts.Layout) maps zero-based coordinates to offsets: a
tree of extents (its shape) and a matching tree of strides. Every leaf is
either static, a `StaticInt` known to the compiler, or an ordinary integer
known only at run time, and the two mix freely. `@Layout` writes static leaves by default and takes `$` for runtime
ones; layouts print the same way.

```jldoctest layouts
julia> l = @Layout (4, 8) (8, 1)   # a row-major 4×8 matrix
@Layout (4, 8) (8, 1)

julia> l((1, 2)), l(9)             # a coordinate, or a linear index, first mode fastest
(10, 10)

julia> n = 100; @Layout (4, $n) (1, 4)
@Layout (4, $100) (1, 4)
```

What the compiler can see decides what the code costs: a static stride of
one proves that elements are contiguous, which is what lets a load become
one 16-byte instruction; a runtime stride of one proves nothing. Matrix
dimensions and leading dimensions are usually runtime, tile shapes and
unit strides static.

### The algebra

`Tylo.Layouts` implements the layout algebra of CuTe, exactly and
symbolically: static inputs give static results, and each operation either
represents its result as a layout or throws.

```jldoctest layouts
julia> compose(@Layout((16,), (2,)), @Layout((4, 2), (2, 1)))   # a(b(c))
@Layout (4, 2) (4, 2)

julia> logical_divide(@Layout((16, 32), (32, 1)), (static(8), static(16)))   # (within tile, tile index) per mode
@Layout ((8, 2), (16, 2)) ((32, 256), (1, 16))

julia> complement(@Layout((4,), (2,)), static(16))   # the positions a layout leaves out
@Layout (2, 2) (1, 8)

julia> left_inverse(@Layout((4, 8), (8, 1)))
@Layout (8, 4) (4, 1)

julia> compose(@Layout((4, 4), (1, 8)), @Layout((3,), (2,)))
ERROR: ArgumentError: composition is not exact: extent 3 does not split evenly at the mode boundary 2
[...]
```

The full set is `compose`, `coalesce`, `complement`, `left_inverse`,
`right_inverse`, `logical_divide`, `zipped_divide`, `tiled_divide`,
`logical_product`, `blocked_product`, `raked_product` and `slice`; their
docstrings state the exact conditions. A [`Swizzle`](@ref Layouts.Swizzle)
composed with a layout XORs bit fields of its offsets, the bank-conflict
patterns of shared memory, and [`window`](@ref Layouts.window) selects a
static-size sub-layout at a runtime origin.

## Ownership

An [`Ownership`](@ref Layouts.Ownership) says which thread holds which
element of a logical `rows × cols` tile. It is a layout with two modes,
threads then slots, from `(thread, slot)` to the column-major index of the
element held there. It says nothing about memory: the same ownership can be
loaded from row-major or column-major storage, swizzled or not.

```jldoctest layouts
julia> o = distribute((64, 32), 128; vector=8, axis=2)   # 16-byte BF16 vectors along rows
Ownership((64, 32), @Layout ((4, 32), (8, 2)) ((512, 1), (64, 32)))

julia> coordinate(o, 5, 9)   # thread 5, slot 9: its second vector's second element
(33, 9)

julia> Ownership((1, 8); threads=32)   # every lane holds the whole row
Ownership((1, 8), @Layout (32, 8) (0, 1))
```

The constructors:

| Constructor | Distribution |
|:--|:--|
| `Ownership((m, n))` | one thread holds the whole tile, column-major slots |
| `Ownership((m, n); threads=k)` | each of `k` threads holds all of it |
| `distribute((m, n), threads; vector, axis)` | vectors along `axis` dealt to threads in order: copies, lane-per-row, warp-striped rows |
| `operand_layout(atom, role)` | what an instruction prescribes for an operand |
| `Ownership((m, n), mapping)` | anything else, as a layout |

and the transformations `permutedims(o)` (the transposed tile, the same
registers), `repeat(o, m, n)` (every thread holds its values of an `m × n`
grid of copies) and, on fragments, `window` and `relayout`.

An ownership may replicate: a zero-stride thread mode means several threads
hold the same values. Reductions never combine a value with its copies, a
store writes each value once, and a copy issues it once.

The constructor stores the mapping in a canonical form, so two ownerships
with the same distribution have the same type, whichever way they were
written, and transposing twice gives back the original.

Every plan Tylo derives, which slots form a vector, which lanes a
reduction shuffles with, which `ldmatrix` covers a fragment, starts from
this mapping. The tests check each derivation against an enumeration of
every `(thread, slot)` pair, for the atoms' ownerships and for random ones.
