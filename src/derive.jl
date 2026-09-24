# Derivations from ownership layouts.
#
# Every ownership answers `mapping(o)`: a two-mode layout from `(thread,
# slot)` to the column-major index of the owned logical coordinate. The
# collective recipes derive from that layout by algebra: projecting onto an
# axis, composing with a tile or unit layout, dividing into copy-atom
# blocks, inverting a value layout. `ownership_table` enumerates the device
# `coordinate` method instead; it is the oracle the host suite checks the
# mappings and the derived plans against. Nothing here runs on the device;
# generated functions call it while their bodies are being constructed.

using .Layouts: Layout, Ownership, static, compose, concat, coalesce, flatten, filter_zeros, mode,
    zipped_divide, left_inverse, agree, is_compact, trivial, cosize

"""
    mapping(ownership) -> Layout

The layout from `(thread, slot)` to the column-major index of the logical
coordinate that thread holds in that slot, over `size(ownership)`. Mode 1
is the thread tree, mode 2 the slot tree. Every derivation below starts here.
"""
mapping(o::Ownership) = o.mapping
mapping(::Layouts.LocalOwnership{N,2}) where N = Layout((static(32),static(N)),(static(1),static(32)))
mapping(::Layouts.LocalOwnership{N,1}) where N = Layout((static(32),static(N)),(static(N),static(1)))
mapping(::Layouts.StripedOwnership{N}) where N = Layout((static(32),static(N)),(static(1),static(32)))
mapping(::TmemTransfer{S,2}) where S = Layout((static(32),static(S[2])),(static(1),static(32)))
mapping(::TmemTransfer{S,1}) where S = Layout((static(32),static(S[1])),(static(S[1]),static(1)))
# PTX ISA wgmma accumulator figure: lane groups t%4 select column pairs,
# (t%32)÷4 the row, warps 16-row bands; slots alternate column, row half, 8-column block.
mapping(::WGMMAOwnership{N}) where N =
    Layout(((static(4),static(8),static(4)),(static(2),static(2),static(N÷8))),
           ((static(128),static(1),static(16)),(static(64),static(8),static(512))))
function mapping(o::PermutedOwnership)
    m, n = map(Int,size(o.parent))
    compose(Layout((static(m),static(n)),(static(n),static(1))),mapping(o.parent))
end
# The tiled accumulator is the atom's ownership embedded in the CTA tile,
# then repeated: warps M-fastest across the tile, atoms M-fastest within a warp.
function mapping(::TiledMMAOwnership{TiledMMA{A,W,R,K}}) where {A,W,R,K}
    atom = operand_layout(A(),Accumulator())
    m, n = size(atom)
    M = m*W[1]*R[1]
    embedded = compose(Layout((static(m),static(n)),(static(1),static(M))),mapping(atom))
    threads = concat(mode(embedded,1),Layout((static(W[1]),static(W[2])),(static(m*R[1]),static(n*R[2]*M))))
    values = concat(mode(embedded,2),Layout((static(R[1]),static(R[2])),(static(m),static(n*M))))
    concat(threads,values)
end
_thread_count(o) = Int(length(mode(mapping(o),1)))
_threads(o) = mode(mapping(o),1)
_slots(o) = mode(mapping(o),2)

"""
    ownership_table(ownership) -> Matrix{Tuple{Int,Int}}

Logical coordinate of every `(thread, slot)` pair, zero-based, as a matrix
indexed by `(thread+1, slot+1)`, enumerated from the device `coordinate`
method. This is the oracle for `mapping`, not an input to the derivations.
"""
const _TABLES = Dict{Any,Matrix{Tuple{Int,Int}}}()
const _TABLE_LOCK = ReentrantLock()
function ownership_table(o)
    lock(_TABLE_LOCK) do
        get!(_TABLES,o) do
            threads, slots = _thread_count(o), _register_count(o)
            [map(Int,Layouts.coordinate(o,t,Val(e))) for t in 0:threads-1, e in 0:slots-1]
        end
    end
end

"The ownership instance of a singleton ownership type, or `nothing`."
_static_instance(::Type{T}) where T = Base.issingletontype(T) ? T.instance : nothing

# Two layouts are the same function when their coalesced forms agree; the
# thread and slot modes are compared separately so that thread counts matter.
_same_layout(a,b) = Int(length(a)) == Int(length(b)) && coalesce(a) == coalesce(b)
"Every logical coordinate of the ownership's shape is held by some thread."
is_complete(o) = Set(Int(mapping(o)(i)) for i in 0:Int(length(mapping(o)))-1) == Set(0:prod(map(Int,size(o)))-1)
"No logical coordinate is held twice."
is_injective(o) = Layouts.is_injective(mapping(o))
"Two ownerships assign the same coordinates to the same thread slots."
same_distribution(a,b) = map(Int,size(a)) == map(Int,size(b)) &&
    _same_layout(_threads(a),_threads(b)) && _same_layout(_slots(a),_slots(b))

# Projections of an ownership onto one logical axis: the coordinate along
# `axis` as a layout over the same thread and slot trees.
function _axis(o,axis::Int)
    m, n = map(Int,size(o))
    compose(Layout((static(m),static(n)),axis == 1 ? (static(1),static(0)) : (static(0),static(1))),mapping(o))
end
_pairs(l::Layout) = collect(zip(map(Int,Layouts.shape(flatten(l))),map(Int,strides(flatten(l)))))
# Composition that reports an inexact case as `nothing` instead of throwing.
function _compose(a,b)
    try
        compose(a,b)
    catch e
        e isa ArgumentError ? nothing : rethrow()
    end
end
function _left_inverse(l)
    try
        left_inverse(l)
    catch e
        e isa ArgumentError ? nothing : rethrow()
    end
end

"""
    ReductionPlan

A thread-uniform recipe for reducing one logical axis of an ownership.
`groups` lists the slots (1-based) that share each kept coordinate within a
thread; `bits` lists thread-index bits whose flip preserves the kept
coordinate, so an xor shuffle over each of them completes the reduction;
`result` is the replicated result ownership.
"""
struct ReductionPlan{O}
    groups::Vector{Vector{Int}}
    bits::Vector{Int}
    result::O
end

"""
    reduction_plan(ownership, axis) -> ReductionPlan or nothing

Derive the collective reduction of logical `axis` for an ownership, or
`nothing` when no local-tree-plus-xor-shuffle recipe reaches every value
sharing a kept coordinate with an identical instruction sequence on all
threads. The kept coordinate is the ownership projected onto the other
axis; slots that differ only in that projection's zero-stride slot modes
share a group, thread modes with zero stride are the shuffle bits, and the
projection must be injective in its remaining modes so that nothing else
holds the same coordinate.
"""
function reduction_plan(o, axis::Int)
    axis in (1,2) || throw(ArgumentError("choose logical axis 1 or 2"))
    threads = _thread_count(o)
    ispow2(threads) || return nothing
    kept = _axis(o,3-axis)
    Kt, Kv = mode(kept,1), mode(kept,2)
    Kvn = filter_zeros(Kv)
    Layouts.is_injective(Kvn) || return nothing
    Layouts.is_injective(concat(filter_zeros(Kt),Kvn)) || return nothing
    slots = Int(length(Kv))
    groups = Vector{Int}[]
    keys = Int[]
    for e in 0:slots-1
        k = Int(Kv(e))
        g = findfirst(==(k),keys)
        g === nothing ? (push!(groups,[e+1]); push!(keys,k)) : push!(groups[g],e+1)
    end
    # Shuffles exchange values within a warp only: bits 0:4 of the thread
    # index. A replicating mode above them spans warps and has no recipe.
    bits = Int[]
    prefix = 1
    for (s,d) in _pairs(Kt)
        if d == 0 && s > 1
            low, high = trailing_zeros(prefix), trailing_zeros(prefix*s)
            high <= 5 || return nothing
            append!(bits,low:high-1)
        end
        prefix *= s
    end
    shape = axis == 2 ? (Int(size(o)[1]),1) : (1,Int(size(o)[2]))
    ReductionPlan(groups,bits,Ownership(Val(shape),concat(Kt,Kvn)))
end

"""
    broadcast_slots(x, anchor) -> Vector{Int} or nothing

For each slot of `anchor`, the slot of `x` supplying its value under array
broadcasting: `x` has the anchor's logical shape, or extent one on the axes
it broadcasts along. The map is the anchor's slot layout, projected onto
`x`'s shape, composed with the inverse of `x`'s slot layout; it exists when
both ownerships distribute threads alike.
"""
function broadcast_slots(x, anchor)
    _thread_count(x) == _thread_count(anchor) || return nothing
    sx, sa = map(Int,size(x)), map(Int,size(anchor))
    sx == sa && return same_distribution(x,anchor) ? collect(1:_register_count(anchor)) : nothing
    all(map((nx,na) -> nx == na || nx == 1, sx, sa)) || return nothing
    onto = Layout((static(sa[1]),static(sa[2])),(static(sx[1] == 1 ? 0 : 1),static(sx[2] == 1 ? 0 : sx[1])))
    projected = _compose(onto,mapping(anchor))
    projected === nothing && return nothing
    _same_layout(mode(projected,1),_threads(x)) || return nothing
    _slot_map(_slots(x),mode(projected,2))
end
# The slot layout `to` expressed through slots of `from`: `from ∘ map == to`.
function _slot_map(from,to)
    inverse = _left_inverse(from)
    inverse === nothing && return nothing
    slots = _compose(inverse,to)
    slots === nothing && return nothing
    agree(compose(from,slots),to) || return nothing
    [Int(slots(e))+1 for e in 0:Int(length(to))-1]
end

"""
    simplify_ownership(ownership)

The lane-local or warp-striped ownership with the same distribution, when
one exists; otherwise the ownership itself.
"""
function simplify_ownership(o)
    _thread_count(o) == 32 || return o
    n = _register_count(o)
    for candidate in (Layouts.LocalOwnership{n,2}(),Layouts.LocalOwnership{n,1}(),
                      Layouts.StripedOwnership{n,2}(),Layouts.StripedOwnership{n,1}())
        size(candidate) == map(Int,size(o)) && same_distribution(candidate,o) && return candidate
    end
    o
end

"""
    relayout_permutation(from, to) -> Vector{Int} or nothing

For each slot of `to`, the slot of `from` holding the same coordinate in the
same thread: the inverse of `from`'s slot layout composed with `to`'s, when
both distribute threads alike and the map is a permutation.
"""
function relayout_permutation(from, to)
    _thread_count(from) == _thread_count(to) || return nothing
    map(Int,size(from)) == map(Int,size(to)) || return nothing
    _register_count(from) == _register_count(to) || return nothing
    _same_layout(_threads(from),_threads(to)) || return nothing
    permutation = _slot_map(_slots(from),_slots(to))
    permutation === nothing || !allunique(permutation) ? nothing : permutation
end

"""
    window_plan(ownership, origin, shape) -> (; slots, ownership) or nothing

The slots (1-based) falling inside a logical window, identical on every
thread, and the window's own ownership with coordinates relative to
`origin`. A slot is inside when its coordinate is inside for the first and
the last thread; the inside slots must form a box in the slot modes so that
the window is a layout of its own, composed with the ownership re-indexed
to the window shape.
"""
function window_plan(o, origin::Tuple{Int,Int}, shape::Tuple{Int,Int})
    M, N = map(Int,size(o))
    all(map((lo,n,full) -> 0 <= lo && 0 < n && lo+n <= full, origin, shape, (M,N))) || return nothing
    rows, cols = _axis(o,1), _axis(o,2)
    span_r, span_c = Int(cosize(mode(rows,1)))-1, Int(cosize(mode(cols,1)))-1
    Rv, Cv = mode(rows,2), mode(cols,2)
    inside(e) = origin[1] <= Int(Rv(e)) && Int(Rv(e))+span_r < origin[1]+shape[1] &&
                origin[2] <= Int(Cv(e)) && Int(Cv(e))+span_c < origin[2]+shape[2]
    slots = [e for e in 0:_register_count(o)-1 if inside(e)]
    isempty(slots) && return nothing
    selection = _box(_pairs(_slots(o)),slots)
    selection === nothing && return nothing
    reindexed = _compose(Layout((static(M),static(N)),(static(1),static(shape[1]))),mapping(o))
    reindexed === nothing && return nothing
    Int(reindexed((0,slots[1]))) == origin[1] + shape[1]*origin[2] || return nothing
    values = _compose(mode(reindexed,2),selection)
    values === nothing && return nothing
    (; slots=slots .+ 1, ownership=Ownership(Val(shape),concat(mode(reindexed,1),values)))
end
# The layout selecting `chosen` slots as a box of digit ranges over flat
# slot modes with the given `(extent, stride)` pairs, or `nothing`.
function _box(pairs,chosen)
    extents = first.(pairs)
    digits(e) = [(e ÷ prod(extents[1:i-1])) % extents[i] for i in 1:length(extents)]
    ds = map(digits,chosen)
    lo = [minimum(d[i] for d in ds) for i in 1:length(extents)]
    hi = [maximum(d[i] for d in ds) for i in 1:length(extents)]
    prod(hi .- lo .+ 1) == length(chosen) || return nothing
    prefix = cumprod([1; extents[1:end-1]])
    sum(lo .* prefix) == chosen[1] || return nothing
    keep = [i for i in 1:length(extents) if hi[i] > lo[i]]
    isempty(keep) && return trivial()
    Layout(Tuple(static(hi[i]-lo[i]+1) for i in keep),Tuple(static(prefix[i]) for i in keep))
end

# Matrix copies: cover a 32-lane ownership with the 8×8 blocks of a
# `CopyAtom`. Coordinates are converted to memory coordinates (rows along
# the non-contiguous axis, 16-bit units along the contiguous one) so the
# same derivation serves K-major and MN-major storage of either operand.
_contiguous_axis(::Type) = nothing
function _contiguous_axis(::Type{Layouts.Layout{S,D}}) where {S,D}
    D <: Tuple && length(D.parameters) == 2 || return nothing
    unit = findall(p -> p === Layouts.StaticInt{1},collect(D.parameters))
    length(unit) == 1 ? unit[1] : nothing
end
_contiguous_axis(::Type{Layouts.Composition{F,L}}) where {F,L} = _contiguous_axis(L)
_contiguous_axis(::Type{Layouts.Window{S,L,O}}) where {S,L,O} = _contiguous_axis(L)

"""
    MatrixCopyPlan

The derived decomposition of a 32-lane ownership into `CopyAtom` blocks.
`trans` selects the transposed instruction, `axis` is the logical axis that
memory stores contiguously, `per_unit` the elements per 16-bit unit, `grid`
the block counts along memory rows and columns, and `words[b]` the
fragment word (1-based) that block `b` transfers, with blocks numbered
column-major over the grid. Consecutive blocks group into instructions of
up to four matrices.
"""
struct MatrixCopyPlan
    trans::Bool
    axis::Int
    per_unit::Int
    grid::Tuple{Int,Int}
    words::Vector{Int}
end

"""
    matrix_copy_plan(ownership, T, axis) -> MatrixCopyPlan or nothing

Cover a 32-lane ownership of `T` elements with 8×8 `ldmatrix`/`stmatrix`
blocks, given the logical `axis` that memory stores contiguously. The
ownership is composed with the layout of 16-bit units, which drops the
element within a unit, and then with the inverse of the unit grid divided
into 8×8 blocks. In that form every lane must hold whole units, the thread
and word-unit modes must equal a copy atom's register ownership of one
block, plain or transposed, and the remaining slot modes must enumerate the
blocks exactly once. Otherwise there is no plan and scalar accesses remain
the correct route.
"""
function matrix_copy_plan(o,::Type{T},axis::Int) where T
    axis in (1,2) || throw(ArgumentError("choose logical axis 1 or 2"))
    _thread_count(o) == 32 || return nothing
    bits = _element_bits(T)
    bits in (8,16) || return nothing
    per = 16 ÷ bits
    M, N = map(Int,size(o))
    (M,N)[axis] % per == 0 || return nothing
    rows, cols = axis == 1 ? (N,M÷per) : (M,N÷per)
    rows % 8 == 0 && cols % 8 == 0 || return nothing
    units = axis == 2 ? Layout((static(M),(static(per),static(cols))),(static(1),(static(0),static(M)))) :
                        Layout(((static(per),static(cols)),static(N)),((static(0),static(rows)),static(1)))
    U = _compose(units,mapping(o))
    U === nothing && return nothing
    # Whole units per lane: the dropped element position is the fastest
    # slot mode and nothing else replicates a unit.
    value = _pairs(mode(U,2))
    per == 1 || (value[1] == (per,0) && all(d != 0 for (_,d) in value[2:end])) || return nothing
    all(d != 0 for (_,d) in _pairs(mode(U,1))) || return nothing
    words = filter_zeros(mode(U,2))
    Int(length(words)) % 2 == 0 || return nothing
    blocks = zipped_divide(Layout((static(rows),static(cols)),(static(1),static(rows))),(static(8),static(8)))
    Q = _compose(left_inverse(blocks),concat(mode(U,1),words))
    Q === nothing && return nothing
    qv = _pairs(mode(Q,2))
    qv[1][1] == 2 && qv[1][2] < 64 && all(d % 64 == 0 for (_,d) in qv[2:end]) || return nothing
    within = concat(mode(Q,1),Layout((static(2),),(static(qv[1][2]),)))
    grid = (rows÷8,cols÷8)
    index = length(qv) > 1 ? Layout(Tuple(static(s) for (s,_) in qv[2:end]),Tuple(static(d÷64) for (_,d) in qv[2:end])) : trivial()
    Int(length(index)) == prod(grid) && is_compact(index) || return nothing
    inverse = _left_inverse(index)
    inverse === nothing && return nothing
    for trans in (false,true)
        atom = mapping(operand_layout(CopyAtom{:load,trans}(),Registers()))
        _same_layout(mode(within,1),mode(atom,1)) && _same_layout(mode(within,2),mode(atom,2)) || continue
        return MatrixCopyPlan(trans,axis,per,grid,[Int(inverse(b))+1 for b in 0:prod(grid)-1])
    end
    nothing
end
"Consecutive block ranges, each one instruction of up to four matrices."
function _matrix_groups(n)
    groups = UnitRange{Int}[]
    start = 1
    while start <= n
        width = n-start+1 >= 4 ? 4 : n-start+1 >= 2 ? 2 : 1
        push!(groups,start:start+width-1)
        start += width
    end
    groups
end

"""
    validate_copy(ownership, T, layout)

Check on the host that the matrix copy derived for a 32-lane `ownership`
over a shared `layout` addresses whole, contiguous, 16-byte aligned rows for
every lane. Throws when no copy derives or a row is broken; returns `nothing`.
"""
function validate_copy(o,::Type{T},l::Layouts.AbstractLayout) where T
    axis = _contiguous_axis(typeof(l))
    axis === nothing && throw(ArgumentError("the layout type has no static unit-stride axis"))
    map(Int,size(l)) == map(Int,size(o)) || throw(DimensionMismatch("ownership and layout shapes differ"))
    plan = matrix_copy_plan(o,T,axis)
    plan === nothing && throw(ArgumentError("no matrix copy covers this ownership"))
    for group in _matrix_groups(length(plan.words)), lane in 0:31
        c = _matrix_address(Val(first(group)-1),Val(length(group)),Val(plan.grid[1]),Val(axis),Val(8plan.per_unit),lane)
        _check_vector(l,c,axis,Val(8plan.per_unit),T)
    end
    nothing
end

"""
    vector_plan(ownership, T, axis, align) -> (; width, groups) or nothing

Group each thread's slots into vectors of `width` bytes (16, 8 or 4) that
are contiguous along logical `axis` and start at coordinates whose byte
offset is a multiple of the width, identically on every thread, given a tile
alignment of `align` bytes. In the ownership projected onto the axis, the
fastest slot mode must run along it with unit stride for a multiple of the
vector length, and every other stride must be a multiple of that length.
The widest such grouping wins; `nothing` when even the narrowest
multi-element vector does not fit.
"""
function vector_plan(o,::Type{T},axis::Int,align::Int) where T
    axis in (1,2) || throw(ArgumentError("choose logical axis 1 or 2"))
    along = _axis(o,axis)
    value = _pairs(coalesce(mode(along,2)))
    thread = _pairs(mode(along,1))
    slots = _register_count(o)
    for width in (16,8,4)
        width <= align || continue
        k = width ÷ sizeof(T)
        k >= 2 && slots % k == 0 || continue
        value[1][2] == 1 && value[1][1] % k == 0 || continue
        all(d % k == 0 for (_,d) in value[2:end]) && all(d % k == 0 for (_,d) in thread) || continue
        return (; width, groups=[collect(g:g+k-1) for g in 1:k:slots])
    end
    nothing
end
