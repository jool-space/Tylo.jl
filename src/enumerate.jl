# Host-side enumeration of thread/value ownership.
#
# Every ownership answers `coordinate(o, thread, Val(slot))`. Enumerating that
# map over all threads and slots gives explicit tables from which collective
# recipes can be derived at generated-function time: which slots of a thread
# share a row, which thread-index bits replicate a row, whether two
# ownerships permit an in-lane permutation, and which slots fall inside a
# logical window. Nothing here runs on the device; generated functions call
# it while their bodies are being constructed.

_thread_count(::Layouts.LocalOwnership) = 32
_thread_count(::Layouts.StripedOwnership) = 32
_thread_count(o::Layouts.Ownership) = Int(size(o.mapping)[1])
_thread_count(::TiledMMAOwnership{TiledMMA{A,W,R,K}}) where {A,W,R,K} = 32prod(W)
_thread_count(::WGMMAOwnership) = 128
_thread_count(::TmemTransfer) = 32
_thread_count(o::PermutedOwnership) = _thread_count(o.parent)

"""
    ownership_table(ownership) -> Matrix{Tuple{Int,Int}}

Logical coordinate of every `(thread, slot)` pair, zero-based, as a matrix
indexed by `(thread+1, slot+1)`.
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

"Every logical coordinate of the ownership's shape is held by some thread."
is_complete(o) = Set(ownership_table(o)) == Set((r,c) for r in 0:size(o)[1]-1, c in 0:size(o)[2]-1)
"No logical coordinate is held twice."
is_injective(o) = allunique(ownership_table(o))
"Two ownerships assign the same coordinates to the same thread slots."
same_distribution(a,b) = _thread_count(a) == _thread_count(b) &&
    _register_count(a) == _register_count(b) && ownership_table(a) == ownership_table(b)

# Affine fit of a coordinate table. Thread and slot indices are decomposed
# into mixed-radix modes; every ordering of each count's prime factors is a
# candidate, so a 24-slot ownership may decompose as (2,2,2,3) or (2,2,3,2).
# A fit that does not reproduce the whole table exactly is rejected, so
# nonlinear ownerships never receive an affine description.
function _factors(n)
    factors = Int[]
    d = 2
    while n > 1
        while n % d == 0
            push!(factors,d); n ÷= d
        end
        d += 1
    end
    factors
end
function _orderings(items)
    length(items) <= 1 && return [Tuple(items)]
    result = Tuple{Vararg{Int}}[]
    for value in unique(items)
        rest = copy(items); deleteat!(rest,findfirst(==(value),rest))
        append!(result,[(value,tail...) for tail in _orderings(rest)])
    end
    result
end
_shape_candidates(n) = n == 1 ? [(1,)] : _orderings(_factors(n))
function _digits(x, shape)
    out = Int[]
    for n in shape
        push!(out, x % n); x ÷= n
    end
    out
end
function _fit_index(index::Matrix{Int})
    threads, slots = size(index)
    ispow2(threads) || return nothing
    index[1,1] == 0 || return nothing
    for tshape in _shape_candidates(threads), sshape in _shape_candidates(slots)
        tstrides = [tshape[b] == 1 ? 0 : index[1+prod(tshape[1:b-1]),1] for b in 1:length(tshape)]
        sstrides = [sshape[b] == 1 ? 0 : index[1,1+prod(sshape[1:b-1])] for b in 1:length(sshape)]
        exact = all(index[t+1,e+1] == sum(_digits(t,tshape) .* tstrides) + sum(_digits(e,sshape) .* sstrides)
                    for t in 0:threads-1, e in 0:slots-1)
        exact && return (tshape, Tuple(tstrides), sshape, Tuple(sstrides))
    end
    nothing
end

"""
    fit_ownership(table, shape) -> Ownership or nothing

An explicit `Layouts.Ownership` reproducing `table` over a logical `shape`,
when the table is affine in the bits of the thread and slot indices.
"""
function fit_ownership(table::Matrix{Tuple{Int,Int}}, shape::Tuple{Int,Int})
    index = [r + shape[1]*c for (r,c) in table]
    fit = _fit_index(index)
    fit === nothing && return nothing
    tshape, tstrides, sshape, sstrides = fit
    s = map(Layouts.static,tshape), map(Layouts.static,sshape)
    d = map(Layouts.static,tstrides), map(Layouts.static,sstrides)
    Layouts.Ownership(Val(shape), Layouts.Layout(s,d))
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
threads.
"""
function reduction_plan(o, axis::Int)
    axis in (1,2) || throw(ArgumentError("choose logical axis 1 or 2"))
    table = ownership_table(o)
    threads, slots = size(table)
    ispow2(threads) || return nothing
    kept = 3-axis
    key(t,e) = table[t+1,e+1][kept]
    groups = Vector{Int}[]
    for e in 1:slots
        g = findfirst(g -> key(0,g[1]-1) == key(0,e-1), groups)
        g === nothing ? push!(groups,[e]) : push!(groups[g],e)
    end
    for t in 0:threads-1
        all(g -> all(e -> key(t,e-1) == key(t,g[1]-1), g), groups) || return nothing
        allunique(key(t,g[1]-1) for g in groups) || return nothing
    end
    # Shuffles exchange values within a warp only: bits 0:4 of the thread
    # index. A kept coordinate whose holders span warps has no recipe here.
    bits = [b for b in 0:min(4,trailing_zeros(threads)-1) if
            all(key(t ⊻ (1<<b),g[1]-1) == key(t,g[1]-1) for t in 0:threads-1, g in groups)]
    masks = [m for m in 0:threads-1 if m & ~sum(1<<b for b in bits; init=0) == 0]
    for t in 0:threads-1, g in groups
        k = key(t,g[1]-1)
        reachable = Set((t ⊻ m, e) for m in masks, e in g)
        holders = Set((t2,e2) for t2 in 0:threads-1, e2 in 1:slots if key(t2,e2-1) == k)
        reachable == holders || return nothing
    end
    reduced = [axis == 2 ? (key(t,g[1]-1),0) : (0,key(t,g[1]-1))
               for t in 0:threads-1, g in groups]
    shape = axis == 2 ? (size(o)[1],1) : (1,size(o)[2])
    result = fit_ownership(reduced,shape)
    result === nothing && return nothing
    ReductionPlan(groups,bits,result)
end

"""
    broadcast_slots(x, anchor) -> Vector{Int} or nothing

For each slot of `anchor`, the slot of `x` supplying its value under array
broadcasting: `x` has the anchor's logical shape, or extent one on the axes
it broadcasts along. The mapping must be identical on every thread.
"""
function broadcast_slots(x, anchor)
    _thread_count(x) == _thread_count(anchor) || return nothing
    sx, sa = map(Int,size(x)), map(Int,size(anchor))
    sx == sa && return same_distribution(x,anchor) ? collect(1:_register_count(anchor)) : nothing
    all(map((nx,na) -> nx == na || nx == 1, sx, sa)) || return nothing
    tx, ta = ownership_table(x), ownership_table(anchor)
    project(c) = map((v,nx) -> nx == 1 ? 0 : v, c, sx)
    slots = [findfirst(==(project(ta[1,e])),tx[1,:]) for e in 1:size(ta,2)]
    any(isnothing,slots) && return nothing
    for t in 2:size(ta,1)
        all(tx[t,slots[e]] == project(ta[t,e]) for e in 1:size(ta,2)) || return nothing
    end
    slots
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
same thread, when that permutation is identical on every thread.
"""
function relayout_permutation(from, to)
    _thread_count(from) == _thread_count(to) || return nothing
    a, b = ownership_table(from), ownership_table(to)
    size(a,2) == size(b,2) || return nothing
    permutation = [findfirst(==(b[1,j]),a[1,:]) for j in 1:size(b,2)]
    any(isnothing,permutation) && return nothing
    allunique(permutation) || return nothing
    for t in 2:size(a,1)
        all(a[t,permutation[j]] == b[t,j] for j in 1:size(b,2)) || return nothing
    end
    permutation
end

"""
    window_plan(ownership, origin, shape) -> (; slots, ownership) or nothing

The slots (1-based) falling inside a logical window, identical on every
thread, and the window's own ownership with coordinates relative to `origin`.
"""
function window_plan(o, origin::Tuple{Int,Int}, shape::Tuple{Int,Int})
    all(map((lo,n,full) -> 0 <= lo && 0 < n && lo+n <= full, origin, shape, map(Int,size(o)))) || return nothing
    table = ownership_table(o)
    inside(c) = all(map((x,lo,n) -> lo <= x < lo+n, c, origin, shape))
    slots = [e for e in 1:size(table,2) if inside(table[1,e])]
    isempty(slots) && return nothing
    for t in 1:size(table,1)
        [e for e in 1:size(table,2) if inside(table[t,e])] == slots || return nothing
    end
    shifted = [map(-,table[t,e],origin) for t in 1:size(table,1), e in slots]
    ownership = fit_ownership(shifted,shape)
    ownership === nothing && return nothing
    (; slots, ownership)
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
_matrix_pattern(trans,t,e) = trans ? (2(t%4)+e,t÷4) : (t÷4,2(t%4)+e)

"""
    matrix_copy_plan(ownership, T, axis) -> MatrixCopyPlan or nothing

Cover a 32-lane ownership of `T` elements with 8×8 `ldmatrix`/`stmatrix`
blocks, given the logical `axis` that memory stores contiguously. Every
lane must hold complete words of adjacent elements along that axis, every
block must be held by the same two units of every lane, and those units
must follow the atom's register pattern, plain or transposed. Otherwise
there is no plan and scalar accesses remain the correct route.
"""
function matrix_copy_plan(o,::Type{T},axis::Int) where T
    axis in (1,2) || throw(ArgumentError("choose logical axis 1 or 2"))
    _thread_count(o) == 32 || return nothing
    bits = _element_bits(T)
    bits in (8,16) || return nothing
    per = 16 ÷ bits
    table = ownership_table(o)
    slots = size(table,2)
    slots % 2per == 0 || return nothing
    nunits = slots ÷ per
    units = Matrix{Tuple{Int,Int}}(undef,32,nunits)
    for t in 1:32, u in 1:nunits
        first = table[t,per*(u-1)+1]
        first[axis] % per == 0 || return nothing
        for j in 1:per-1
            c = table[t,per*(u-1)+1+j]
            c[axis] == first[axis]+j && c[3-axis] == first[3-axis] || return nothing
        end
        units[t,u] = (first[3-axis],first[axis] ÷ per)
    end
    rows, cols = Int(size(o)[3-axis]), Int(size(o)[axis]) ÷ per
    rows % 8 == 0 && cols % 8 == 0 || return nothing
    grid = (rows÷8,cols÷8)
    inside(t,br,bc) = [u for u in 1:nunits if 8br <= units[t,u][1] < 8br+8 && 8bc <= units[t,u][2] < 8bc+8]
    for trans in (false,true)
        words = Int[]
        fits = true
        for bc in 0:grid[2]-1, br in 0:grid[1]-1
            us = inside(1,br,bc)
            length(us) == 2 && us[2] == us[1]+1 && isodd(us[1]) &&
                all(inside(t,br,bc) == us for t in 2:32) || (fits = false; break)
            all(units[t+1,us[1]+e] == (8br,8bc) .+ _matrix_pattern(trans,t,e) for t in 0:31, e in 0:1) ||
                (fits = false; break)
            push!(words,(us[1]+1)÷2)
        end
        fits && allunique(words) && length(words) == nunits÷2 &&
            return MatrixCopyPlan(trans,axis,per,grid,words)
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
alignment of `align` bytes. The widest such grouping wins; `nothing` when
even the narrowest multi-element vector does not fit.
"""
function vector_plan(o,::Type{T},axis::Int,align::Int) where T
    axis in (1,2) || throw(ArgumentError("choose logical axis 1 or 2"))
    table = ownership_table(o)
    threads, slots = size(table)
    for width in (16,8,4)
        width <= align || continue
        k = width ÷ sizeof(T)
        k >= 2 && slots % k == 0 || continue
        groups = [collect(g:g+k-1) for g in 1:k:slots]
        fits = all(begin
            c0 = table[t,g[1]]
            (c0[axis]*sizeof(T)) % width == 0 &&
                all(table[t,g[j]] == (axis == 1 ? (c0[1]+j-1,c0[2]) : (c0[1],c0[2]+j-1)) for j in 2:k)
        end for t in 1:threads, g in groups)
        fits && return (; width, groups)
    end
    nothing
end
