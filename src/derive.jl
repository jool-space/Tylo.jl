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
_thread_count(o) = Int(length(mode(mapping(o),1)))
_threads(o) = mode(mapping(o),1)
_slots(o) = mode(mapping(o),2)

"""
    ownership_table(ownership) -> Matrix{Tuple{Int,Int}}

Logical coordinate of every `(thread, slot)` pair, zero-based, as a matrix
indexed by `(thread+1, slot+1)`, enumerated from the device `coordinate`
method. This is the oracle for `mapping`, not an input to the derivations.
"""
ownership_table(o) = [map(Int,Layouts._coordinate(o,t,Val(e))) for t in 0:_thread_count(o)-1, e in 0:_register_count(o)-1]
# The same table from the mapping, for derivations. Generators call this;
# it is a pure function of the ownership, with no cache or lock.
function _mapped_table(o)
    m = Int(size(o)[1])
    index(t,e) = Int(mapping(o)((t,e)))
    [(index(t,e) % m, index(t,e) ÷ m) for t in 0:_thread_count(o)-1, e in 0:_register_count(o)-1]
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
    # A slot repeating another slot's value in the same thread is not
    # combined with it, so sums count every logical value once.
    slots = Int(length(Kv))
    groups = Vector{Int}[]
    keys = Int[]
    seen = Set{Int}()
    for e in 0:slots-1
        Int(_slots(o)(e)) in seen && continue
        push!(seen,Int(_slots(o)(e)))
        k = Int(Kv(e))
        g = findfirst(==(k),keys)
        g === nothing ? (push!(groups,[e+1]); push!(keys,k)) : push!(groups[g],e+1)
    end
    # Thread bits whose flip keeps the kept coordinate but changes the value
    # are combined by xor shuffles, within a warp only: bits 0:4 of the
    # thread index. Bits along which the ownership replicates every value
    # hold copies already and are never combined, even across warps.
    replicas = replica_mask(o)
    bits = Int[]
    prefix = 1
    for (s,d) in _pairs(Kt)
        if d == 0 && s > 1
            for b in trailing_zeros(prefix):trailing_zeros(prefix*s)-1
                replicas & (1 << b) == 0 || continue
                b <= 4 || return nothing
                push!(bits,b)
            end
        end
        prefix *= s
    end
    shape = axis == 2 ? (Int(size(o)[1]),1) : (1,Int(size(o)[2]))
    ReductionPlan(groups,bits,Ownership(shape,concat(Kt,Kvn)))
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
    selection_plan(from, to) -> Vector or nothing

For each slot of `to`, the `(coordinate, source slot)` candidates that
supply it: one when the source slot is the same on every thread, several
when it depends on the thread, in which case the source's slots must hold
the same coordinate on every thread so that comparing the target
coordinate with it selects. Singleton axes of `from` broadcast. `nothing`
when some thread lacks a value it needs.
"""
function selection_plan(from, to)
    _thread_count(from) == _thread_count(to) || return nothing
    sf, st = map(Int,size(from)), map(Int,size(to))
    all(map((a,b) -> a == b || a == 1,sf,st)) || return nothing
    tf, tt = _mapped_table(from), _mapped_table(to)
    project(c) = (sf[1] == 1 ? 0 : c[1], sf[2] == 1 ? 0 : c[2])
    plan = Vector{Vector{Tuple{Tuple{Int,Int},Int}}}()
    for e in axes(tt,2)
        choices = Tuple{Tuple{Int,Int},Int}[]
        for t in axes(tt,1)
            c = project(tt[t,e])
            s = findfirst(==(c),view(tf,t,:))
            s === nothing && return nothing
            any(x -> x[2] == s,choices) || push!(choices,(c,s))
        end
        if length(choices) > 1
            # Candidates must name thread-invariant source coordinates.
            for (c,s) in choices
                all(t -> tf[t,s] == c,axes(tf,1)) || return nothing
            end
        end
        push!(plan,choices)
    end
    plan
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
    (; slots=slots .+ 1, ownership=Ownership(shape,concat(mode(reindexed,1),values)))
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
        atom = mapping(_operand_layout(CopyAtom{:load,trans}(),Registers()))
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

# ── Memory offsets of an ownership in a tile ────────────────────────────
# A tile's layout as it varies within the tile, read from the layout type:
# per axis, leaves `(extent, stride)` covering the tile's extent, with a
# stride that is an `Int` or `(:runtime, axis, leaf)` for a runtime value.
# Flat modes, windows of flat layouts, the inner modes of a `tile` and the
# layout under a swizzle qualify.
function _local_leaves(::Type{Layouts.Layout{S,D}},shape) where {S,D}
    axes = Vector{Vector{Tuple{Int,Any}}}()
    for a in 1:2
        s_t, d_t = S.parameters[a], D.parameters[a]
        leaves = Tuple{Int,Any}[]
        if !(s_t <: Tuple)
            push!(leaves,(shape[a], d_t <: Layouts.StaticInt ? d_t.parameters[1] : (:runtime,a,0)))
        else
            volume = 1
            for (j,(st,dt)) in enumerate(_type_leaves(s_t,d_t))
                volume >= shape[a] && break
                st <: Layouts.StaticInt || return nothing
                n = st.parameters[1]
                push!(leaves,(min(n,cld(shape[a],volume)), dt <: Layouts.StaticInt ? dt.parameters[1] : (:runtime,a,j)))
                volume *= n
            end
            volume >= shape[a] || return nothing
        end
        push!(axes,leaves)
    end
    axes
end
# The (shape, stride) leaf types of a mode, first fastest.
_type_leaves(s::Type,d::Type) = s <: Tuple ? reduce(vcat,[_type_leaves(x,y) for (x,y) in zip(s.parameters,d.parameters)];init=Tuple{Any,Any}[]) : Tuple{Any,Any}[(s,d)]
_flat(::Type{Layouts.Layout{S,D}}) where {S,D} = !any(p -> p <: Tuple,S.parameters)
_flat(::Type) = false
_local_leaves(::Type{Layouts.Window{W,L,O}},shape) where {W,L,O} = _flat(L) ? _local_leaves(L,shape) : nothing
_local_leaves(::Type{Layouts.Composition{F,L}},shape) where {F<:Layouts.Swizzle,L} = _local_leaves(L,shape)
_local_leaves(::Type,shape) = nothing
# The offset of a coordinate: a static part and the multiplicity of each runtime stride.
function _local_offset(leaves,c)
    static = 0; runtime = Dict{Any,Int}()
    for a in 1:2
        x = c[a]
        for (j,(n,d)) in enumerate(leaves[a])
            digit = j == length(leaves[a]) ? x : x % n
            x = x ÷ n
            d isa Int ? (static += digit*d) : (runtime[d] = get(runtime,d,0) + digit)
        end
    end
    filter!(p -> p.second != 0,runtime)
    (static,runtime)
end
_add_offsets(a,b) = (a[1]+b[1],mergewith(+,a[2],b[2]))

"""
    replica_mask(ownership) -> Int

The thread-index bits along which the ownership replicates every value
(zero-stride thread modes of power-of-two extent at power-of-two positions).
A store writes each value once, from the thread whose bits in the mask are
zero. Other zero-stride modes are not reported (they store redundantly).
"""
function replica_mask(o)
    mask = 0; prefix = 1
    for (s,d) in _pairs(_threads(o))
        if d == 0 && s > 1 && ispow2(s) && ispow2(prefix)
            mask |= (s-1) << trailing_zeros(prefix)
        end
        prefix *= s
    end
    mask
end

"""
    tile_offsets(ownership, layout_type) -> (; threads, slots) or nothing

The storage offsets of an ownership's values in a tile, split into each
thread's part and each slot's part, when the offset of every value is the
sum of the two: for flat layouts always, for hierarchical ones when threads
and slots occupy different digits. Offsets are `(static, runtime multiples)`.
"""
function tile_offsets(o,::Type{ML}) where ML
    leaves = _local_leaves(ML,map(Int,size(o)))
    leaves === nothing && return nothing
    table = _mapped_table(o)
    slots = [Layouts._slot_coordinate(o,e) for e in 0:_register_count(o)-1]
    so = [_local_offset(leaves,c) for c in slots]
    to = [_local_offset(leaves,table[t,1] .- slots[1]) for t in axes(table,1)]
    for t in axes(table,1), e in axes(table,2)
        _local_offset(leaves,table[t,e]) == _add_offsets(to[t],so[e]) || return nothing
    end
    (; threads = to, slots = so)
end

"""
    vector_plan(ownership, T, layout_type, align) -> (; width, groups) or nothing

Group each thread's values into vectors of `width` bytes (16, 8 or 4, at most
`align`) whose storage offsets in the tile are consecutive and aligned, the
same on every thread. Offsets come from composing the tile's layout with the
ownership, so a vector may gather slots that are far apart in the fragment,
as a K-permuted layout requires. 8- and 16-bit values group by whole
register words. `groups` lists each vector's slots (1-based) in storage
order; the widest grouping wins; `nothing` when no multi-element vector fits.
"""
function vector_plan(o,::Type{T},::Type{ML},align::Int) where {T,ML}
    offsets = tile_offsets(o,ML)
    offsets === nothing && return nothing
    per = _element_bits(T) < 32 ? 32 ÷ _element_bits(T) : 1
    slots = _register_count(o)
    slots % per == 0 || return nothing
    so = offsets.slots
    # Units: whole words for packed values, elements otherwise.
    units = [(per*(u-1)+1):(per*u) for u in 1:slots÷per]
    for r in units
        all(i -> so[r[i]] == (so[r[1]][1]+i-1,so[r[1]][2]),eachindex(r)) || return nothing
    end
    for width in (16,8,4)
        width <= align || continue
        k = width ÷ sizeof(T)
        k >= 2 && k % per == 0 || continue
        n = k ÷ per
        length(units) % n == 0 || continue
        all(t -> t[1] % k == 0,offsets.threads) || continue
        order = sort(units;by = r -> (sort(collect(so[r[1]][2])),so[r[1]][1]))
        groups = Vector{Vector{Int}}()
        valid = true
        for g in 1:n:length(order)
            members = order[g:g+n-1]
            start = so[members[1][1]]
            valid = start[1] % k == 0 &&
                all(i -> so[members[i][1]] == (start[1]+(i-1)*per,start[2]),eachindex(members))
            valid || break
            push!(groups,reduce(vcat,collect.(members)))
        end
        valid || continue
        sort!(groups;by = minimum)
        return (; width, groups)
    end
    nothing
end
