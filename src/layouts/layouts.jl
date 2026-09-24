# Pure coordinate machinery. Shapes and coordinates use logical elements,
# never byte addresses or packed TMEM columns.
module Layouts

using Static: StaticInt, static

export Layout, @Layout, Ownership, distribute, Swizzle, compose, window, shape, cosize, static, coordinate
export concat, coalesce, complement, right_inverse, left_inverse,
       logical_divide, zipped_divide, tiled_divide,
       logical_product, blocked_product, raked_product,
       agree, is_injective, is_compact

include("affine.jl")
include("notation.jl")
include("transforms.jl")
include("algebra.jl")

function layout end

# ── Display ─────────────────────────────────────────────────────────────
# Layouts print as the `@Layout` call that builds them: static leaves as
# numbers, runtime leaves interpolated with `$`.
_show_tree(io::IO,x::StaticInt) = print(io,Int(x))
_show_tree(io::IO,x::Integer) = print(io,'$',x)
function _show_tree(io::IO,t::Tuple)
    print(io,'(')
    for (i,x) in enumerate(t)
        i > 1 && print(io,", ")
        _show_tree(io,x)
    end
    length(t) == 1 && print(io,',')
    print(io,')')
end
function Base.show(io::IO,l::Layout)
    print(io,"@Layout ")
    _show_tree(io,shape(l)); print(io,' ')
    _show_tree(io,strides(l))
end
Base.show(io::IO,::Swizzle{B,M,S}) where {B,M,S} = print(io,"Swizzle{",B,", ",M,", ",S,"}()")
Base.show(io::IO,c::Composition) = (print(io,"compose("); show(io,c.outer); print(io,", "); show(io,c.inner); print(io,')'))
function Base.show(io::IO,w::Window{S}) where S
    print(io,"window("); show(io,w.parent); print(io,", ")
    _show_tree(io,w.origin); print(io,", ",S,')')
end
function Base.show(io::IO,o::Ownership{S}) where S
    print(io,"Ownership(",S,", ")
    show(io,o.mapping)
    print(io,')')
end

# ── Ownership coordinates ───────────────────────────────────────────────
# A static ownership's coordinate is evaluated through its projections onto
# the two logical axes, worked out while generating: each axis is a sum of
# `((t ÷ prefix) % extent) * stride` terms over the thread modes plus the
# slot's constant, with shifts and masks for powers of two since thread
# indices are non-negative. Zero-stride modes contribute nothing, and the
# outermost thread mode takes no remainder, since a coordinate is only
# defined for the ownership's own threads.
function _axis_expr(axis::Layout,e::Int)
    pairs = [(Int(s),Int(d)) for (s,d) in zip(shape(flatten(mode(axis,1))),strides(flatten(mode(axis,1))))]
    constant = Int(mode(axis,2)(e))
    terms = Any[]
    prefix = 1
    for (i,(s,d)) in enumerate(pairs)
        if s > 1 && d != 0
            # Thread indices are non-negative: powers of two become shifts and masks.
            x = prefix == 1 ? :t : ispow2(prefix) ? :(t >> $(trailing_zeros(prefix))) : :(t ÷ oftype(t,$prefix))
            if i != length(pairs)
                x = ispow2(s) ? :($x & oftype(t,$(s-1))) : :($x % oftype(t,$s))
            end
            push!(terms,d == 1 ? x : :($x * oftype(t,$d)))
        end
        prefix *= s
    end
    constant == 0 || push!(terms,:(oftype(t,$constant)))
    isempty(terms) ? :(zero(t)) : length(terms) == 1 ? only(terms) : Expr(:call,:+,terms...)
end
"""
    coordinate(ownership, thread, slot)

The zero-based logical coordinate `(row, col)` that `thread` holds in its
zero-based `slot`. In a kernel `slot` must be a constant; the evaluation is
then a few integer operations on `thread`, derived while generating.
"""
Base.@constprop :aggressive @inline coordinate(o::Ownership,t::Integer,e::Integer) = _coordinate(o,t,Val(Int(e)))
@generated function _coordinate(o::Ownership{S,L},t::Integer,::Val{E}) where {S,L,E}
    Base.issingletontype(L) || return :(_indexed_coordinate(o,t,Val(E)))
    l = L.instance
    E isa Int && 0 <= E < Int(size(l)[2]) || return :(throw(BoundsError()))
    m, n = S
    rows = compose(Layout((static(m),static(n)),(static(1),static(0))),l)
    cols = compose(Layout((static(m),static(n)),(static(0),static(1))),l)
    quote
        Base.@inline
        ($(_axis_expr(rows,E)),$(_axis_expr(cols,E)))
    end
end
# The same coordinate in two parts: the thread's part, evaluated at run
# time, and the slot's constant part. `coordinate == thread part + slot part`
# on each axis, since the mapping is a sum over its thread and slot modes.
@generated function _thread_coordinate(o::Ownership{S,L},t::Integer) where {S,L}
    l = L.instance
    m, n = S
    rows = compose(Layout((static(m),static(n)),(static(1),static(0))),l)
    cols = compose(Layout((static(m),static(n)),(static(0),static(1))),l)
    strip(axis) = Layout((shape(axis)[1],static(1)),(strides(axis)[1],static(0)))
    quote
        Base.@inline
        ($(_axis_expr(strip(rows),0)),$(_axis_expr(strip(cols),0)))
    end
end
function _slot_coordinate(o::Ownership{S},e::Int) where S
    m, n = S
    rows = compose(Layout((static(m),static(n)),(static(1),static(0))),o.mapping)
    cols = compose(Layout((static(m),static(n)),(static(0),static(1))),o.mapping)
    (Int(mode(rows,2)(e)),Int(mode(cols,2)(e)))
end
@inline function _indexed_coordinate(o::Ownership{S},t::Integer,::Val{E}) where {S,E}
    E isa Int && 0 <= E < size(o.mapping)[2] || throw(BoundsError())
    i = o.mapping((t,oftype(t,E)))
    m = oftype(i,S[1])
    (i % m,i ÷ m)
end

"""
    permutedims(ownership)

The same threads and slots holding the transposed tile: coordinate `(r, c)`
becomes `(c, r)`.
"""
function Base.permutedims(o::Ownership{S},perm=(2,1)) where S
    _check_permutation(perm)
    perm == (1,2) && return o
    m, n = S
    Ownership((n,m),compose(Layout((static(m),static(n)),(static(n),static(1))),o.mapping))
end

# An ownership's canonical mapping: thread and slot modes each coalesced, a
# mode of one leaf written as that leaf.
@inline _canonical_mapping(l) = l
@inline function _canonical_mapping(l::Layout)
    t, v = coalesce(mode(l,1)), coalesce(mode(l,2))
    Layout((_unwrap(shape(t)),_unwrap(shape(v))),(_unwrap(strides(t)),_unwrap(strides(v))))
end
@inline _unwrap(x::Tuple{Any}) = x[1]
@inline _unwrap(x) = x

"""
    repeat(ownership, rm, rn)

The ownership of an `rm × rn` grid of copies of `ownership`'s tile in which
every thread holds its values of every copy: slots of the original first,
then the copies, M fastest. It describes, for example, two consecutive K
blocks of an MMA operand loaded at once. `rm` and `rn` must be constants.
"""
Base.@constprop :aggressive @inline Base.repeat(o::Ownership,rm::Integer,rn::Integer) = _repeat(o,Val((Int(rm),Int(rn))))
@generated _repeat(o::O,::Val{R}) where {O<:Ownership,R} = :($(_repeat_ownership(O.instance,R)))
function _repeat_ownership(o::Ownership{S},(rm,rn)) where S
    m, n = S
    M = m*rm
    embedded = compose(Layout((static(m),static(n)),(static(1),static(M))),o.mapping)
    values = concat(mode(embedded,2),Layout((static(rm),static(rn)),(static(m),static(n*M))))
    Ownership((M,n*rn),concat(mode(embedded,1),values))
end

end
