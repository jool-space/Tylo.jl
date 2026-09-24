# Array spelling over register values and explicit thread/value ownership.
# Instruction operands retain their packing/role-specific representations.
# Distribution questions are answered from ownership layouts while
# generating code, so every ownership with a static description participates.
"""
    permutedims(fragment, (2, 1))

Exchange the two logical axes without moving values between registers or
threads: the ownership becomes its transpose. Applying it twice restores the
original distribution. `maximum(permutedims(f); dims=1)` performs the same
collective as `maximum(f; dims=2)`. This does not convert values into
another MMA operand distribution.
"""
@generated function Base.permutedims(f::Fragment{T,N,L},permutation=(2,1)) where {T,N,L}
    o = _static_instance(L)
    o === nothing && return :(throw(ArgumentError("axis permutation requires a static ownership")))
    transposed = permutedims(o)
    quote
        Base.@inline
        Layouts._check_permutation(permutation)
        permutation == (1,2) ? f : _with_ownership(f,$transposed)
    end
end

_local_count(::Type{<:Fragment{T,N}}) where {T,N} = N
@inline _register_value(f,::Val{I}) where I = f.data[I]
@inline _register_values(f::Fragment) = f.data
@inline Fragment(f::Fragment) = f

# Elementwise operations may change element type. An MMA instruction still
# accepts only its prescribed operand/accumulator scalar types.
struct FragmentStyle <: Base.Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:Fragment}) = FragmentStyle()
Base.BroadcastStyle(::FragmentStyle,::FragmentStyle) = FragmentStyle()
Base.BroadcastStyle(::FragmentStyle,::Base.Broadcast.DefaultArrayStyle{0}) = FragmentStyle()
Base.broadcastable(f::Fragment) = f
@inline Base.Broadcast.instantiate(bc::Base.Broadcast.Broadcasted{FragmentStyle}) = bc

# Distribution queries resolve while generating. Two ownerships are the same
# distribution when their coalesced mappings agree; a broadcast source with a
# singleton axis supplies each anchor slot through a thread-uniform slot map.
# Ownerships with runtime leaves have no static mapping while generating; two
# such values of one type compare at run time and never broadcast across axes.
@generated function _same_distribution(a::A,b::B) where {A,B}
    x, y = _static_instance(A), _static_instance(B)
    (x === nothing || y === nothing) && return A === B ? :(isequal(a,b)) : :(false)
    :($(same_distribution(x,y)))
end
@generated function _broadcast_slots(x::X,a::A) where {X,A}
    ox, oa = _static_instance(X), _static_instance(A)
    if ox === nothing || oa === nothing
        X === A || return :(nothing)
        # An empty tuple marks the identity map; slots resolve in _broadcast_slot.
        return :(isequal(x,a) ? () : nothing)
    end
    slots = broadcast_slots(ox,oa)
    slots === nothing ? :(nothing) : :($(Tuple(slots)))
end
@generated function _broadcast_slot(::X,::A,::Val{I}) where {X,A,I}
    ox, oa = _static_instance(X), _static_instance(A)
    # Runtime-valued ownerships were already checked for equality by
    # _check_broadcast; their slot map is the identity.
    (ox === nothing || oa === nothing) && return :(Val($I))
    slots = broadcast_slots(ox,oa)
    slots === nothing ? :(nothing) : :(Val($(slots[I])))
end

# The anchor is the fragment with the complete logical shape; every other
# fragment either shares its distribution or broadcasts along singleton axes.
@inline _broadcast_anchor(x) = nothing
@inline _broadcast_anchor(x::Fragment) = x
@inline _broadcast_anchor(x::Base.Broadcast.Broadcasted) = _broadcast_anchor(x.args)
@inline _broadcast_anchor(::Tuple{}) = nothing
@inline _broadcast_anchor(xs::Tuple) =
    _choose_anchor(_broadcast_anchor(first(xs)),_broadcast_anchor(Base.tail(xs)))
@inline _choose_anchor(::Nothing,b) = b
@inline _choose_anchor(a,::Nothing) = a
@inline _choose_anchor(::Nothing,::Nothing) = nothing
@inline _choose_anchor(a::Fragment,b::Fragment) =
    _covers(size(Layouts.layout(b)),size(Layouts.layout(a))) ? b : a
@inline _covers(larger,smaller) = larger != smaller && all(map(>=,larger,smaller))

@inline function _check_broadcast(fragment::Fragment, anchor::Fragment)
    _broadcast_slots(Layouts.layout(fragment),Layouts.layout(anchor)) === nothing &&
        throw(DimensionMismatch("fragment axes or ownership differ; redistribute explicitly"))
    nothing
end
@inline _check_broadcast(::Number,::Fragment) = nothing
@inline _check_broadcast(::Ref,::Fragment) = nothing
@inline _check_broadcast(::Tuple{},anchor) = nothing
@inline function _check_broadcast(xs::Tuple,anchor)
    _check_broadcast(first(xs),anchor)
    _check_broadcast(Base.tail(xs),anchor)
end
@inline _check_broadcast(bc::Base.Broadcast.Broadcasted,anchor) = _check_broadcast(bc.args,anchor)
@inline _check_broadcast(x,anchor) = throw(ArgumentError("fragment broadcasting accepts fragments and scalars"))

@inline _broadcast_value(x::Number,anchor,i) = x
@inline _broadcast_value(x::Ref,anchor,i) = x[]
@inline _broadcast_value(::Base.RefValue{Type{T}},anchor,i) where T = T
@inline _broadcast_value(x::Fragment,anchor,i::Val) =
    _register_value(x,_broadcast_slot(Layouts.layout(x),Layouts.layout(anchor),i))
@inline function _broadcast_value(bc::Base.Broadcast.Broadcasted,anchor,i)
    bc.f(_broadcast_arguments(bc.args,anchor,i)...)
end
@inline _broadcast_arguments(::Tuple{},anchor,i) = ()
@inline _broadcast_arguments(xs::Tuple,anchor,i) =
    (_broadcast_value(first(xs),anchor,i),_broadcast_arguments(Base.tail(xs),anchor,i)...)

@inline _rebuild_fragment(f::Fragment,data) = Fragment(data,Layouts.layout(f))
@inline function _materialize_fragment(anchor::F,bc) where F<:Fragment
    values = @rtuple(i -> _broadcast_value(bc,anchor,Val(i)), 1:_local_count(F))
    _rebuild_fragment(anchor,values)
end
@inline function Base.copy(bc::Base.Broadcast.Broadcasted{FragmentStyle})
    anchor = _broadcast_anchor(bc)
    _check_broadcast(bc,anchor)
    _materialize_fragment(anchor,bc)
end

# map combines corresponding elements; broadcasting also expands reduced axes.
@inline function Base.map(operation, first_fragment::Fragment,
                         second_fragment::Fragment, rest::Fragment...)
    _check_map(first_fragment, (second_fragment, rest...))
    copy(Base.Broadcast.broadcasted(operation, first_fragment, second_fragment, rest...))
end
@inline _check_map(first_fragment, ::Tuple{}) = nothing
@inline function _check_map(first_fragment, remaining::Tuple)
    _same_distribution(Layouts.layout(first_fragment), Layouts.layout(first(remaining))) ||
        throw(DimensionMismatch("map requires matching ownership; use broadcasting for reduction results"))
    _check_map(first_fragment, Base.tail(remaining))
end

Base.@constprop :aggressive @inline _reduction_axis_argument(d::Integer) =
    d == 1 || d == 2 ? Int(d) : throw(ArgumentError("unsupported reduction axis for a fragment"))
Base.@constprop :aggressive @inline _reduction_axis_argument(d::Tuple{Integer}) = _reduction_axis_argument(only(d))
@inline _reduction_axis_argument(d) = throw(ArgumentError("unsupported reduction axis for a fragment"))
Base.@constprop :aggressive @inline function _fragment_reduce(operation, fragment::Fragment, dimensions)
    axis = _reduction_axis_argument(dimensions)
    eltype(fragment) === Float32 ||
        throw(ArgumentError("fragment reductions currently require Float32 values"))
    layout = Layouts.layout(fragment)
    Fragment(_reduce_values(operation, _register_values(fragment), layout, Val(axis)),
             _reduced_ownership(layout, Val(axis)))
end

"""
    sum(fragment; dims)
    maximum(fragment; dims)
    minimum(fragment; dims)

Reduce FP32 values over a logical axis, retaining a singleton dimension and
explicit result ownership. Broadcast the result back with ordinary dotted
arithmetic, for example `fragment .- maximum(fragment; dims=2)`.

The recipe is derived from the ownership: a local tree over the slots that
share a kept coordinate, then xor shuffles over the lane bits that replicate
it. An axis whose values span warps has no recipe and is rejected rather than
silently reducing only this thread's values. Reductions with shuffles require
all 32 lanes of each warp. The result type depends on the axis, so `dims`
must be a constant in a kernel.
"""
@inline Base.sum(f::Fragment;dims=:) = _fragment_reduce(+,f,dims)
@inline Base.maximum(f::Fragment;dims=:) = _fragment_reduce(max,f,dims)
@inline Base.minimum(f::Fragment;dims=:) = _fragment_reduce(min,f,dims)

"""
    window(fragment, origin, shape)

Select a static logical register window, preserving all participating
threads: the values whose coordinates fall inside the `shape` box at
`origin`, with an ownership relative to that origin. The slots inside the
window must be the same on every thread. Both tuples must be constants.
Runtime register indexing and implicit redistribution are not provided.
"""
Base.@constprop :aggressive @inline window(f::Fragment,origin::Tuple{Integer,Integer},shape::Tuple{Integer,Integer}) =
    _window(f,Val(map(Int,origin)),Val(map(Int,shape)))
@generated function _window(f::Fragment{T,N,L},::Val{O},::Val{S}) where {T,N,L,O,S}
    o = _static_instance(L)
    o === nothing && return :(throw(ArgumentError("register windows require a static ownership")))
    plan = window_plan(o,O,S)
    plan === nothing && return :(throw(ArgumentError("register window must retain the participating threads")))
    slots = plan.slots
    # Whole register words stay words.
    if _packed(T,N) && _packed(T,length(slots))
        per = _per_word(T)
        starts = slots[1:per:end]
        if all(i -> slots[i] == slots[per*((i-1)÷per)+1] + (i-1)%per && (slots[per*((i-1)÷per)+1]-1) % per == 0,eachindex(slots))
            words = [:(getfield(f,:storage)[$((w-1)÷per+1)]) for w in starts]
            return quote
                Base.@inline
                _from_words($T,($(words...),),$(plan.ownership))
            end
        end
    end
    quote
        Base.@inline
        Fragment(($([:(f.data[$e]) for e in slots]...),),$(plan.ownership))
    end
end

"""
    relayout(target, fragment)

Reinterpret this thread's values under `target` ownership. Valid when every
thread holds the same element set in both ownerships, so the change is a
register permutation without communication. Other conversions are rejected.
"""
@generated function relayout(::TO,f::Fragment{T,N,L}) where {TO,T,N,L}
    from, to = _static_instance(L), _static_instance(TO)
    (from === nothing || to === nothing) && return :(throw(ArgumentError("relayout requires static ownerships")))
    permutation = relayout_permutation(from,to)
    permutation === nothing &&
        return :(throw(ArgumentError("no in-lane relayout between these ownerships; communication is required")))
    quote
        Base.@inline
        Fragment(($([:(f.data[$i]) for i in permutation]...),),$to)
    end
end
