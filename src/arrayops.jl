# Array spelling over register values and explicit thread/value ownership.
# Instruction operands retain their packing/role-specific representations.
const BroadcastFragment = Union{Fragment,MMAFragment{<:Any,Accumulator},MMAAccumulator}
_register_count(::TiledMMAOwnership{TiledMMA{A,W,R,K}}) where {A,W,R,K} = 4prod(R)

struct PermutedOwnership{L}
    parent::L
end
Base.size(ownership::PermutedOwnership) = reverse(size(ownership.parent))
@inline Layouts.coordinate(ownership::PermutedOwnership, t::Integer, e::Val) =
    reverse(Layouts.coordinate(ownership.parent, t, e))
_register_count(ownership::PermutedOwnership) = _register_count(ownership.parent)
const PermutedFragment{T,N,L} = Fragment{T,N,PermutedOwnership{L}}
Base.parent(f::PermutedFragment) = Fragment(f.data,Layouts.layout(f).parent)

"""
    permutedims(fragment, (2, 1))

Exchange the two logical axes without moving values between registers or
threads. Applying the permutation twice restores the original ownership.
For example, `maximum(permutedims(f); dims=1)` performs the same collective as
`maximum(f; dims=2)`, with the result's logical axes exchanged as well.
This does not convert values into another MMA operand distribution.
"""
@inline function Base.permutedims(fragment::BroadcastFragment, permutation=(2, 1))
    permutation isa Tuple{Integer, Integer} || throw(ArgumentError("expected a two-axis permutation tuple"))
    permutation == (1, 2) && return fragment
    permutation == (2, 1) || throw(ArgumentError("expected a permutation of (1,2)"))
    _swap_axes(Fragment(fragment))
end
@inline _swap_axes(fragment::Fragment) =
    Fragment(fragment.data, PermutedOwnership(Layouts.layout(fragment)))
@inline _swap_axes(fragment::PermutedFragment) = parent(fragment)
@inline _swap_axes(fragment::Fragment{T,N,<:TmemTransfer}) where {T,N} =
    Fragment(fragment.data, permutedims(Layouts.layout(fragment)))
@inline Base.only(fragment::Fragment{T,1,<:PermutedOwnership{<:ReducedFragmentLayout}}) where T =
    only(fragment.data)

Base.eltype(::Type{<:Fragment{T}}) where T = T
Base.eltype(::Type{<:MMAFragment{T,Accumulator}}) where T = T
Base.eltype(::Type{MMAAccumulator{P,N,T}}) where {P,N,T} = T
Base.eltype(f::BroadcastFragment) = eltype(typeof(f))
_local_count(::Type{<:Fragment{T,N}}) where {T,N} = N
_local_count(::Type{MMAFragment{T,Accumulator,N,R}}) where {T,N,R} = N
_local_count(::Type{MMAAccumulator{P,N,T}}) where {P,N,T} = 4N
@inline _register_value(f,::Val{I}) where I = f.data[I]
@inline _register_value(f::MMAAccumulator,::Val{I}) where I = f.data[(I-1)÷4+1].data[(I-1)%4+1]
@generated function _register_values(f::F) where F<:BroadcastFragment
    values = [:(_register_value(f,Val($i))) for i in 1:_local_count(F)]
    quote
        Base.@inline
        ($(values...),)
    end
end
@inline Fragment(f::Fragment) = f
"Expose accumulator values with their ownership, preserving every thread/value slot."
@inline Fragment(f::Union{MMAFragment{<:Any,Accumulator},MMAAccumulator}) =
    Fragment(_register_values(f),Layouts.layout(f))

# Elementwise operations may change element type. An MMA instruction still
# accepts only its prescribed operand/accumulator scalar types.
struct FragmentStyle <: Base.Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:BroadcastFragment}) = FragmentStyle()
Base.BroadcastStyle(::FragmentStyle,::FragmentStyle) = FragmentStyle()
Base.BroadcastStyle(::FragmentStyle,::Base.Broadcast.DefaultArrayStyle{0}) = FragmentStyle()
Base.broadcastable(f::BroadcastFragment) = f
@inline Base.Broadcast.instantiate(bc::Base.Broadcast.Broadcasted{FragmentStyle}) = bc

# Canonicalize only parameters that do not affect the thread/value mapping.
# Compare the values too: two runtime layouts can share a type but differ.
_canonical_ownership(l) = l
_canonical_ownership(l::PermutedOwnership) = PermutedOwnership(_canonical_ownership(l.parent))
_canonical_ownership(::TiledMMAOwnership{TiledMMA{A,W,R,K}}) where {A,W,R,K} =
    TiledMMAOwnership{TiledMMA{MMA16x8x16{BFloat16},W,R,16}}()
_distribution(f::BroadcastFragment) = _canonical_ownership(Layouts.layout(f))
_is_reduced(l) = false
_is_reduced(::ReducedFragmentLayout) = true
_is_reduced(l::PermutedOwnership) = _is_reduced(l.parent)

# Implemented collective recipes. An arbitrary ownership can participate in
# elementwise arithmetic without pretending it has a reduction implementation.
_reduced_layout(l) = throw(ArgumentError("no reduction implementation for this ownership layout"))
_reduced_layout(::Layouts.LaneRows) = ReducedFragmentLayout{LaneRowOwnership}()
_reduced_layout(::WarpRowLayout) = ReducedFragmentLayout{WarpRowOwnership}()
@inline function _reduced_layout(l::Layouts.Ownership)
    isequal(l,operand_layout(MMA16x8x16(BFloat16),Accumulator())) ||
        throw(ArgumentError("no reduction implementation for this ownership layout"))
    ReducedFragmentLayout{MMARowOwnership{1,1}}()
end
@inline function _reduced_layout(::TiledMMAOwnership{P}) where P
    ReducedFragmentLayout{typeof(row_ownership(_plan(P)))}()
end
_reduced_layout(l::PermutedOwnership) = PermutedOwnership(_reduced_layout(l.parent))

@inline _broadcast_anchor(x) = nothing
@inline _broadcast_anchor(x::BroadcastFragment) = x
@inline _broadcast_anchor(x::Base.Broadcast.Broadcasted) = _broadcast_anchor(x.args)
@inline _broadcast_anchor(::Tuple{}) = nothing
@inline _broadcast_anchor(xs::Tuple) =
    _choose_anchor(_broadcast_anchor(first(xs)),_broadcast_anchor(Base.tail(xs)))
@inline _choose_anchor(::Nothing,b) = b
@inline _choose_anchor(a,::Nothing) = a
@inline _choose_anchor(::Nothing,::Nothing) = nothing
@inline _choose_anchor(a::BroadcastFragment,b::BroadcastFragment) =
    _is_reduced(Layouts.layout(a)) ? b : a

@inline function _check_broadcast(fragment::BroadcastFragment, anchor::BroadcastFragment)
    expected = if _is_reduced(Layouts.layout(fragment)) && !_is_reduced(Layouts.layout(anchor))
        _reduced_layout(Layouts.layout(anchor))
    else
        _distribution(anchor)
    end
    isequal(_distribution(fragment), expected) ||
        throw(DimensionMismatch("fragment axes or ownership differ; redistribute explicitly"))
    nothing
end
@inline _check_broadcast(::Number,::BroadcastFragment) = nothing
@inline _check_broadcast(::Ref,::BroadcastFragment) = nothing
@inline _check_broadcast(::Tuple{},anchor) = nothing
@inline function _check_broadcast(xs::Tuple,anchor)
    _check_broadcast(first(xs),anchor)
    _check_broadcast(Base.tail(xs),anchor)
end
@inline _check_broadcast(bc::Base.Broadcast.Broadcasted,anchor) = _check_broadcast(bc.args,anchor)
@inline _check_broadcast(x,anchor) = throw(ArgumentError("fragment broadcasting accepts fragments and scalars"))

@inline _reduction_slot(::Union{Layouts.LaneRows,WarpRowLayout},::Val) = Val(1)
@inline _reduction_slot(::Layouts.Ownership,::Val{I}) where I = Val((I-1)÷2+1)
@inline _reduction_slot(l::PermutedOwnership,i::Val) = _reduction_slot(l.parent,i)
@inline function _reduction_slot(::TiledMMAOwnership{TiledMMA{A,W,R,K}},::Val{I}) where {A,W,R,K,I}
    Val(2*(((I-1)÷4)%R[1])+(I-1)%4÷2+1)
end
@inline _broadcast_value(x::Number,anchor,i) = x
@inline _broadcast_value(x::Ref,anchor,i) = x[]
@inline _broadcast_value(::Base.RefValue{Type{T}},anchor,i) where T = T
@inline function _broadcast_value(x::BroadcastFragment,anchor,i)
    slot = _is_reduced(Layouts.layout(x)) && !_is_reduced(Layouts.layout(anchor)) ?
        _reduction_slot(Layouts.layout(anchor),i) : i
    _register_value(x,slot)
end
@inline function _broadcast_value(bc::Base.Broadcast.Broadcasted,anchor,i)
    bc.f(_broadcast_arguments(bc.args,anchor,i)...)
end
@inline _broadcast_arguments(::Tuple{},anchor,i) = ()
@inline _broadcast_arguments(xs::Tuple,anchor,i) =
    (_broadcast_value(first(xs),anchor,i),_broadcast_arguments(Base.tail(xs),anchor,i)...)

@inline _rebuild_fragment(f::Fragment,data) = Fragment(data,Layouts.layout(f))
@inline _rebuild_fragment(::MMAFragment{T,Accumulator},data) where T = MMAFragment(eltype(data),Accumulator(),data)
@generated function _rebuild_fragment(::MMAAccumulator{P,N},data::NTuple{M,T}) where {P,N,M,T}
    M == 4N || error("accumulator register count differs")
    values = [:(MMAFragment($T,Accumulator(),($( [:(data[$(4j+i)]) for i in 1:4]... ),))) for j in 0:N-1]
    quote
        Base.@inline
        MMAAccumulator(_plan($P),($(values...),))
    end
end
@generated function _materialize_fragment(anchor::F,bc) where F<:BroadcastFragment
    values = [:(_broadcast_value(bc,anchor,Val($i))) for i in 1:_local_count(F)]
    quote
        Base.@inline
        _rebuild_fragment(anchor,($(values...),))
    end
end
@inline function Base.copy(bc::Base.Broadcast.Broadcasted{FragmentStyle})
    anchor = _broadcast_anchor(bc)
    _check_broadcast(bc,anchor)
    _materialize_fragment(anchor,bc)
end

# map combines corresponding elements; broadcasting also expands reduced axes.
@inline function Base.map(operation, first_fragment::BroadcastFragment,
                         second_fragment::BroadcastFragment, rest::BroadcastFragment...)
    _check_map(first_fragment, (second_fragment, rest...))
    copy(Base.Broadcast.broadcasted(operation, first_fragment, second_fragment, rest...))
end
@inline _check_map(first_fragment, ::Tuple{}) = nothing
@inline function _check_map(first_fragment, remaining::Tuple)
    isequal(_distribution(first_fragment), _distribution(first(remaining))) ||
        throw(DimensionMismatch("map requires matching ownership; use broadcasting for reduction results"))
    _check_map(first_fragment, Base.tail(remaining))
end

@inline _reduce_values(op,data,::Layouts.LaneRows) = (_local_reduce(op,data),)
@inline _reduce_values(op,data,l::PermutedOwnership) = _reduce_values(op,data,l.parent)
@inline function _fragment_reduce(operation, fragment::BroadcastFragment, dimensions)
    layout = Layouts.layout(fragment)
    reduction_axis = _reduction_axis(layout)
    ((dimensions isa Integer && dimensions == reduction_axis) ||
     (dimensions isa Tuple{Integer} && only(dimensions) == reduction_axis)) ||
        throw(ArgumentError("unsupported reduction axis for this ownership layout"))
    eltype(fragment) === Float32 ||
        throw(ArgumentError("fragment reductions currently require Float32 values"))
    result_layout = _reduced_layout(layout)
    Fragment(_reduce_values(operation, _register_values(fragment), layout), result_layout)
end
# Compatibility with the original explicitly row-oriented spelling.
@inline _row_reduce(op,f::BroadcastFragment) = _fragment_reduce(op,f,2)

"""
    sum(fragment; dims)
    maximum(fragment; dims)
    minimum(fragment; dims)

Reduce FP32 values over a logical axis, retaining a singleton dimension and
explicit result ownership. Broadcast the result back with ordinary dotted
arithmetic, for example `fragment .- maximum(fragment; dims=2)`.

Lane-local, warp-striped and MMA ownerships currently implement `dims=2`, or
`dims=1` after `permutedims`. A one-element tuple is also accepted. The axis
must be known to inference in a GPU kernel. Other ownerships, axes and full
reductions are rejected rather than silently reducing only this thread's
values. Warp-striped and MMA reductions require all 32 lanes; multiple N
warps require explicit communication.
"""
@inline Base.sum(f::BroadcastFragment;dims=:) = _fragment_reduce(+,f,dims)
@inline Base.maximum(f::BroadcastFragment;dims=:) = _fragment_reduce(max,f,dims)
@inline Base.minimum(f::BroadcastFragment;dims=:) = _fragment_reduce(min,f,dims)

# TMEM transfer ownership is a local register distribution in either orientation.
_canonical_ownership(p::TmemTransfer{S,A}) where {S,A} =
    A == 2 ? Layouts.LaneRows{S[A]}() : PermutedOwnership(Layouts.LaneRows{S[A]}())
_reduced_layout(p::TmemTransfer) = _reduced_layout(_canonical_ownership(p))
@inline _reduce_values(op,data,::TmemTransfer) = (_local_reduce(op,data),)
@inline _reduction_slot(::TmemTransfer,::Val) = Val(1)
_reduction_axis(l) = 2
_reduction_axis(l::PermutedOwnership) = 3-_reduction_axis(l.parent)
_reduction_axis(::TmemTransfer{S,A}) where {S,A} = A

_register_window_axis(::Type{Layouts.LaneRows{N}}) where N = 2
_register_window_axis(::Type{TmemTransfer{S,A}}) where {S,A} = A
_register_window_axis(::Type{PermutedOwnership{L}}) where L = 3-_register_window_axis(L)
_register_window_axis(::Type) = throw(ArgumentError("no register window implementation for this ownership"))
_register_window_shape(::Type{Layouts.LaneRows{N}}) where N = (32,N)
_register_window_shape(::Type{TmemTransfer{S,A}}) where {S,A} = S
_register_window_shape(::Type{PermutedOwnership{L}}) where L = reverse(_register_window_shape(L))
_window_ownership(::Type{Layouts.LaneRows{N}},s) where N = Layouts.LaneRows{s[2]}()
_window_ownership(::Type{TmemTransfer{S,A}},s) where {S,A} = TmemTransfer{s,A}()
_window_ownership(::Type{PermutedOwnership{L}},s) where L =
    PermutedOwnership(_window_ownership(L,reverse(s)))
function _check_register_window(::Type{L},origin,shape;packed=false) where L
    axis = _register_window_axis(L)
    full = _register_window_shape(L)
    origin isa Tuple{Int,Int} && shape isa Tuple{Int,Int} &&
        all(map((o,n,s) -> 0 <= o && 0 < s && o+s <= n,origin,full,shape)) &&
        origin[3-axis] == 0 && shape[3-axis] == full[3-axis] ||
        throw(ArgumentError("register window must retain the participating threads"))
    packed && !(iseven(origin[axis]) && iseven(shape[axis])) &&
        throw(ArgumentError("packed windows require complete BF16 pairs"))
    axis
end

"""
    window(fragment, Val(origin), Val(shape))

Select a static logical register window, preserving all participating threads.
Currently supports lane-local and TMEM-transfer ownership in either axis
orientation. Packed BF16 windows must retain complete word pairs. Runtime
register indexing and implicit redistribution are not provided.
"""
@generated function window(f::Fragment{T,N,L},::Val{O},::Val{S}) where {T,N,L,O,S}
    axis = _check_register_window(L,O,S)
    values = [:(f.data[$i]) for i in O[axis]+1:O[axis]+S[axis]]
    quote
        Base.@inline
        Fragment(($(values...),),_window_ownership($L,$S))
    end
end
@generated function window(f::PackedBF16{N,L},::Val{O},::Val{S}) where {N,L,O,S}
    axis = _check_register_window(L,O,S;packed=true)
    values = [:(f.data[$i]) for i in O[axis]÷2+1:(O[axis]+S[axis])÷2]
    quote
        Base.@inline
        PackedBF16(($(values...),),_window_ownership($L,$S))
    end
end
_swap_packed_ownership(l) = PermutedOwnership(l)
_swap_packed_ownership(l::PermutedOwnership) = l.parent
_swap_packed_ownership(l::TmemTransfer) = permutedims(l)
function Base.permutedims(f::PackedBF16,perm=(2,1))
    Layouts._check_permutation(perm)
    perm == (1,2) ? f : PackedBF16(f.data,_swap_packed_ownership(Layouts.layout(f)))
end
