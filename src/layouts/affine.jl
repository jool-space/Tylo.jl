abstract type AbstractLayout end
const IntLike = Union{Integer,StaticInt}
@inline _constant(c::Integer,n) = oftype(c,Int(n))
@inline _constant(c::StaticInt,n) = static(Int(n))

# Staticness belongs to individual leaves: dimensions/strides may mix
# StaticInt constants and runtime integers. Coordinates are zero based.
_volume(n::IntLike) = n
_volume(t::Tuple) = prod(map(_volume,t))
_valid_tree(s::IntLike, d::IntLike) = s > 0 && d >= 0
_valid_tree(s::Tuple, d::Tuple) = !isempty(s) && length(s) == length(d) &&
    all(map(_valid_tree,s,d))
_valid_tree(s, d) = false

"""
    Layout(shape, strides)

A hierarchical coordinate-to-element-offset map. Shape and stride trees must
match, with positive extents and nonnegative strides. Zero strides represent
broadcasting; a layout is not necessarily injective. Use `static(n)` at leaves
that must specialize; ordinary integers remain runtime values.

`l((row, col))` evaluates zero-based coordinates. An integer coordinate is
decomposed first-mode-fastest, recursively. Evaluation assumes in-range
coordinates; memory views provide checked windows.
"""
struct Layout{S<:Tuple,D<:Tuple} <: AbstractLayout
    shape::S
    strides::D
    Base.@propagate_inbounds function Layout(s::Tuple,d::Tuple)
        @boundscheck _valid_tree(s,d) || throw(ArgumentError("invalid shape/stride trees"))
        new{typeof(s),typeof(d)}(s,d)
    end
end
shape(l::Layout) = l.shape
Base.strides(l::Layout) = l.strides
Base.size(l::AbstractLayout) = map(_volume,shape(l))
Base.length(l::AbstractLayout) = _volume(shape(l))
Base.:(==)(a::Layout,b::Layout) = shape(a) == shape(b) && strides(a) == strides(b)

@inline _leaf(c::IntLike,d::StaticInt) = c * _constant(c,d)
@inline _leaf(c::StaticInt,d::StaticInt) = c * d
@inline _leaf(c::IntLike,d::IntLike) = c * d
@inline _eval(s::IntLike,d::IntLike,c::IntLike) = _leaf(c,d)
@inline function _eval(s::Tuple,d::Tuple,c::Tuple)
    length(s) == length(c) || throw(ArgumentError("coordinate rank mismatch"))
    +(map(_eval,s,d,c)...)
end
# The outermost mode takes the remainder by a static extent, which LLVM
# merges with the other modes' masks, but a runtime extent's quotient as it
# is: coordinates are in range, and that remainder would divide.
@inline _split(s::Tuple{Any},d::Tuple{Any},c) = _outermost(first(s),first(d),c,_volume(first(s)))
@inline _outermost(s,d,c,n::StaticInt) = _eval(s,d,c % _constant(c,n))
@inline _outermost(s,d,c,n) = _eval(s,d,c)
@inline function _split(s::Tuple,d::Tuple,c)
    n = _constant(c,_volume(first(s)))
    _eval(first(s),first(d),c % n) + _split(Base.tail(s),Base.tail(d),c ÷ n)
end
# The offset of an index known to be non-negative, such as a tile index:
# a static power of two splits it with a mask and a shift rather than a
# signed division. (General coordinates keep `%` and `÷`, which LLVM
# recombines across nested windows.)
@inline _offset_nonnegative(l::Layout,c::Integer) = _split_nonnegative(l.shape,l.strides,c)
@inline _split_nonnegative(s::Tuple{Any},d::Tuple{Any},c) = _eval(first(s),first(d),c)
@inline function _split_nonnegative(s::Tuple,d::Tuple,c)
    n = _volume(first(s))
    _eval(first(s),first(d),_low(c,n)) + _split_nonnegative(Base.tail(s),Base.tail(d),_high(c,n))
end
@inline _low(c,n) = c % _constant(c,n)
@inline _high(c,n) = c ÷ _constant(c,n)
@inline _low(c::Integer,n::StaticInt) = (m = Int(n); ispow2(m) ? c & oftype(c,m-1) : c % oftype(c,m))
@inline _high(c::Integer,n::StaticInt) = (m = Int(n); ispow2(m) ? c >> trailing_zeros(m) : c ÷ oftype(c,m))
@inline _eval(s::Tuple,d::Tuple,c::IntLike) = _split(s,d,c)
@inline (l::Layout)(c) = _eval(l.shape,l.strides,c)

_span(s::IntLike,d::IntLike) = (s-1)*d
_span(s::Tuple,d::Tuple) = sum(map(_span,s,d))
"Number of storage elements up to the largest offset, including holes."
cosize(l::Layout) = 1 + _span(shape(l),strides(l))

# Explicit factorization preserves the function while exposing tile/outer
# coordinates. A non-divisible dimension is rejected, not silently padded.
"""
    tile(l::Layout, (m, n, ...))

Factor each flat mode into `(within_tile, tile_index)`, first component fastest.
For example a 16×32 matrix tiled by 8×16 has shape `((8,2),(16,2))`.
This is [`logical_divide`](@ref) by static contiguous tiles, one per mode:
it changes coordinates, not storage or ownership.
"""
Base.@constprop :aggressive @inline tile(l::Layout,t::Tuple) = _tile(l,Val(map(Int,t)))
function _tile(l::Layout,::Val{T}) where T
    s = shape(l)
    length(s) == length(T) || throw(ArgumentError("tile rank mismatch"))
    all(x -> x isa IntLike,s) || throw(ArgumentError("tile expects flat modes"))
    all(map((n,t) -> t isa Int && t > 0 && n % t == 0,s,T)) ||
        throw(ArgumentError("tile extents must divide the shape"))
    logical_divide(l,map(static,T))
end
