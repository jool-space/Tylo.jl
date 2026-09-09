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
@inline _split(::Tuple{},::Tuple{},c) = zero(c)
@inline function _split(s::Tuple,d::Tuple,c)
    n = _constant(c,_volume(first(s)))
    _eval(first(s),first(d),c % n) + _split(Base.tail(s),Base.tail(d),c ÷ n)
end
@inline _eval(s::Tuple,d::Tuple,c::IntLike) = _split(s,d,c)
@inline (l::Layout)(c) = _eval(l.shape,l.strides,c)

_span(s::IntLike,d::IntLike) = (s-1)*d
_span(s::Tuple,d::Tuple) = sum(map(_span,s,d))
"Number of storage elements up to the largest offset, including holes."
cosize(l::Layout) = 1 + _span(shape(l),strides(l))

# Explicit factorization preserves the function while exposing tile/outer
# coordinates. A non-divisible dimension is rejected, not silently padded.
"""
    tile(l::Layout, Val((m, n, ...)))

Factor each flat mode into `(within_tile, tile_index)`, first component fastest.
For example a 16×32 matrix tiled by 8×16 has shape `((8,2),(16,2))`.
This algebraic operation changes coordinates, not storage or ownership.
"""
function tile(l::Layout,::Val{T}) where T
    s,d = shape(l),strides(l)
    length(s) == length(T) || throw(ArgumentError("tile rank mismatch"))
    all(x -> x isa IntLike,s) || throw(ArgumentError("tile expects flat modes"))
    all(map((n,t) -> t isa Int && t > 0 && n % t == 0,s,T)) ||
        throw(ArgumentError("tile extents must divide the shape"))
    Layout(map((n,t) -> (static(t),n ÷ static(t)),s,T),
           map((stride,t) -> (stride,stride*static(t)),d,T))
end

"Drop hierarchy and merge adjacent modes when their strides are contiguous."
function coalesce(l::Layout)
    pairs = _pairs(shape(l),strides(l))
    dims = Tuple{Int,Int}[]
    # This is host-side algebra. Preserve the dynamic representation unless
    # every leaf was static; never turn a runtime stride into specialization.
    fixed = all(p -> p[1] isa StaticInt && p[2] isa StaticInt,pairs)
    for (s,d) in pairs
        s == 1 && continue
        if !isempty(dims) && d == prod(dims[end])
            n,stride = pop!(dims)
            push!(dims,(n*Int(s),stride))
        else
            push!(dims,(Int(s),Int(d)))
        end
    end
    isempty(dims) && push!(dims,(1,0))
    cast = fixed ? static : identity
    Layout(Tuple(cast(s) for (s,_) in dims),Tuple(cast(d) for (_,d) in dims))
end
_pairs(s::IntLike,d::IntLike) = ((s,d),)
_pairs(s::Tuple,d::Tuple) = Tuple(p for ps in map(_pairs,s,d) for p in ps)
