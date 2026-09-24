"""
    Swizzle{Bits,Base,Shift}()

XOR two disjoint bit fields in an element offset. `Base` low bits remain
unchanged; positive `Shift` moves the source field right. It is its own
inverse. Units come from the inner layout, not implicitly from bytes.
"""
struct Swizzle{B,M,S}
    function Swizzle{B,M,S}() where {B,M,S}
        B isa Int && M isa Int && S isa Int && B >= 0 && M >= 0 &&
            abs(S) >= B && B+M+abs(S) <= 31 ||
            throw(ArgumentError("swizzle needs disjoint fields within 31 bits"))
        new{B,M,S}()
    end
end
@inline function (::Swizzle{B,M,S})(x::IntLike) where {B,M,S}
    mask = ((1 << B)-1) << (M+max(0,S))
    bits = x & _constant(x,mask)
    x ⊻ (S >= 0 ? bits >> S : bits << -S)
end

struct Composition{F,L<:AbstractLayout} <: AbstractLayout
    outer::F
    inner::L
end
"Function composition: `compose(f,l)(c) == f(l(c))`; retains the inner domain."
compose(f,l::AbstractLayout) = Composition(f,l)
shape(l::Composition) = shape(l.inner)
@inline (l::Composition)(c) = l.outer(l.inner(c))

# Keeping the origin INSIDE the parent mapping matters for swizzles:
# swizzle(parent(origin + c)) generally differs from advancing the pointer
# by swizzle(parent(origin)) and then applying a fresh local swizzle.
struct Window{S,L<:AbstractLayout,O<:Tuple} <: AbstractLayout
    parent::L
    origin::O
end
shape(::Window{S}) where S = map(static,S)
@inline (l::Window)(c::Tuple) = l.parent(map(+,l.origin,c))
@inline (l::Window)(c::IntLike) = l(_coordinates(size(l),c))
@inline _coordinates(::Tuple{},c) = ()
@inline function _coordinates(s::Tuple,c)
    n = _constant(c,first(s))
    (c % n,_coordinates(Base.tail(s),c ÷ n)...)
end

"""
    window(layout, origin, shape)

A static-size logical window with a potentially runtime origin. `shape` must
be a constant tuple. Offsets stay relative to the parent's allocation base,
preserving swizzle phase. Bounds are checked unless the caller uses
`@inbounds` after establishing validity.
"""
Base.@constprop :aggressive Base.@propagate_inbounds window(l::AbstractLayout,o::Tuple,s::Tuple) =
    _window(l,o,Val(map(Int,s)))
Base.@propagate_inbounds function _window(l::AbstractLayout,o::Tuple,::Val{S}) where S
    length(S) == length(o) == length(size(l)) || throw(ArgumentError("window rank mismatch"))
    all(n -> n isa Int && n > 0,S) || throw(ArgumentError("positive static extents required"))
    @boundscheck all(map((n,x,w) -> 0 <= x && x <= n-w,size(l),o,S)) ||
        throw(BoundsError(l,(o,S)))
    Window{S,typeof(l),typeof(o)}(l,o)
end
Base.@propagate_inbounds function _window(l::Window,o::Tuple,::Val{S}) where S
    length(S) == length(o) == length(size(l)) || throw(ArgumentError("window rank mismatch"))
    all(n -> n isa Int && n > 0,S) || throw(ArgumentError("positive static extents required"))
    @boundscheck all(map((n,x,w) -> 0 <= x && x <= n-w,size(l),o,S)) ||
        throw(BoundsError(l,(o,S)))
    @inbounds _window(l.parent,map(+,l.origin,o),Val(S))
end

# Exact planning queries; enumerate nonlinear layouts on the host, never
# in a kernel's allocation path. Windows retain allocation-relative offsets.
cosize(l::Union{Composition,Window}) = 1 + maximum(l(i) for i in 0:Int(length(l))-1)

"""
    Ownership((rows, cols), mapping)

Which thread holds which value of a `rows × cols` logical tile. `mapping` is
a layout with two modes, threads then slots, from `(thread, slot)` to the
column-major index of the logical coordinate held there. It says nothing
about memory or register packing. Every thread/value distribution in Tylo
is an `Ownership`: instruction operands, reduction results, windows and
transposes. `coordinate(ownership, thread, slot)` evaluates it.

The constructor stores the mapping in a canonical form, its thread and slot
modes each coalesced, so two ownerships with the same distribution have the
same type.
"""
struct Ownership{S,L<:AbstractLayout}
    mapping::L
    function Ownership{S,L}(l::L) where {S,L<:AbstractLayout}
        S isa Tuple{Int,Int} && all(>(0),S) && length(size(l)) == 2 ||
            throw(ArgumentError("ownership requires a matrix shape and (thread, slot) modes"))
        new{S,L}(l)
    end
end
Base.@constprop :aggressive @inline function Ownership(s::Tuple{Integer,Integer},l::AbstractLayout)
    c = _canonical_mapping(l)
    Ownership{map(Int,s),typeof(c)}(c)
end
"""
    Ownership((rows, cols); threads=1)

The whole tile held by one thread, slot `r + rows*c` holding `(r, c)`; with
`threads`, every one of that many threads holds all of it, as a value
replicated across a warp does. Shape and thread count must be constants.
"""
Base.@constprop :aggressive @inline Ownership(s::Tuple{Integer,Integer};threads::Integer=1) =
    Ownership(s,Layout((static(Int(threads)),static(Int(s[1])*Int(s[2]))),(static(0),static(1))))
Base.size(::Ownership{S}) where S = S
Base.:(==)(a::Ownership,b::Ownership) = size(a) == size(b) && a.mapping == b.mapping

# Logical axis permutation preserves the storage map and parent-relative origin.
@inline function _check_permutation(perm)
    perm isa Tuple{Integer,Integer} && perm in ((1,2),(2,1)) ||
        throw(ArgumentError("expected a permutation tuple of (1,2)"))
    nothing
end
@inline function Base.permutedims(l::Layout,perm=(2,1))
    _check_permutation(perm)
    length(shape(l)) == 2 || throw(ArgumentError("axis permutation currently requires two modes"))
    # A permutation of a valid layout is valid.
    perm == (1,2) ? l : @inbounds Layout(reverse(shape(l)),reverse(strides(l)))
end
@inline function Base.permutedims(l::Window{S},perm=(2,1)) where S
    _check_permutation(perm)
    length(S) == 2 || throw(ArgumentError("axis permutation currently requires two modes"))
    perm == (1,2) && return l
    Window{reverse(S),typeof(permutedims(l.parent)),typeof(reverse(l.origin))}(permutedims(l.parent),reverse(l.origin))
end
@inline Base.permutedims(l::Composition,perm=(2,1)) = compose(l.outer,permutedims(l.inner,perm))
