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
    window(layout, origin, Val(shape))

A static-size logical window with a potentially runtime origin. Offsets stay
relative to the parent's allocation base, preserving swizzle phase. Bounds
are checked unless the caller uses `@inbounds` after establishing validity.
"""
Base.@propagate_inbounds function window(l::AbstractLayout,o::Tuple,::Val{S}) where S
    length(S) == length(o) == length(size(l)) || throw(ArgumentError("window rank mismatch"))
    all(n -> n isa Int && n > 0,S) || throw(ArgumentError("positive static extents required"))
    @boundscheck all(map((n,x,w) -> 0 <= x && x <= n-w,size(l),o,S)) ||
        throw(BoundsError(l,(o,S)))
    Window{S,typeof(l),typeof(o)}(l,o)
end
Base.@propagate_inbounds function window(l::Window,o::Tuple,::Val{S}) where S
    length(S) == length(o) == length(size(l)) || throw(ArgumentError("window rank mismatch"))
    all(n -> n isa Int && n > 0,S) || throw(ArgumentError("positive static extents required"))
    @boundscheck all(map((n,x,w) -> 0 <= x && x <= n-w,size(l),o,S)) ||
        throw(BoundsError(l,(o,S)))
    @inbounds window(l.parent,map(+,l.origin,o),Val(S))
end

# Exact planning queries; enumerate nonlinear layouts on the host, never
# in a kernel's allocation path. Windows retain allocation-relative offsets.
cosize(l::Union{Composition,Window}) = 1 + maximum(l(i) for i in 0:Int(length(l))-1)

"""
    Ownership(Val((rows, cols)), thread_value_layout)

Map `(lane, logical_value)` to a matrix coordinate through a hierarchical
layout. The layout's output is a column-major logical index, independent of
physical memory layout and register packing.
"""
struct Ownership{S,L<:AbstractLayout}
    mapping::L
end
function Ownership(::Val{S},l::AbstractLayout) where S
    length(S) == 2 && all(n -> n isa Int && n > 0,S) && length(size(l)) == 2 ||
        throw(ArgumentError("ownership requires matrix and thread/value domains"))
    Ownership{S,typeof(l)}(l)
end
Base.size(::Ownership{S}) where S = S
@inline function coordinate(o::Ownership{S},lane::Integer,::Val{E}) where {S,E}
    E isa Int && 0 <= E < size(o.mapping)[2] || throw(BoundsError())
    i = o.mapping((lane,oftype(lane,E)))
    m = oftype(i,S[1])
    (i % m,i ÷ m)
end
