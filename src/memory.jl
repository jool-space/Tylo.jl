"""
    GlobalTile(pointer, layout, Val(align)=Val(sizeof(T)))
    SharedTile(pointer, layout, Val(align)=Val(sizeof(T)))

Borrow typed global (address space 1) or shared (address space 3) memory.
The layout maps logical coordinates to element offsets from the allocation
base. Pointer addition occurs in bytes only at `pointer(tile, coordinate)`.
The caller owns allocation, lifetime, alignment and synchronization.

`align` declares, in bytes, that the pointer and every stride of the
layout other than the unit stride are multiples of `align`, so that any
element whose coordinate along the unit-stride axis is a multiple of
`align ÷ sizeof(T)` has an `align`-aligned address. Loads and stores of
fragments vectorize up to that width. A window's origin along the
unit-stride axis must keep the declaration. Bounds checks verify the
pointer, the strides and window origins unless elided with `@inbounds`.
"""
struct MemoryTile{T,AS,L<:Layouts.AbstractLayout,A}
    ptr::Core.LLVMPtr{T,AS}
    layout::L
    Base.@propagate_inbounds function MemoryTile(ptr::Core.LLVMPtr{T,AS},l::Layouts.AbstractLayout,::Val{A}=Val(sizeof(T))) where {T,AS,A}
        AS == 1 || AS == 3 || throw(ArgumentError("global/shared pointer required"))
        A isa Int && ispow2(A) && A >= sizeof(T) || throw(ArgumentError("alignment is a power of two of at least the element size"))
        @boundscheck A == sizeof(T) || _aligned(ptr,l,T,A) || throw(ArgumentError("pointer, strides or window origin break the declared alignment"))
        new{T,AS,typeof(l),A}(ptr,l)
    end
end
const GlobalTile{T,L,A} = MemoryTile{T,1,L,A}
const SharedTile{T,L,A} = MemoryTile{T,3,L,A}
Base.@propagate_inbounds GlobalTile(p::Core.LLVMPtr{T,1},l::Layouts.AbstractLayout,a::Val=Val(sizeof(T))) where T = MemoryTile(p,l,a)
Base.@propagate_inbounds SharedTile(p::Core.LLVMPtr{T,3},l::Layouts.AbstractLayout,a::Val=Val(sizeof(T))) where T = MemoryTile(p,l,a)
Layouts.layout(t::MemoryTile) = t.layout
Base.size(t::MemoryTile) = size(t.layout)
Base.eltype(::Type{<:MemoryTile{T}}) where T = T
Base.eltype(t::MemoryTile) = eltype(typeof(t))
alignment(::MemoryTile{T,AS,L,A}) where {T,AS,L,A} = A
alignment(::Type{<:MemoryTile{T,AS,L,A}}) where {T,AS,L,A} = A
@inline function Base.pointer(t::MemoryTile{T,AS},c::Tuple) where {T,AS}
    t.ptr + _byte_offset(Val(AS),T,t.layout,c)
end
Base.@propagate_inbounds window(t::MemoryTile{T,AS,L,A},o::Tuple,s::Val) where {T,AS,L,A} =
    MemoryTile(t.ptr,Layouts.window(t.layout,o,s),Val(A))

# Global products may exceed 32-bit element or byte offsets even when each
# matrix dimension fits Int32. Widen coordinates BEFORE evaluating strides.
@inline _byte_offset(::Val{1},::Type{T},l,c) where T = l(map(Int,c))*sizeof(T)
@inline function _byte_offset(::Val{3},::Type{T},l,c) where T
    offset = l(c)
    offset isa Layouts.StaticInt ? Int(offset)*sizeof(T) : offset*oftype(offset,sizeof(T))
end

# The alignment declaration, checked structurally: every non-unit stride
# and the pointer are multiples of `A`; a window's origin along the unit
# stride keeps the multiple; a swizzle limits the claim to the bytes it
# leaves in place.
@inline _aligned(ptr,l,::Type{T},A) where T = reinterpret(UInt64,ptr) % UInt64(A) == 0 && _aligned_layout(l,T,A)
@inline _aligned_layout(l::Layouts.Layout,::Type{T},A) where T = _aligned_strides(Layouts.shape(l),strides(l),T,A)
@inline _aligned_strides(s::Tuple,d::Tuple,::Type{T},A) where T = all(map((x,y) -> _aligned_strides(x,y,T,A),s,d))
@inline _aligned_strides(s,d,::Type{T},A) where T = Int(s) <= 1 || Int(d) == 1 || (Int(d)*sizeof(T)) % A == 0
@inline _aligned_layout(l::Layouts.Composition{Layouts.Swizzle{B,M,S}},::Type{T},A) where {B,M,S,T} =
    A <= (1 << M)*sizeof(T) && _aligned_layout(l.inner,T,A)
@inline function _aligned_layout(l::Layouts.Window,::Type{T},A) where T
    _aligned_layout(l.parent,T,A) || return false
    axis = _unit_axis(l.parent)
    axis === nothing || (Int(l.origin[axis])*sizeof(T)) % A == 0
end
@inline _aligned_layout(l,::Type,A) = false
@inline _unit_axis(l::Layouts.Layout) = _unit_axis(strides(l))
@inline _unit_axis(l::Layouts.Composition) = _unit_axis(l.inner)
@inline _unit_axis(l::Layouts.Window) = _unit_axis(l.parent)
@inline _unit_axis(d::Tuple) = _unit_axis(d,1)
@inline _unit_axis(::Tuple{},i) = nothing
@inline _unit_axis(d::Tuple,i) = _has_unit(first(d)) ? i : _unit_axis(Base.tail(d),i+1)
@inline _has_unit(d::Tuple) = any(map(_has_unit,d))
@inline _has_unit(d) = Int(d) == 1
