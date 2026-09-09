"""
    GlobalTile(pointer, layout)
    SharedTile(pointer, layout)

Borrow typed global (address space 1) or shared (address space 3) memory.
The layout maps logical coordinates to element offsets from the allocation
base. Pointer addition occurs in bytes only at `pointer(tile, coordinate)`.
The caller owns allocation, lifetime, alignment and synchronization.
"""
struct MemoryTile{T,AS,L<:Layouts.AbstractLayout}
    ptr::Core.LLVMPtr{T,AS}
    layout::L
    function MemoryTile(ptr::Core.LLVMPtr{T,AS},l::Layouts.AbstractLayout) where {T,AS}
        AS == 1 || AS == 3 || throw(ArgumentError("global/shared pointer required"))
        new{T,AS,typeof(l)}(ptr,l)
    end
end
const GlobalTile{T,L} = MemoryTile{T,1,L}
const SharedTile{T,L} = MemoryTile{T,3,L}
GlobalTile(p::Core.LLVMPtr{T,1},l::Layouts.AbstractLayout) where T = MemoryTile(p,l)
SharedTile(p::Core.LLVMPtr{T,3},l::Layouts.AbstractLayout) where T = MemoryTile(p,l)
Layouts.layout(t::MemoryTile) = t.layout
Base.size(t::MemoryTile) = size(t.layout)
Base.eltype(::Type{<:MemoryTile{T}}) where T = T
Base.eltype(t::MemoryTile) = eltype(typeof(t))
@inline function Base.pointer(t::MemoryTile{T,AS},c::Tuple) where {T,AS}
    t.ptr + _byte_offset(Val(AS),T,t.layout,c)
end
Base.@propagate_inbounds window(t::MemoryTile,o::Tuple,s::Val) =
    MemoryTile(t.ptr,Layouts.window(t.layout,o,s))

# Global products may exceed 32-bit element or byte offsets even when each
# matrix dimension fits Int32. Widen coordinates BEFORE evaluating strides.
@inline _byte_offset(::Val{1},::Type{T},l,c) where T = l(map(Int,c))*sizeof(T)
@inline function _byte_offset(::Val{3},::Type{T},l,c) where T
    offset = l(c)
    offset isa Layouts.StaticInt ? Int(offset)*sizeof(T) : offset*oftype(offset,sizeof(T))
end
