"""
    GlobalTile(pointer, layout, align=sizeof(T))
    SharedTile(pointer, layout, align=sizeof(T))

Borrow typed global (address space 1) or shared (address space 3) memory.
The layout maps logical coordinates to element offsets from the allocation
base. Pointer addition occurs in bytes only at `pointer(tile, coordinate)`.
The caller owns allocation, lifetime, alignment and synchronization.

`align` declares, in bytes, that the tile's base address and every runtime
stride of its layout are multiples of `align`. It must be a constant. Static
strides are known exactly, so loads and stores of fragments derive vectors
up to `align` bytes from the layout itself, even through permuted or
hierarchical layouts. Bounds checks verify the declaration unless elided with
`@inbounds`.

A tile may carry the extents of its valid region, from [`bound`](@ref);
loads through it zero the elements outside and stores skip them.
"""
struct MemoryTile{T,AS,L<:Layouts.AbstractLayout,A,B}
    ptr::Core.LLVMPtr{T,AS}
    layout::L
    bounds::B
    Base.@propagate_inbounds function MemoryTile(ptr::Core.LLVMPtr{T,AS},l::Layouts.AbstractLayout,::Val{A}=Val(sizeof(T)),
            bounds::B=nothing) where {T,AS,A,B}
        AS == 1 || AS == 3 || throw(ArgumentError("global/shared pointer required"))
        A isa Int && ispow2(A) && A >= sizeof(T) || throw(ArgumentError("alignment is a power of two of at least the element size"))
        B === Nothing || B <: Tuple{Union{Integer,Nothing},Union{Integer,Nothing}} ||
            throw(ArgumentError("bounds are the valid extents of the tile's two axes, or nothing"))
        @boundscheck A == sizeof(T) || _aligned(ptr,l,T,A) || throw(ArgumentError("pointer, strides or window origin break the declared alignment"))
        new{T,AS,typeof(l),A,B}(ptr,l,bounds)
    end
end
# The 32-bit shared-window address of a shared pointer. A shared pointer's
# value is that address, which host code (descriptor tests) reads directly;
# kernels convert with the instruction CUDACoreExt supplies.
@inline _smem_address(p::Core.LLVMPtr{T,3}) where T = reinterpret(UInt64,p) % UInt32
const GlobalTile{T,L,A} = MemoryTile{T,1,L,A}
const SharedTile{T,L,A} = MemoryTile{T,3,L,A}
Base.@constprop :aggressive Base.@propagate_inbounds GlobalTile(p::Core.LLVMPtr{T,1},l::Layouts.AbstractLayout,
    align::Integer=sizeof(T)) where T = MemoryTile(p,l,Val(Int(align)))
Base.@constprop :aggressive Base.@propagate_inbounds SharedTile(p::Core.LLVMPtr{T,3},l::Layouts.AbstractLayout,
    align::Integer=sizeof(T)) where T = MemoryTile(p,l,Val(Int(align)))
"""
    GlobalTile(A::DenseArray, align=sizeof(eltype(A)))
    SharedTile(A::DenseArray, align=sizeof(eltype(A)))
    GlobalTile(T, pointer, layout, align=sizeof(T))
    SharedTile(T, pointer, layout, align=sizeof(T))

The tile of a device array as the column-major matrix it is (a vector as one
column), with `Int32` extents; or a tile of `T` at a pointer of any element
type, such as a workspace's byte pointer.
"""
Base.@constprop :aggressive Base.@propagate_inbounds GlobalTile(A::DenseArray,align::Integer=sizeof(eltype(A))) = GlobalTile(pointer(A),_column_major(A),align)
Base.@constprop :aggressive Base.@propagate_inbounds SharedTile(A::DenseArray,align::Integer=sizeof(eltype(A))) = SharedTile(pointer(A),_column_major(A),align)
Base.@constprop :aggressive Base.@propagate_inbounds GlobalTile(::Type{T},p::Core.LLVMPtr{<:Any,1},l::Layouts.AbstractLayout,align::Integer=sizeof(T)) where T =
    GlobalTile(reinterpret(Core.LLVMPtr{T,1},p),l,align)
Base.@constprop :aggressive Base.@propagate_inbounds SharedTile(::Type{T},p::Core.LLVMPtr{<:Any,3},l::Layouts.AbstractLayout,align::Integer=sizeof(T)) where T =
    SharedTile(reinterpret(Core.LLVMPtr{T,3},p),l,align)
"""
    permutedims(tile)

The same memory as the transposed tile: coordinate `(r, c)` becomes `(c, r)`,
bounds included.
"""
@inline Base.permutedims(t::MemoryTile{T,AS,L,A},perm=(2,1)) where {T,AS,L,A} = perm == (1,2) ? t :
    @inbounds MemoryTile(t.ptr,permutedims(t.layout,perm),Val(A),t.bounds === nothing ? nothing : reverse(t.bounds))
# An array's own extents need no validity check.
@inline _column_major(A::AbstractVector) = (n = length(A) % Int32; @inbounds Layouts.Layout((n,Layouts.static(1)),(Layouts.static(1),n)))
@inline _column_major(A::AbstractMatrix) = (m = size(A,1) % Int32; @inbounds Layouts.Layout((m,size(A,2) % Int32),(Layouts.static(1),m)))
Layouts.layout(t::MemoryTile) = t.layout
Base.size(t::MemoryTile) = size(t.layout)
Base.eltype(::Type{<:MemoryTile{T}}) where T = T
Base.eltype(t::MemoryTile) = eltype(typeof(t))
alignment(::MemoryTile{T,AS,L,A}) where {T,AS,L,A} = A
alignment(::Type{<:MemoryTile{T,AS,L,A}}) where {T,AS,L,A} = A
@inline function Base.pointer(t::MemoryTile{T,AS},c::Tuple) where {T,AS}
    t.ptr + _byte_offset(Val(AS),T,t.layout,c)
end
"""
    tile[(r, c)]
    tile[(r, c)] = value

One element, at a zero-based coordinate as for [`window`](@ref) and
`pointer(tile, (r, c))`, loaded or stored with the element's alignment. The
coordinate must lie inside the tile and its bounds (checked unless elided
with `@inbounds`). Fragments move many elements at once; this is for the
odd element and for tests.
"""
Base.@propagate_inbounds function Base.getindex(t::MemoryTile{T},c::Tuple{Integer,Integer}) where T
    @boundscheck _element_inside(t,c) || throw(BoundsError(t,c))
    unsafe_load(pointer(t,c),1,Val(sizeof(T)))
end
Base.@propagate_inbounds function Base.setindex!(t::MemoryTile{T},x,c::Tuple{Integer,Integer}) where T
    @boundscheck _element_inside(t,c) || throw(BoundsError(t,c))
    unsafe_store!(pointer(t,c),convert(T,x),1,Val(sizeof(T)))
    t
end
@inline _element_inside(t::MemoryTile,c) =
    all(map((x,n,b) -> 0 <= x < n && (b === nothing || x < b),c,size(t),t.bounds === nothing ? (nothing,nothing) : t.bounds))
"""
    window(tile, origin, shape)

The `shape` sub-tile at a possibly runtime `origin`, addressing the same
allocation. `shape` must be a constant tuple. The window must lie inside
the tile (checked unless elided with `@inbounds`), except along an axis
with a bound (see [`bound`](@ref)), past which it may extend: its bound
then says which of its elements exist.
"""
Base.@constprop :aggressive Base.@propagate_inbounds function window(t::MemoryTile{T,AS,L,A},o::Tuple,s::Tuple) where {T,AS,L,A}
    @boundscheck _window_inside(t.bounds,size(t),o,s) || throw(BoundsError(t,(o,s)))
    MemoryTile(t.ptr,Layouts._window_unchecked(t.layout,o,Val(map(Int,s))),Val(A),_shift_bounds(t.bounds,o))
end
@inline _window_inside(::Nothing,n,o,s) = all(map((n,x,w) -> 0 <= x && x <= n-w,n,o,s))
@inline _window_inside(b::Tuple,n,o,s) = all(map((b,n,x,w) -> 0 <= x && (b !== nothing || x <= n-w),b,n,o,s))
@inline _shift_bounds(::Nothing,o) = nothing
@inline _shift_bounds(b::Tuple,o) = map(_shift,b,o)
@inline _shift(::Nothing,o) = nothing
@inline _shift(x,o) = x - oftype(x,o)

"""
    bound(tile, extents)

The same tile, valid only below `extents` along each axis (`nothing` leaves
an axis unchecked). Loads through a bounded tile zero the elements outside
and stores skip them, a vector at a time: a bound falling inside a vector is
an error, checked unless elided with `@inbounds`, so along the axis vectors
run the extent must be a multiple of the vector length. Coordinates are
non-negative, so only the upper bounds are checked.
Sub-tiles from `window` and `tile` inherit the bounds, shifted to their
origin.
"""
@inline bound(t::MemoryTile{T,AS,L,A},extents::Tuple{Any,Any}) where {T,AS,L,A} =
    @inbounds MemoryTile(t.ptr,t.layout,Val(A),map(_extent,extents))
# Static extents, as `size` gives them for static modes, bound as integers.
@inline _extent(::Nothing) = nothing
@inline _extent(n::Integer) = n
@inline _extent(n::Layouts.StaticInt) = Int(n)

"""
    tile(t, shape, index)

The zero-based `index`-th tile of `shape` in a tiling of `t`, as a tile of
its own: the pointer advanced to that tile and a layout of the modes inside
it. Along an axis whose mode is hierarchical, `shape` must cover whole inner
modes; the tile then never carries into the outer modes, so no runtime
division is needed to address it, which a `window` at an arbitrary origin
could not promise. `shape` must be a constant. The tile must lie inside `t` (checked unless
elided with `@inbounds`), except along a bounded axis.

A bounded parent (see [`bound`](@ref)) gives a bounded tile: the tile may
then extend past the valid region, whose elements loads zero and stores
skip.
"""
Base.@constprop :aggressive Base.@propagate_inbounds function Layouts.tile(t::MemoryTile{T,AS,L,A},shape::Tuple{Integer,Integer},
        index::Tuple{Integer,Integer}) where {T,AS,L,A}
    inner, outer = _split_tile(t.layout,Val(map(Int,shape)))
    offset = _tile_offset(outer[1],index[1]) + _tile_offset(outer[2],index[2])
    origin = map((i,n) -> i*oftype(i,n),index,shape)
    @boundscheck _window_inside(t.bounds,size(t),origin,shape) || throw(BoundsError(t,(index,shape)))
    # The parent's declaration covers the tile: runtime strides are multiples
    # of it, and the declaration narrows to the static outer strides.
    @inbounds MemoryTile(t.ptr + _widen(Val(AS),offset)*sizeof(T),inner,_tile_alignment(typeof(outer),T,Val(A)),
                         _shift_bounds(t.bounds,origin))
end
# The largest power of two up to `A` dividing every static outer stride, in bytes.
@generated _tile_alignment(::Type{O},::Type{T},::Val{A}) where {O,T,A} = :(Val($(_tile_alignment_bytes(O,T,A))))
function _tile_alignment_bytes(O,T,A)
    strides = Int[]
    for o in O.parameters
        if o <: Layouts.StaticInt
            push!(strides,o.parameters[1])
        elseif o <: Layouts.Layout
            for (_,d) in _type_leaves(o.parameters[1],o.parameters[2])
                d <: Layouts.StaticInt && push!(strides,d.parameters[1])
            end
        end
    end
    a = A
    while a > sizeof(T) && any(d -> (d*sizeof(T)) % a != 0,strides)
        a ÷= 2
    end
    a
end
_static_volume(::Type{Layouts.StaticInt{n}}) where n = n
_static_volume(T::Type{<:Tuple}) = (v = map(_static_volume,T.parameters); any(isnothing,v) ? nothing : prod(v; init=1))
_static_volume(::Type) = nothing
@inline _tile_offset(stride,i) = i*stride
@inline _tile_offset(l::Layouts.Layout,i) = Layouts._offset_nonnegative(l,i)
@inline _widen(::Val{1},x) = Int(x)
@inline _widen(::Val{3},x) = x
# The inner layout of a tile of `shape` and the stride of the tile index
# along each axis: a flat mode splits at the tile extent; a hierarchical
# mode must cover `shape` with a prefix of its leaves exactly.
@generated function _split_tile(l::Layouts.Layout{S,D},::Val{E}) where {S,D,E}
    inner_s, inner_d, outer = Any[], Any[], Any[]
    for a in 1:2
        s_t, d_t = S.parameters[a], D.parameters[a]
        if !(s_t <: Tuple)
            push!(inner_s,:(Layouts.static($(E[a])))); push!(inner_d,:(strides(l)[$a]))
            push!(outer,:(Layouts.static($(E[a]))*strides(l)[$a]))
        else
            # Children inside the tile, whole subtrees of static extent; a
            # static leaf may split where the extent ends.
            children = collect(s_t.parameters); volume = 1; k = 0
            ins, ind = Any[], Any[]
            while volume < E[a]
                k < length(children) || return :(throw(ArgumentError("the tile extent exceeds the mode")))
                k += 1; n = _static_volume(children[k])
                n === nothing && return :(throw(ArgumentError("a tile must cover static inner modes")))
                if volume*n <= E[a]
                    push!(ins,:(Layouts.shape(l)[$a][$k])); push!(ind,:(strides(l)[$a][$k])); volume *= n
                else
                    children[k] <: Layouts.StaticInt ||
                        return :(throw(ArgumentError("a tile may split only a leaf of the mode")))
                    part = E[a] ÷ volume
                    part*volume == E[a] && n % part == 0 ||
                        return :(throw(ArgumentError("the tile extent must split the inner modes evenly")))
                    push!(ins,:(Layouts.static($part))); push!(ind,:(strides(l)[$a][$k])); volume *= part
                    # The tile index then runs over the rest of this leaf first, then the outer children.
                    rest = [(:(Layouts.static($(n ÷ part))),:(Layouts.static($part)*strides(l)[$a][$k]))]
                    for j in k+1:length(children)
                        push!(rest,(:(Layouts.shape(l)[$a][$j]),:(strides(l)[$a][$j])))
                    end
                    push!(inner_s,Expr(:tuple,ins...)); push!(inner_d,Expr(:tuple,ind...))
                    push!(outer,:(@inbounds Layouts.Layout(($([r[1] for r in rest]...),),($([r[2] for r in rest]...),))))
                    @goto next_axis
                end
            end
            k < length(children) || return :(throw(ArgumentError("the tile must leave an outer mode to index")))
            # One child inside: keep it as the axis's mode; several: a tuple of them.
            push!(inner_s,length(ins) == 1 ? ins[1] : Expr(:tuple,ins...))
            push!(inner_d,length(ind) == 1 ? ind[1] : Expr(:tuple,ind...))
            # The tile index runs over the outer children, first fastest.
            push!(outer,k+1 == length(children) ? :(strides(l)[$a][$(k+1)]) :
                :(@inbounds Layouts.Layout(($([:(Layouts.shape(l)[$a][$j]) for j in k+1:length(children)]...),),
                                           ($([:(strides(l)[$a][$j]) for j in k+1:length(children)]...),))))
        end
        @label next_axis
    end
    quote
        Base.@inline
        @inbounds (Layouts.Layout(($(inner_s...),),($(inner_d...),)),($(outer...),))
    end
end

# Global products may exceed 32-bit element or byte offsets even when each
# matrix dimension fits Int32. Widen coordinates BEFORE evaluating strides.
@inline _byte_offset(::Val{1},::Type{T},l,c) where T = l(map(Int,c))*sizeof(T)
@inline function _byte_offset(::Val{3},::Type{T},l,c) where T
    offset = l(c)
    offset isa Layouts.StaticInt ? Int(offset)*sizeof(T) : offset*oftype(offset,sizeof(T))
end

# The alignment declaration: the base address and every runtime stride are
# multiples of `A`; a swizzle limits the claim to the bytes it leaves in
# place; a window's base is its parent's address of the origin.
@inline _aligned(ptr,l,::Type{T},A) where T = _aligned_base(ptr,l,T,A) && _aligned_layout(l,T,A)
@inline _aligned_base(ptr,l,::Type{T},A) where T = reinterpret(UInt64,ptr) % UInt64(A) == 0
@inline _aligned_base(ptr,l::Layouts.Window,::Type{T},A) where T =
    (reinterpret(UInt64,ptr) + UInt64(Int(l.parent(l.origin))*sizeof(T))) % UInt64(A) == 0
@inline _aligned_layout(l::Layouts.Layout,::Type{T},A) where T = _aligned_strides(Layouts.shape(l),strides(l),T,A)
@inline _aligned_strides(s::Tuple,d::Tuple,::Type{T},A) where T = all(map((x,y) -> _aligned_strides(x,y,T,A),s,d))
@inline _aligned_strides(s,d::Layouts.StaticInt,::Type{T},A) where T = true
@inline _aligned_strides(s,d,::Type{T},A) where T = Int(s) <= 1 || Int(d) == 1 || (Int(d)*sizeof(T)) % A == 0
@inline _aligned_layout(l::Layouts.Composition{Layouts.Swizzle{B,M,S}},::Type{T},A) where {B,M,S,T} =
    A <= (1 << M)*sizeof(T) && _aligned_layout(l.inner,T,A)
@inline _aligned_layout(l::Layouts.Window,::Type{T},A) where T = _aligned_layout(l.parent,T,A)
@inline _aligned_layout(l,::Type,A) = false


# Affine tiles, a flat layout or a window of one, address linearly: the
# address of `a + b` is the address of `a` plus the offset of `b`. Loads and
# stores of an ownership then form one base pointer per thread and add each
# slot's constant offset, which the backend folds into the instruction.
_affine(::Type) = false
_affine(::Type{Layouts.Layout{S,D}}) where {S,D} = !any(p -> p <: Tuple,S.parameters) && !any(p -> p <: Tuple,D.parameters)
_affine(::Type{Layouts.Window{S,L,O}}) where {S,L,O} = _affine(L)
@inline _linear(l::Layouts.Layout) = l
@inline _linear(l::Layouts.Window) = l.parent
"The address of slot `e` of `ownership` in an affine tile, from the thread's base address."
@inline _slot_address(t::MemoryTile{T},base,c::Tuple{Int,Int}) where T =
    base + _linear(t.layout)(c)*sizeof(T)


function Base.show(io::IO,t::MemoryTile{T,AS,L,A,B}) where {T,AS,L,A,B}
    B === Nothing || print(io,"bound(")
    print(io,AS == 1 ? "GlobalTile(" : "SharedTile(",T,", 0x",string(reinterpret(UInt64,t.ptr);base=16),", ")
    show(io,t.layout)
    print(io,", ",A,')')
    B === Nothing || (print(io,", "); show(io,t.bounds); print(io,')'))
end
