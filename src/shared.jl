# A kernel's dynamic shared memory, declared: named groups of barriers,
# scalars and buffers, laid out in order. The declaration is an ordinary
# value built from constants, so the same function serves the host (the
# launch's byte count) and the kernel (the addresses), and every offset folds
# to a constant in the machine code.

"""
    barriers(n; arrivals)
    barriers((m, n); arrivals)

A group of `n` (or `m × n`, indexed `[i, j]`, `i` fastest) mbarriers in a
[`SharedLayout`](@ref), each initialized for `arrivals` arrivals per phase.
"""
struct BarrierGroup
    shape::Tuple{Int,Int}
    arrivals::Int
end
Base.@constprop :aggressive @inline barriers(n::Integer;arrivals::Integer) = BarrierGroup((Int(n),1),Int(arrivals))
Base.@constprop :aggressive @inline barriers((m,n)::Tuple{Integer,Integer};arrivals::Integer) = BarrierGroup((Int(m),Int(n)),Int(arrivals))

"""
    scalars(T, n=1)

`n` values of type `T` in a [`SharedLayout`](@ref), at `T`'s alignment.
"""
struct ScalarGroup{T}
    count::Int
end
Base.@constprop :aggressive @inline scalars(::Type{T},n::Integer=1) where T = ScalarGroup{T}(Int(n))

"""
    buffers(bytes, n; align=1024)

`n` consecutive buffers of `bytes` each in a [`SharedLayout`](@ref), the
first at an address that is a multiple of `align` (a swizzled operand's
1024 bytes, say; `bytes` a multiple of it keeps every buffer so aligned).
"""
struct BufferGroup
    bytes::Int
    count::Int
    align::Int
end
Base.@constprop :aggressive @inline function buffers(bytes::Integer,n::Integer=1;align::Integer=1024)
    ispow2(align) && align >= 16 || throw(ArgumentError("buffers align to a power of two of at least 16 bytes"))
    BufferGroup(Int(bytes),Int(n),Int(align))
end

_bytes(g::BarrierGroup) = 8*prod(g.shape)
_bytes(g::ScalarGroup{T}) where T = sizeof(T)*g.count
_bytes(g::BufferGroup) = g.bytes*g.count
_align(::BarrierGroup) = 8
_align(::ScalarGroup{T}) where T = Base.datatype_alignment(T)
_align(g::BufferGroup) = g.align
_aligned(g) = _align(g) > 16                  # placed in the aligned part, after the rest

"""
    SharedLayout(groups::NamedTuple)

The dynamic shared memory of a kernel, as named [`barriers`](@ref),
[`scalars`](@ref) and [`buffers`](@ref). Groups aligned to at most 16 bytes
come first, in declaration order, each at its natural alignment; groups
aligned further (swizzled stages) follow from the first address past them
that is so aligned, found when the layout is bound, since dynamic shared
memory itself is only 16-byte aligned. Declare it once, as a function of
the kernel's constants, and call that function on the host and in the
kernel:

```julia
@inline layout(stages, bytes) = SharedLayout((;
    full = barriers(stages; arrivals=1), empty = barriers(stages; arrivals=1),
    slot = scalars(UInt32),
    stage = buffers(bytes, stages; align=1024)))

shmem = shared_bytes(layout(4, bytes))                                   # host
smem = bind(layout(4, bytes), pointer(CuDynamicSharedArray(UInt8, 0)))   # kernel
thread == 0 && (initialize!(smem); fence_barrier_init())
wait(smem.full[s], phase); smem.stage[s]; smem.slot[]
```

Tylo allocates nothing: the kernel binds the layout to its own memory.
"""
struct SharedLayout{G<:NamedTuple}
    groups::G
end

# The offset of each group: from the base for the first part, from the
# aligned start for the second.
@inline _round(x,a) = (x+a-1) & -a
@inline function _offsets(groups::Tuple)
    plain = _place(groups,0,false)
    aligned = _place(groups,0,true)
    ntuple(i -> _aligned(groups[i]) ? aligned[1][i] : plain[1][i],Val(length(groups))), plain[2], aligned[2]
end
@inline _place(::Tuple{},at,second) = ((),at)
@inline function _place(groups::Tuple,at,second)
    g = first(groups)
    mine = _aligned(g) == second
    start = mine ? _round(at,_align(g)) : at
    rest, stop = _place(Base.tail(groups),mine ? start+_bytes(g) : start,second)
    ((start,rest...),stop)
end
@inline _largest_alignment(groups::Tuple) = maximum(map(_align,groups);init=16)

"""
    shared_bytes(layout::SharedLayout) -> Int

The dynamic shared memory a launch must provide for `layout`, including the
slack that aligns its aligned part wherever the memory begins.
"""
@inline function shared_bytes(l::SharedLayout)
    groups = values(l.groups)
    _, plain, aligned = _offsets(groups)
    any(map(_aligned,groups)) || return _round(plain,16)
    _round(plain,16)+_largest_alignment(groups)+aligned
end

"""
    BoundLayout

A [`SharedLayout`](@ref) bound to shared memory by [`bind`](@ref): each
group is a property, `smem.name`. Barriers are a [`Barriers`](@ref) view
(`smem.full[s]` is a [`Barrier`](@ref)), scalars a [`Scalars`](@ref) view
(`smem.sums[i]`, `smem.slot[]`), buffers a [`Buffers`](@ref) view
(`smem.stage[s]`, a shared byte pointer).
"""
struct BoundLayout{G<:NamedTuple}
    layout::SharedLayout{G}
    base::Core.LLVMPtr{UInt8,3}
end
"""
    bind(layout::SharedLayout, pointer) -> BoundLayout

`layout` placed at the shared `pointer`, which must be 16-byte aligned and
lead [`shared_bytes`](@ref)`(layout)` bytes.
"""
@inline Base.bind(l::SharedLayout,p::Core.LLVMPtr{<:Any,3}) = BoundLayout(l,reinterpret(Core.LLVMPtr{UInt8,3},p))

"""
    Barriers

A group of mbarriers in shared memory: `b[i]` (or `b[i, j]`, `i` fastest),
zero-based, is a [`Barrier`](@ref).
"""
struct Barriers
    ptr::Core.LLVMPtr{UInt64,3}
    shape::Tuple{Int,Int}
    strides::Tuple{Int,Int}         # in barriers
end
@inline Base.getindex(b::Barriers,i::Integer) = Barrier(b.ptr,i*oftype(i,b.strides[1]))
@inline Base.getindex(b::Barriers,i::Integer,j::Integer) = Barrier(b.ptr,i*oftype(i,b.strides[1])+j*oftype(j,b.strides[2]))
@inline Base.length(b::Barriers) = prod(b.shape)
"""
    selectdim(b::Barriers, 1, i)

The barriers `b[i, j]` of a two-dimensional group, for every `j`, as a
one-dimensional group: the slots of a [`Ring`](@ref) of their own.
"""
@inline Base.selectdim(b::Barriers,d::Integer,i::Integer) = d == 1 ?
    Barriers(b[i,0].ptr,(b.shape[2],1),(b.strides[2],0)) :
    Barriers(b[0,i].ptr,(b.shape[1],1),(b.strides[1],0))

"""
    Scalars{T}

A group of values of type `T` in shared memory: `x[i]`, zero-based, reads
and `x[i] = v` writes the `i`-th, at `T`'s alignment; `x[]` is the first.
`pointer(x)` is the address of the first.
"""
struct Scalars{T}
    ptr::Core.LLVMPtr{T,3}
    count::Int
end
@inline Base.pointer(x::Scalars) = x.ptr
@inline Base.length(x::Scalars) = x.count
@inline function Base.getindex(x::Scalars{T},i::Integer) where T
    @boundscheck 0 <= i < x.count || throw(BoundsError())
    unsafe_load(x.ptr,i+1,Val(Base.datatype_alignment(T)))
end
@inline function Base.setindex!(x::Scalars{T},v,i::Integer) where T
    @boundscheck 0 <= i < x.count || throw(BoundsError())
    unsafe_store!(x.ptr,convert(T,v),i+1,Val(Base.datatype_alignment(T)))
    x
end
@inline Base.getindex(x::Scalars) = x[0]
@inline Base.setindex!(x::Scalars,v) = (x[0] = v)

"""
    Buffers

A group of consecutive shared buffers: `b[i]`, zero-based, is the byte
pointer to the `i`-th.
"""
struct Buffers
    ptr::Core.LLVMPtr{UInt8,3}
    bytes::Int
    count::Int
end
@inline Base.getindex(b::Buffers,i::Integer) = b.ptr+i*oftype(i,b.bytes)
@inline Base.length(b::Buffers) = b.count

@inline function Base.getproperty(smem::BoundLayout,name::Symbol)
    name === :layout && return getfield(smem,:layout)
    name === :base && return getfield(smem,:base)
    _group(smem,Val(name))
end
Base.propertynames(smem::BoundLayout) = keys(getfield(smem,:layout).groups)
@inline function _group(smem::BoundLayout{G},::Val{name}) where {G,name}
    groups = getfield(smem,:layout).groups
    i = Base.fieldindex(G,name)
    offsets, plain, _ = _offsets(values(groups))
    g = groups[i]
    base = getfield(smem,:base)
    if _aligned(g)
        # The aligned part begins at the first address past the rest that is
        # aligned as its most aligned group.
        a = _largest_alignment(values(groups))
        start = base+_round(plain,16)
        start += (UInt32(0)-_smem_address(start)) & UInt32(a-1)
        _view(g,start+offsets[i])
    else
        _view(g,base+offsets[i])
    end
end
@inline _view(g::BarrierGroup,p) = Barriers(reinterpret(Core.LLVMPtr{UInt64,3},p),g.shape,(1,g.shape[1]))
@inline _view(g::ScalarGroup{T},p) where T = Scalars{T}(reinterpret(Core.LLVMPtr{T,3},p),g.count)
@inline _view(g::BufferGroup,p) = Buffers(p,g.bytes,g.count)

"""
    initialize!(smem::BoundLayout)

Initialize every barrier of the layout for its group's arrivals. One thread
initializes; publish with [`fence_barrier_init`](@ref) and a thread (or
cluster) barrier before use, as for a single barrier.
"""
@inline function initialize!(smem::BoundLayout)
    groups = getfield(smem,:layout).groups
    _initialize!(smem,keys(groups),values(groups))
    nothing
end
@inline _initialize!(smem,::Tuple{},::Tuple{}) = nothing
@inline function _initialize!(smem,names::Tuple,groups::Tuple)
    g = first(groups)
    if g isa BarrierGroup
        b = getproperty(smem,first(names))
        for i in Int32(0):Int32(length(b)-1)
            initialize!(b[i],g.arrivals)
        end
    end
    _initialize!(smem,Base.tail(names),Base.tail(groups))
end

"""
    invalidate!(smem::BoundLayout)

Invalidate every barrier of the layout, before its memory is used for
anything else (another task's layout, say).
"""
@inline function invalidate!(smem::BoundLayout)
    groups = getfield(smem,:layout).groups
    _invalidate!(smem,keys(groups),values(groups))
    nothing
end
@inline _invalidate!(smem,::Tuple{},::Tuple{}) = nothing
@inline function _invalidate!(smem,names::Tuple,groups::Tuple)
    if first(groups) isa BarrierGroup
        b = getproperty(smem,first(names))
        for i in Int32(0):Int32(length(b)-1)
            invalidate!(b[i])
        end
    end
    _invalidate!(smem,Base.tail(names),Base.tail(groups))
end
