"""
    TmemTile(T, address::UInt32, layout)

Borrow TMEM with a logical-to-storage layout. Layout offsets count typed
slots in a virtual column-major grid with 128 hardware lanes: offset `i`
selects lane `i % 128` and typed position `i ÷ 128` along its storage.
FP32 has one position per hardware word; BF16/FP16 have two. Logical axes may be
permuted or partitioned independently of this hardware addressing convention.

The caller owns the allocation, lifetime and synchronization. Construction
neither allocates storage nor proves that the view fits the allocation.
"""
struct TmemTile{T,L<:Layouts.AbstractLayout}
    address::UInt32
    layout::L
    function TmemTile(::Type{T},address::UInt32,l::L) where {T,L<:Layouts.AbstractLayout}
        (T === Float32 || T === BFloat16 || T === Float16) || throw(ArgumentError("TMEM views currently support FP32, BF16 and FP16"))
        length(size(l)) == 2 || throw(ArgumentError("TMEM views require two logical modes"))
        isbitstype(L) || throw(ArgumentError("TMEM layouts must be isbits"))
        new{T,L}(address,l)
    end
end
Layouts.layout(t::TmemTile) = t.layout
Base.size(t::TmemTile) = size(t.layout)
Base.eltype(::Type{<:TmemTile{T}}) where T = T
Base.eltype(t::TmemTile) = eltype(typeof(t))
_tmem_packing(::Type{Float32}) = 1
_tmem_packing(::Type{BFloat16}) = 2
_tmem_packing(::Type{Float16}) = 2
Base.@constprop :aggressive Base.@propagate_inbounds window(t::TmemTile,o::Tuple,s::Tuple) =
    TmemTile(eltype(t),t.address,Layouts.window(t.layout,o,s))
@inline Base.permutedims(t::TmemTile,perm=(2,1)) =
    TmemTile(eltype(t),t.address,permutedims(t.layout,perm))

"""
    tmem_location(tile, coordinate) -> (; address, bit_offset)

Resolve a logical coordinate to an ISA word address and subword bit offset.
This is an address query, not a memory access. Coordinates are zero-based.
Checks cover the view and hardware bounds, not the caller's allocation size.
"""
Base.@propagate_inbounds function tmem_location(tile::TmemTile{T}, coordinate::Tuple) where T
    @boundscheck length(coordinate) == 2 &&
        all(map((value, extent) -> 0 <= value < extent, coordinate, size(tile))) ||
        throw(BoundsError(tile, coordinate))
    slot_index = tile.layout(coordinate)
    @boundscheck 0 <= slot_index < 128 * 1024 || throw(BoundsError(tile, coordinate))
    slot = slot_index % UInt32                           # in range when checked; no InexactError path
    packing = UInt32(_tmem_packing(T))
    lane = (tile.address >> UInt32(16)) + slot % UInt32(128)
    column = (tile.address & UInt32(0xffff)) + slot ÷ UInt32(128) ÷ packing
    bit_offset = (slot ÷ UInt32(128) % packing) * UInt32(32 ÷ _tmem_packing(T))
    @boundscheck lane < UInt32(128) && column < UInt32(512) ||
        throw(BoundsError(tile, coordinate))
    (; address=tile.address + ((slot % UInt32(128)) << UInt32(16)) +
       slot ÷ UInt32(128) ÷ packing, bit_offset)
end

# Recognize affine modes, including a hierarchy that flattens contiguously.
# Nonlinear layouts may describe views, but this transfer does not accept them.
_tmem_mode_stride(s::Layouts.IntLike,d::Layouts.IntLike) = d
function _tmem_mode_stride(s::Tuple,d::Tuple)
    ds = map(_tmem_mode_stride,s,d)
    step = first(ds)
    _check_tmem_mode(s,ds,step,one(step))
    step
end
_check_tmem_mode(::Tuple{},::Tuple{},step,extent) = nothing
function _check_tmem_mode(s::Tuple,ds::Tuple,step,extent)
    first(ds) == step*extent || throw(ArgumentError("TMEM transfer requires an affine logical mode"))
    _check_tmem_mode(Base.tail(s),Base.tail(ds),step,extent*Layouts._volume(first(s)))
end
_tmem_affine(l) = throw(ArgumentError("no TMEM transfer implementation for this storage layout"))
_tmem_affine(l::Layouts.Layout) = (UInt32(0),map(_tmem_mode_stride,Layouts.shape(l),strides(l)))
function _tmem_affine(l::Layouts.Window)
    origin,ds = _tmem_affine(l.parent)
    origin + sum(map((c,d) -> c*Layouts._constant(c,d),l.origin,ds)),ds
end

# The `.32x32b` transfer: lane `t` of a warp holds line `t` of a tile of 32
# lines, its values along the other axis. As an ownership that is
# `distribute((32, n), 32; vector=n, axis=2)`, or its transpose along axis 1;
# any ownership with the same distribution is accepted.
@generated function _transfer_axis(::O) where O<:Layouts.Ownership
    o = _static_instance(O)
    o === nothing && return :(throw(ArgumentError("TMEM transfers require a static ownership")))
    m, n = map(Int,size(o))
    m == 32 && same_distribution(o,Layouts._distribution((32,n),32,n,2)) && return :(Val(2))
    n == 32 && same_distribution(o,Layouts._distribution((m,32),32,m,1)) && return :(Val(1))
    :(throw(ArgumentError("a TMEM transfer gives lane t line t of 32 lines: distribute((32, n), 32; vector=n, axis=2) or its transpose")))
end

# A transfer bound to storage: the band's first word address and the ownership.
struct TmemPartition{T,Words,O}
    address::UInt32
    ownership::O
end
Base.eltype(::Type{<:TmemPartition{T}}) where T = T
Base.eltype(p::TmemPartition) = eltype(typeof(p))
Layouts.layout(p::TmemPartition) = p.ownership
Base.size(p::TmemPartition) = size(p.ownership)

# The tile must cover one aligned 32-lane hardware band with the ownership's
# lines on consecutive lanes and its values in consecutive columns, a power
# of two from 1 to 128 words.
Base.@propagate_inbounds _partition(o::Layouts.Ownership,t::TmemTile) = _partition(o,t,_transfer_axis(o))
Base.@propagate_inbounds function _partition(o::Layouts.Ownership{S},t::TmemTile{T},::Val{A}) where {S,T,A}
    size(t) == S || throw(DimensionMismatch("the TMEM tile must have the ownership's shape"))
    packing = _tmem_packing(T)
    S[A] % packing == 0 || throw(ArgumentError("transfer requires complete TMEM words"))
    words = S[A] ÷ packing
    words in (1,2,4,8,16,32,64,128) || throw(ArgumentError("TMEM transfer supports 1, 2, 4, 8, 16, 32, 64 or 128 words per thread"))
    _,ds = _tmem_affine(t.layout)
    ds[A] == 128 && ds[3-A] == 1 ||
        throw(ArgumentError("storage layout does not match TMEM transfer ownership"))
    loc = tmem_location(t,(UInt32(0),UInt32(0)))
    @boundscheck loc.bit_offset == UInt32(0) || throw(ArgumentError("TMEM transfer starts inside a packed word"))
    @boundscheck begin
        lane,column = loc.address >> UInt32(16),loc.address & UInt32(0xffff)
        lane % UInt32(32) == UInt32(0) && lane <= UInt32(96) ||
            throw(ArgumentError("TMEM transfer must cover an aligned hardware lane band"))
        column + UInt32(words) <= UInt32(512) || throw(BoundsError(t,S))
    end
    TmemPartition{T,words,typeof(o)}(loc.address,o)
end

"""
    load_async(ownership, tile::TmemTile)

Issue a warp-collective `.32x32b` TMEM load of `tile`, whose values the
warp holds in `ownership`: lane `t` holds line `t` of 32, as
`distribute((32, n), 32; vector=n, axis=2)` or its transpose. The tile covers
one aligned 32-lane hardware band, the executing warp's, with its values in
consecutive columns: 1 to 128 words per lane, FP32 or packed BF16/FP16. All
32 lanes execute with the same tile. The result is pending: [`wait_load`](@ref)
returns the fragment.
"""
Base.@propagate_inbounds load_async(o::Layouts.Ownership,t::TmemTile) = load_async(_partition(o,t))
"""
    store_async!(tile::TmemTile, fragment)

Issue a warp-collective `.32x32b` TMEM store of a fragment in a transfer
ownership (see [`load_async`](@ref)) to a tile of its shape: FP32 to FP32
tiles, BF16/FP16 packed two per word. The storage remains in use until
[`wait_stores`](@ref)`()`.
"""
Base.@propagate_inbounds store_async!(t::TmemTile,f::Fragment) = store_async!(_partition(Layouts.layout(f),t),f)

"""
    reinterpret_tile(T, tile::TmemTile; dims)

View the same storage as FP32, BF16 or FP16, explicitly choosing the logical axis
whose extent changes. This changes representation, not values or readiness.
Currently requires a flat rectangular affine view with that axis traversing
consecutive storage positions. 16-bit-to-FP32 views require complete aligned pairs.
"""
Base.@constprop :aggressive Base.@propagate_inbounds function reinterpret_tile(::Type{T},t::TmemTile{U};dims) where {T,U}
    dims isa Integer && dims in (1,2) || throw(ArgumentError("choose logical axis 1 or 2"))
    (T === Float32 || T === BFloat16 || T === Float16) || throw(ArgumentError("TMEM views support FP32, BF16 and FP16"))
    T === U && return t
    _tmem_packing(T) == _tmem_packing(U) && return TmemTile(T,t.address,t.layout)
    all(n -> n isa Layouts.IntLike,Layouts.shape(t.layout)) ||
        throw(ArgumentError("reinterpretation requires flat logical modes"))
    origin,ds = _tmem_affine(t.layout)
    ds[dims] == 128 && ds[3-dims] == 1 ||
        throw(ArgumentError("reinterpretation axis must follow consecutive storage positions"))
    n = size(t)[dims]
    _tmem_packing(U) > _tmem_packing(T) && (isodd(n) || isodd(origin ÷ 128)) &&
        throw(ArgumentError("FP32 reinterpretation requires complete aligned element pairs"))
    loc = tmem_location(t,(UInt32(0),UInt32(0)))
    loc.bit_offset == 0 || throw(ArgumentError("reinterpretation starts inside a packed word"))
    # Rebase an affine view at its first word; this preserves the original
    # allocation-relative address while changing only typed slot interpretation.
    resize(n) = n * Layouts.static(_tmem_packing(T)) ÷ Layouts.static(_tmem_packing(U))
    a,b = size(t)
    s = dims == 1 ? (resize(a),b) : (a,resize(b))
    TmemTile(T,loc.address,Layouts.Layout(s,ds))
end

# Pending registers carry the transfer ownership through the explicit wait.
struct PendingLoad{T,N,L}
    words::NTuple{N,UInt32}
    ownership::L
end
PendingLoad(::Type{T}, words::NTuple{N,UInt32}, ownership::L) where {T,N,L} = PendingLoad{T,N,L}(words,ownership)

@inline function _check_tmem_store(p::TmemPartition,f)
    _same_distribution(Layouts.layout(p),Layouts.layout(f)) ||
        throw(DimensionMismatch("TMEM store ownership differs from transfer"))
    nothing
end

function Base.show(io::IO,t::TmemTile{T}) where T
    print(io,"TmemTile(",T,", 0x",string(t.address;base=16,pad=8),", ")
    show(io,t.layout)
    print(io,')')
end
