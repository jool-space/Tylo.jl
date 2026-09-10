"""
    TmemTile(T, address::UInt32, layout)

Borrow TMEM with a logical-to-storage layout. Layout offsets count typed
slots in a virtual column-major grid with 128 hardware lanes: offset `i`
selects lane `i % 128` and typed position `i ÷ 128` along its storage.
FP32 has one position per hardware word; BF16 has two. Logical axes may be
permuted or partitioned independently of this hardware addressing convention.

The caller owns the allocation, lifetime and synchronization. Construction
neither allocates storage nor proves that the view fits the allocation.
"""
struct TmemTile{T,L<:Layouts.AbstractLayout}
    address::UInt32
    layout::L
    function TmemTile(::Type{T},address::UInt32,l::L) where {T,L<:Layouts.AbstractLayout}
        T in (Float32,BFloat16) || throw(ArgumentError("TMEM views currently support FP32 and BF16"))
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
Base.@propagate_inbounds window(t::TmemTile,o::Tuple,s::Val) =
    TmemTile(eltype(t),t.address,Layouts.window(t.layout,o,s))
Base.permutedims(t::TmemTile,perm=(2,1)) =
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
    slot = UInt32(slot_index)
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

"""
    TmemTransfer{Shape,Axis}()

A warp-collective TMEM/register transfer using the `.32x32b` instruction.
`Shape` is the logical partition shape. Each thread holds values along
logical `Axis`; the other axis has extent 32. This plan also describes the
resulting register ownership. `permutedims(plan)` exchanges its logical axes.

Bind it to a storage view with `partition(plan, tile)`. Supported payloads
are 16, 32 or 64 words per thread: FP32 loads/stores, and packed BF16 stores.
Instruction shape, ownership and storage compatibility are checked explicitly.
"""
struct TmemTransfer{S,Axis}
    function TmemTransfer{S,Axis}() where {S,Axis}
        S isa Tuple && length(S) == 2 && all(n -> n isa Int && n > 0,S) &&
            Axis isa Int && Axis in (1,2) && S[3-Axis] == 32 ||
            throw(ArgumentError("TMEM transfer needs 32 threads along one logical axis"))
        new{S,Axis}()
    end
end
Base.size(::TmemTransfer{S}) where S = S
Layouts.layout(p::TmemTransfer) = p
_register_count(::TmemTransfer{S,A}) where {S,A} = S[A]
@inline function Layouts.coordinate(::TmemTransfer{S,A},t::Integer,::Val{E}) where {S,A,E}
    E isa Int && 0 <= E < S[A] || throw(BoundsError())
    A == 2 ? (t,oftype(t,E)) : (oftype(t,E),t)
end
function Base.permutedims(p::TmemTransfer{S,A},perm=(2,1)) where {S,A}
    perm isa Tuple{Integer,Integer} || throw(ArgumentError("expected a two-axis permutation tuple"))
    perm == (1,2) && return p
    perm == (2,1) || throw(ArgumentError("expected a permutation of (1,2)"))
    TmemTransfer{reverse(S),3-A}()
end

struct TmemPartition{T,Words,P}
    address::UInt32
    transfer::P
end
Base.eltype(::Type{<:TmemPartition{T}}) where T = T
Base.eltype(p::TmemPartition) = eltype(typeof(p))
Layouts.layout(p::TmemPartition) = p.transfer
Base.size(p::TmemPartition) = size(p.transfer)
Base.permutedims(p::TmemPartition{T,W},perm=(2,1)) where {T,W} =
    TmemPartition{T,W,typeof(permutedims(p.transfer,perm))}(p.address,permutedims(p.transfer,perm))

"""
    partition(transfer::TmemTransfer, tile::TmemTile)

Bind a transfer to a logical storage view with matching shape and ownership.
Use `window` to select a warp's region before partitioning. The transfer must
cover one aligned 32-lane hardware band and contiguous storage positions.
The executing warp must match that band's index within its warpgroup, and
all 32 threads must use the same partition. Types do not prove participation.

Bounds checks may be omitted with `@inbounds` after establishing the storage
contract; unsupported instruction widths/layouts are always rejected.
"""
Base.@propagate_inbounds function partition(p::TmemTransfer{S,A},t::TmemTile{T}) where {S,A,T}
    size(t) == S || throw(DimensionMismatch("transfer and TMEM view shapes differ"))
    packing = _tmem_packing(T)
    S[A] % packing == 0 || throw(ArgumentError("transfer requires complete TMEM words"))
    words = S[A] ÷ packing
    words in (16,32,64) || throw(ArgumentError("TMEM transfer supports 16, 32 or 64 words per thread"))
    _,ds = _tmem_affine(t.layout)
    ds[A] == 128 && ds[3-A] == 1 ||
        throw(ArgumentError("storage layout does not match TMEM transfer ownership"))
    loc = tmem_location(t,(UInt32(0),UInt32(0)))
    loc.bit_offset == UInt32(0) || throw(ArgumentError("TMEM transfer starts inside a packed word"))
    @boundscheck begin
        lane,column = loc.address >> UInt32(16),loc.address & UInt32(0xffff)
        lane % UInt32(32) == UInt32(0) && lane <= UInt32(96) ||
            throw(ArgumentError("TMEM transfer must cover an aligned hardware lane band"))
        column + UInt32(words) <= UInt32(512) || throw(BoundsError(t,S))
    end
    TmemPartition{T,words,typeof(p)}(loc.address,p)
end

"""
    reinterpret_tile(T, tile::TmemTile; dims)

View the same storage as FP32 or BF16, explicitly choosing the logical axis
whose extent changes. This changes representation, not values or readiness.
Currently requires a flat rectangular affine view with that axis traversing
consecutive storage positions. BF16-to-FP32 views require complete aligned pairs.
"""
Base.@constprop :aggressive Base.@propagate_inbounds function reinterpret_tile(::Type{T},t::TmemTile{U};dims) where {T,U}
    dims isa Integer && dims in (1,2) || throw(ArgumentError("choose logical axis 1 or 2"))
    T in (Float32,BFloat16) || throw(ArgumentError("TMEM views support FP32 and BF16"))
    T === U && return t
    all(n -> n isa Layouts.IntLike,Layouts.shape(t.layout)) ||
        throw(ArgumentError("reinterpretation requires flat logical modes"))
    origin,ds = _tmem_affine(t.layout)
    ds[dims] == 128 && ds[3-dims] == 1 ||
        throw(ArgumentError("reinterpretation axis must follow consecutive storage positions"))
    n = size(t)[dims]
    U === BFloat16 && (isodd(n) || isodd(origin ÷ 128)) &&
        throw(ArgumentError("FP32 reinterpretation requires complete aligned BF16 pairs"))
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
struct PendingLoad{N,L}
    words::NTuple{N,UInt32}
    ownership::L
end

@inline function _check_tmem_store(p::TmemPartition,f)
    isequal(_canonical_ownership(Layouts.layout(p)),_canonical_ownership(Layouts.layout(f))) ||
        throw(DimensionMismatch("TMEM store ownership differs from transfer"))
    nothing
end
