# Origins are logical; the descriptor's inner dimension is the plan's axis
# A. A rank-3 binding adds the matrix index as its third coordinate. A tile
# of several stripes is one copy per stripe: the next W bytes of the inner
# axis, into the next stripe of storage.
# Stripe s: `s*W*outer` bytes into storage, `s*W÷sizeof(T)` along the inner axis.
_stripe_bytes(::TMATile{T,S,A,W}) where {T,S,A,W} = W*S[3-A]
_stripe_extent(::TMATile{T,S,A,W}) where {T,S,A,W} = Int32(W ÷ sizeof(T))
Base.@propagate_inbounds function tma_load!(dst::SharedTile{T,L},
        b::TMABinding{TMATile{T,S,A,W},D,2},origin::Tuple{Int32,Int32},barrier::Barrier) where {T,L,S,A,W,D}
    _check_tma(dst,b,origin)
    PTX.Utils.@unroll 8 for s in 0:_stripes(b.plan)-1
        bytes, inner = s*_stripe_bytes(b.plan), Int32(s)*_stripe_extent(b.plan)
        ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            reinterpret(Core.LLVMPtr{UInt8,3},dst.ptr)+bytes,b.descriptor,origin[A]+inner,origin[3-A],barrier.ptr)
    end
    nothing
end
Base.@propagate_inbounds function tma_load!(dst::SharedTile{T,L},
        b::TMABinding{TMATile{T,S,A,W},D,3},origin::Tuple{Int32,Int32,Int32},barrier::Barrier) where {T,L,S,A,W,D}
    _check_tma(dst,b,origin)
    PTX.Utils.@unroll 8 for s in 0:_stripes(b.plan)-1
        bytes, inner = s*_stripe_bytes(b.plan), Int32(s)*_stripe_extent(b.plan)
        ptx"cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            reinterpret(Core.LLVMPtr{UInt8,3},dst.ptr)+bytes,b.descriptor,origin[A]+inner,origin[3-A],origin[3],barrier.ptr)
    end
    nothing
end
Base.@propagate_inbounds function tma_store!(b::TMABinding{TMATile{T,S,A,W},D,2},src::SharedTile{T,L},
        origin::Tuple{Int32,Int32}) where {T,L,S,A,W,D}
    _check_tma(src,b,origin)
    @boundscheck origin[1] >= 0 && origin[2] >= 0 || throw(ArgumentError("store origins are non-negative"))
    PTX.Utils.@unroll 8 for s in 0:_stripes(b.plan)-1
        bytes, inner = s*_stripe_bytes(b.plan), Int32(s)*_stripe_extent(b.plan)
        ptx"cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"(
            b.descriptor,origin[A]+inner,origin[3-A],reinterpret(Core.LLVMPtr{UInt8,3},src.ptr)+bytes)
    end
    nothing
end
Base.@propagate_inbounds function tma_store!(b::TMABinding{TMATile{T,S,A,W},D,3},src::SharedTile{T,L},
        origin::Tuple{Int32,Int32,Int32}) where {T,L,S,A,W,D}
    _check_tma(src,b,origin)
    @boundscheck origin[1] >= 0 && origin[2] >= 0 && origin[3] >= 0 || throw(ArgumentError("store origins are non-negative"))
    PTX.Utils.@unroll 8 for s in 0:_stripes(b.plan)-1
        bytes, inner = s*_stripe_bytes(b.plan), Int32(s)*_stripe_extent(b.plan)
        ptx"cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"(
            b.descriptor,origin[A]+inner,origin[3-A],origin[3],reinterpret(Core.LLVMPtr{UInt8,3},src.ptr)+bytes)
    end
    nothing
end
Base.@propagate_inbounds function _check_tma(t::SharedTile{T,L},b::TMABinding{TMATile{T,S,A,W}},origin) where {T,L,S,A,W}
    L === typeof(shared_layout(b.plan)) || throw(ArgumentError("the shared layout is not the plan's canonical storage"))
    @boundscheck begin
        _smem_address(t.ptr) % UInt32(8W) == 0 || throw(ArgumentError("TMA storage alignment"))
        (origin[A]*Int32(sizeof(T))) % Int32(16) == 0 || throw(ArgumentError("the inner origin must start a 16-byte chunk"))
    end
    nothing
end
@inline commit_tma_stores() = ptx"cp.async.bulk.commit_group"()
Base.@constprop :aggressive @inline wait_tma_reads(n::Integer) = ptx"cp.async.bulk.wait_group.read"(Val(Int(n)))
Base.@constprop :aggressive @inline wait_tma_stores(n::Integer) = ptx"cp.async.bulk.wait_group"(Val(Int(n)))
# Origins of any integer type, as the Int32 coordinates TMA takes.
Base.@propagate_inbounds tma_load!(dst::SharedTile,b::TMABinding,origin::Tuple{Integer,Integer},barrier::Barrier) =
    tma_load!(dst,b,map(x -> x % Int32,origin),barrier)
Base.@propagate_inbounds tma_load!(dst::SharedTile,b::TMABinding,origin::Tuple{Integer,Integer,Integer},barrier::Barrier) =
    tma_load!(dst,b,map(x -> x % Int32,origin),barrier)
Base.@propagate_inbounds tma_store!(b::TMABinding,src::SharedTile,origin::Tuple{Integer,Integer}) =
    tma_store!(b,src,map(x -> x % Int32,origin))
Base.@propagate_inbounds tma_store!(b::TMABinding,src::SharedTile,origin::Tuple{Integer,Integer,Integer}) =
    tma_store!(b,src,map(x -> x % Int32,origin))
