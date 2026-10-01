# Origins are logical; the descriptor's inner dimension is the plan's axis
# A. A binding of rank 3 to 5 adds the matrix's batch coordinates. A tile
# of several stripes is one copy per stripe: the next W bytes of the inner
# axis, into the next stripe of storage.
# Stripe s: `s*W*outer` bytes into storage, `s*W÷sizeof(T)` along the inner axis.
_stripe_bytes(::TMATile{T,S,A,W}) where {T,S,A,W} = W*S[3-A]
_stripe_extent(::TMATile{T,S,A,W}) where {T,S,A,W} = Int32(W ÷ sizeof(T))
# With `cta`, the copy completes on the barrier at the same address in the
# CTA of that rank, this CTA or its pair peer (`.cta_group::2`).
Base.@propagate_inbounds function tma_load!(dst::SharedTile{T,L},
        b::TMABinding{TMATile{T,S,A,W},D,2},origin::Tuple{Int32,Int32},barrier::Barrier;cta::Union{Nothing,Integer}=nothing) where {T,L,S,A,W,D}
    _check_tma(dst,b,origin)
    if cta === nothing
        PTX.Utils.@unroll 8 for s in 0:_stripes(b.plan)-1
            bytes, inner = s*_stripe_bytes(b.plan), Int32(s)*_stripe_extent(b.plan)
            ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                reinterpret(Core.LLVMPtr{UInt8,3},dst.ptr)+bytes,b.descriptor,origin[A]+inner,origin[3-A],barrier.ptr)
        end
    else
        remote = reinterpret(Core.LLVMPtr{UInt8,3},UInt64(ptx"mapa.shared::cluster.u32"(_smem_address(barrier.ptr),cta % UInt32)))
        PTX.Utils.@unroll 8 for s in 0:_stripes(b.plan)-1
            bytes, inner = s*_stripe_bytes(b.plan), Int32(s)*_stripe_extent(b.plan)
            ptx"cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"(
                reinterpret(Core.LLVMPtr{UInt8,3},dst.ptr)+bytes,b.descriptor,origin[A]+inner,origin[3-A],remote)
        end
    end
    nothing
end
# Multicast: the box lands at the same shared address in every CTA of the
# cluster whose bit is set in `ctas`, and completes on the barrier at the
# same address in each.
Base.@propagate_inbounds function tma_load!(dst::SharedTile{T,L},
        b::TMABinding{TMATile{T,S,A,W},D,2},origin::Tuple{Int32,Int32},barrier::Barrier,ctas::UInt16) where {T,L,S,A,W,D}
    _check_tma(dst,b,origin)
    PTX.Utils.@unroll 8 for s in 0:_stripes(b.plan)-1
        bytes, inner = s*_stripe_bytes(b.plan), Int32(s)*_stripe_extent(b.plan)
        ptx"cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
            reinterpret(Core.LLVMPtr{UInt8,3},dst.ptr)+bytes,b.descriptor,origin[A]+inner,origin[3-A],barrier.ptr,ctas)
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
# Ranks 3 to 5: matrices along one to three batch axes, whose coordinates
# follow the matrix's two.
_ptx_op(spelling) = Expr(:macrocall,Symbol("@ptx_str"),LineNumberNode(@__LINE__,Symbol(@__FILE__)),spelling)
for R in 3:5
    local_load = _ptx_op("cp.async.bulk.tensor.$(R)d.shared::cta.global.tile.mbarrier::complete_tx::bytes")
    pair_load = _ptx_op("cp.async.bulk.tensor.$(R)d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes")
    store = _ptx_op("cp.async.bulk.tensor.$(R)d.global.shared::cta.tile.bulk_group")
    @eval begin
        Base.@propagate_inbounds function tma_load!(dst::SharedTile{T,L},
                b::TMABinding{TMATile{T,S,A,W},D,$R},origin::NTuple{$R,Int32},barrier::Barrier;cta::Union{Nothing,Integer}=nothing) where {T,L,S,A,W,D}
            _check_tma(dst,b,origin)
            batch = Base.tail(Base.tail(origin))
            if cta === nothing
                PTX.Utils.@unroll 8 for s in 0:_stripes(b.plan)-1
                    bytes, inner = s*_stripe_bytes(b.plan), Int32(s)*_stripe_extent(b.plan)
                    $local_load(reinterpret(Core.LLVMPtr{UInt8,3},dst.ptr)+bytes,b.descriptor,origin[A]+inner,origin[3-A],batch...,barrier.ptr)
                end
            else
                remote = reinterpret(Core.LLVMPtr{UInt8,3},UInt64(ptx"mapa.shared::cluster.u32"(_smem_address(barrier.ptr),cta % UInt32)))
                PTX.Utils.@unroll 8 for s in 0:_stripes(b.plan)-1
                    bytes, inner = s*_stripe_bytes(b.plan), Int32(s)*_stripe_extent(b.plan)
                    $pair_load(reinterpret(Core.LLVMPtr{UInt8,3},dst.ptr)+bytes,b.descriptor,origin[A]+inner,origin[3-A],batch...,remote)
                end
            end
            nothing
        end
        Base.@propagate_inbounds function tma_store!(b::TMABinding{TMATile{T,S,A,W},D,$R},src::SharedTile{T,L},
                origin::NTuple{$R,Int32}) where {T,L,S,A,W,D}
            _check_tma(src,b,origin)
            @boundscheck all(>=(Int32(0)),origin) || throw(ArgumentError("store origins are non-negative"))
            batch = Base.tail(Base.tail(origin))
            PTX.Utils.@unroll 8 for s in 0:_stripes(b.plan)-1
                bytes, inner = s*_stripe_bytes(b.plan), Int32(s)*_stripe_extent(b.plan)
                $store(b.descriptor,origin[A]+inner,origin[3-A],batch...,reinterpret(Core.LLVMPtr{UInt8,3},src.ptr)+bytes)
            end
            nothing
        end
    end
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
# Bulk copies of plain bytes.
@inline bulk_copy!(dst::Core.LLVMPtr{<:Any,1},src::Core.LLVMPtr{<:Any,3},bytes::Integer) =
    (ptx"cp.async.bulk.global.shared::cta.bulk_group"(dst,src,bytes % UInt32); nothing)
@inline bulk_copy!(dst::Core.LLVMPtr{<:Any,3},src::Core.LLVMPtr{<:Any,1},bytes::Integer,b::Barrier) =
    (ptx"cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"(dst,src,bytes % UInt32,b.ptr); nothing)
@inline function bulk_copy!(dst::Core.LLVMPtr{<:Any,3},src::Core.LLVMPtr{<:Any,3},bytes::Integer,b::Barrier;cta::Integer)
    ptx"cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"(
        reinterpret(Core.LLVMPtr{UInt8,3},_remote(dst,cta)),src,bytes % UInt32,_remote(b.ptr,cta))
    nothing
end
Base.@constprop :aggressive @inline wait_tma_reads(n::Integer) = ptx"cp.async.bulk.wait_group.read"(Val(Int(n)))
Base.@constprop :aggressive @inline wait_tma_stores(n::Integer) = ptx"cp.async.bulk.wait_group"(Val(Int(n)))
# Origins of any integer type, as the Int32 coordinates TMA takes.
Base.@propagate_inbounds tma_load!(dst::SharedTile,b::TMABinding{P,D,R},origin::NTuple{R,Integer},barrier::Barrier;cta::Union{Nothing,Integer}=nothing) where {P,D,R} =
    tma_load!(dst,b,map(x -> x % Int32,origin),barrier;cta)
Base.@propagate_inbounds tma_store!(b::TMABinding{P,D,R},src::SharedTile,origin::NTuple{R,Integer}) where {P,D,R} =
    tma_store!(b,src,map(x -> x % Int32,origin))
