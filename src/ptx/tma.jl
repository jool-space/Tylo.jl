# Origins are logical; the descriptor's inner dimension is the plan's axis
# A. A rank-3 binding adds the matrix index as its third coordinate.
Base.@propagate_inbounds function tma_load!(dst::SharedTile{T,L},
        b::TMABinding{TMATile{T,S,A,W},D,2},origin::Tuple{Int32,Int32},barrier::Core.LLVMPtr{UInt64,3}) where {T,L,S,A,W,D}
    _check_tma(dst,b,origin)
    ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
        reinterpret(Core.LLVMPtr{UInt8,3},dst.ptr),b.descriptor,origin[A],origin[3-A],barrier)
    nothing
end
Base.@propagate_inbounds function tma_load!(dst::SharedTile{T,L},
        b::TMABinding{TMATile{T,S,A,W},D,3},origin::Tuple{Int32,Int32,Int32},barrier::Core.LLVMPtr{UInt64,3}) where {T,L,S,A,W,D}
    _check_tma(dst,b,origin)
    ptx"cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
        reinterpret(Core.LLVMPtr{UInt8,3},dst.ptr),b.descriptor,origin[A],origin[3-A],origin[3],barrier)
    nothing
end
Base.@propagate_inbounds function tma_store!(b::TMABinding{TMATile{T,S,A,W},D,2},src::SharedTile{T,L},
        origin::Tuple{Int32,Int32}) where {T,L,S,A,W,D}
    _check_tma(src,b,origin)
    @boundscheck origin[1] >= 0 && origin[2] >= 0 || throw(ArgumentError("store origins are non-negative"))
    ptx"cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"(
        b.descriptor,origin[A],origin[3-A],reinterpret(Core.LLVMPtr{UInt8,3},src.ptr))
    nothing
end
Base.@propagate_inbounds function tma_store!(b::TMABinding{TMATile{T,S,A,W},D,3},src::SharedTile{T,L},
        origin::Tuple{Int32,Int32,Int32}) where {T,L,S,A,W,D}
    _check_tma(src,b,origin)
    @boundscheck origin[1] >= 0 && origin[2] >= 0 && origin[3] >= 0 || throw(ArgumentError("store origins are non-negative"))
    ptx"cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"(
        b.descriptor,origin[A],origin[3-A],origin[3],reinterpret(Core.LLVMPtr{UInt8,3},src.ptr))
    nothing
end
Base.@propagate_inbounds function _check_tma(t::SharedTile{T,L},b::TMABinding{TMATile{T,S,A,W}},origin) where {T,L,S,A,W}
    L === typeof(shared_layout(b.plan)) || throw(ArgumentError("the shared layout is not the plan's canonical storage"))
    @boundscheck begin
        PTX.smem_addr_u32(t.ptr) % UInt32(8W) == 0 || throw(ArgumentError("TMA storage alignment"))
        (origin[A]*Int32(sizeof(T))) % Int32(16) == 0 || throw(ArgumentError("the inner origin must start a 16-byte chunk"))
    end
    nothing
end
@inline commit_tma_stores() = ptx"cp.async.bulk.commit_group"()
@inline wait_tma_reads(::Val{N}) where N = ptx"cp.async.bulk.wait_group.read"(Val(N))
@inline wait_tma_stores(::Val{N}) where N = ptx"cp.async.bulk.wait_group"(Val(N))
