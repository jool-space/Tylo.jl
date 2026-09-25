@inline initialize!(b::Barrier,arrivals::Integer) = (ptx"mbarrier.init.shared.b64"(b.ptr,arrivals % UInt32); nothing)
# Locally, or (`cta`) at the same address in the cluster's CTA of that rank,
# which mapa translates.
@inline function arrive!(b::Barrier;cta::Union{Nothing,Integer}=nothing)
    if cta === nothing
        ptx"mbarrier.arrive.shared.b64"(b.ptr)
    else
        remote = ptx"mapa.shared::cluster.u32"(_smem_address(b.ptr),cta % UInt32)
        ptx"mbarrier.arrive.shared::cluster.b64"(reinterpret(Core.LLVMPtr{UInt64,3},UInt64(remote)))
    end
    nothing
end
@inline arrive!(b::Barrier,bytes::Integer) = (ptx"mbarrier.arrive.expect_tx.shared.b64"(b.ptr,bytes % UInt32); nothing)
@inline expect_bytes!(b::Barrier,bytes::Integer) = (ptx"mbarrier.expect_tx.shared.b64"(b.ptr,bytes % UInt32); nothing)
@inline function Base.wait(b::Barrier,parity::Integer)
    p = (parity % UInt32) & UInt32(1)
    while !ptx"mbarrier.try_wait.parity.shared.b64"(b.ptr,p) end
    nothing
end
@inline invalidate!(b::Barrier) = (ptx"mbarrier.inval.shared.b64"(b.ptr); nothing)
@inline cluster_rank() = ptx"mov.u32"(PTX.@sreg_str("%cluster_ctarank")) % Int32
@inline cluster_sync() = (ptx"barrier.cluster.arrive.aligned"(); ptx"barrier.cluster.wait.aligned"(); nothing)
@inline fence_barrier_init() = ptx"fence.mbarrier_init.release.cluster"()
@inline fence_async_shared() = ptx"fence.proxy.async.shared::cta"()
# The thread's prior cp.async copies arrive on the barrier as they complete:
# one of its expected arrivals.
@inline commit_copies(b::Barrier) = (ptx"cp.async.mbarrier.arrive.noinc.shared.b64"(b.ptr); nothing)
