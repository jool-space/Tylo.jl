@inline initialize!(b::Barrier,arrivals::Integer) = (ptx"mbarrier.init.shared.b64"(b.ptr,arrivals % UInt32); nothing)
# Locally, or (`cta`) at the same address in the cluster's CTA of that rank,
# which mapa translates.
@inline function arrive!(b::Barrier;cta::Union{Nothing,Integer}=nothing,scope::Symbol=:cta)
    if cta === nothing
        scope === :cta || throw(ArgumentError("a local arrival has CTA scope"))
        Arrival(ptx"mbarrier.arrive.shared.b64"(b.ptr))
    elseif scope === :cta
        ptx"mbarrier.arrive.shared::cluster.b64"(_remote(b.ptr,cta))
        nothing
    else
        scope === :cluster || throw(ArgumentError("scope is :cta or :cluster"))
        ptx"mbarrier.arrive.release.cluster.shared::cluster.b64"(_remote(b.ptr,cta))
        nothing
    end
end
# The same shared address in the cluster's CTA of rank `cta`.
@inline _remote(p::Core.LLVMPtr{T,3},cta::Integer) where T =
    reinterpret(Core.LLVMPtr{T,3},UInt64(ptx"mapa.shared::cluster.u32"(_smem_address(p),cta % UInt32)))
@inline arrive!(b::Barrier,bytes::Integer) = Arrival(ptx"mbarrier.arrive.expect_tx.shared.b64"(b.ptr,bytes % UInt32))
@inline expect_bytes!(b::Barrier,bytes::Integer) = (ptx"mbarrier.expect_tx.shared.b64"(b.ptr,bytes % UInt32); nothing)
@inline function Base.wait(b::Barrier,parity::Integer;scope::Symbol=:cta)
    p = (parity % UInt32) & UInt32(1)
    scope === :cta && return _wait(b,p,wait_timeout())
    scope === :cluster || throw(ArgumentError("scope is :cta or :cluster"))
    while !ptx"mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"(b.ptr,p) end
    nothing
end
@inline function _wait(b::Barrier,p::UInt32,::Nothing)
    while !ptx"mbarrier.try_wait.parity.shared.b64"(b.ptr,p) end
    nothing
end
@inline function _wait(b::Barrier,p::UInt32,limit::Integer)
    start = ptx"mov.u64"(PTX.@sreg_str("%globaltimer"))
    while !ptx"mbarrier.try_wait.parity.shared.b64"(b.ptr,p)
        if ptx"mov.u64"(PTX.@sreg_str("%globaltimer"))-start > limit % UInt64
            _report_hang(b,p)
            ptx"trap"()
        end
    end
    nothing
end
@inline invalidate!(b::Barrier) = (ptx"mbarrier.inval.shared.b64"(b.ptr); nothing)
@inline cluster_rank() = ptx"mov.u32"(PTX.@sreg_str("%cluster_ctarank")) % Int32
@inline cluster_sync() = (ptx"barrier.cluster.arrive.aligned"(); ptx"barrier.cluster.wait.aligned"(); nothing)
@inline fence_barrier_init() = ptx"fence.mbarrier_init.release.cluster"()
@inline fence_async_shared() = ptx"fence.proxy.async.shared::cta"()
@inline fence_async_global() = ptx"fence.proxy.async.global"()
# The thread's prior cp.async copies arrive on the barrier as they complete:
# one of its expected arrivals.
@inline commit_copies(b::Barrier) = (ptx"cp.async.mbarrier.arrive.noinc.shared.b64"(b.ptr); nothing)
@inline function Base.wait(b::NamedBarrier;aligned::Bool=true)
    aligned ? ptx"bar.sync"(b.id % UInt32,b.threads % UInt32) : ptx"barrier.sync"(b.id % UInt32,b.threads % UInt32)
    nothing
end
@inline function arrive!(b::NamedBarrier;aligned::Bool=true)
    aligned ? ptx"bar.arrive"(b.id % UInt32,b.threads % UInt32) : ptx"barrier.arrive"(b.id % UInt32,b.threads % UInt32)
    nothing
end
