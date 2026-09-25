@inline initialize!(b::Barrier,arrivals::Integer) = (ptx"mbarrier.init.shared.b64"(b.ptr,arrivals % UInt32); nothing)
@inline arrive!(b::Barrier) = (ptx"mbarrier.arrive.shared.b64"(b.ptr); nothing)
@inline arrive!(b::Barrier,bytes::Integer) = (ptx"mbarrier.arrive.expect_tx.shared.b64"(b.ptr,bytes % UInt32); nothing)
@inline expect_bytes!(b::Barrier,bytes::Integer) = (ptx"mbarrier.expect_tx.shared.b64"(b.ptr,bytes % UInt32); nothing)
@inline function Base.wait(b::Barrier,parity::Integer)
    p = (parity % UInt32) & UInt32(1)
    while !ptx"mbarrier.try_wait.parity.shared.b64"(b.ptr,p) end
    nothing
end
@inline invalidate!(b::Barrier) = (ptx"mbarrier.inval.shared.b64"(b.ptr); nothing)
@inline fence_barrier_init() = ptx"fence.mbarrier_init.release.cluster"()
@inline fence_async_shared() = ptx"fence.proxy.async.shared::cta"()
# The thread's prior cp.async copies arrive on the barrier as they complete:
# one of its expected arrivals.
@inline commit_copies(b::Barrier) = (ptx"cp.async.mbarrier.arrive.noinc.shared.b64"(b.ptr); nothing)
