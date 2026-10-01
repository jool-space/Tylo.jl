"""
    Barrier(pointer)
    Barrier(pointer, i)
    Barrier(address::UInt32)

An mbarrier: 8 bytes of shared memory, 8-byte aligned, at a shared pointer
or a 32-bit shared address; `Barrier(pointer, i)` is the zero-based `i`-th
of an array of them, one per pipeline stage, say. It counts arrivals and asynchronous
transaction bytes; a phase completes when both reach zero, and its parity
flips. TMA loads ([`tma_load!`](@ref)), tcgen05 MMAs ([`commit_mma`](@ref))
and `cp.async` copies ([`commit_copies`](@ref)) complete on barriers.

```julia
bar = Barrier(pointer(smem))               # 8 bytes of shared memory
thread == 0 && initialize!(bar, 1)         # one arrival per phase
fence_barrier_init(); sync_threads()       # visible to TMA, then to every thread
thread == 0 && (arrive!(bar, transfer_bytes(binding)); tma_load!(tile, binding, origin, bar))
wait(bar, phase)                           # phase 0 first, then 1, 0, ...
```

A barrier is a view: the kernel allocates it, initializes it once, and
invalidates it before the memory is reused.
"""
struct Barrier
    ptr::Core.LLVMPtr{UInt64,3}
end
@inline Barrier(p::Core.LLVMPtr{T,3}) where T = Barrier(reinterpret(Core.LLVMPtr{UInt64,3},p))
@inline Barrier(p::Core.LLVMPtr{T,3},i::Integer) where T = Barrier(reinterpret(Core.LLVMPtr{UInt64,3},p) + 8i)
@inline Barrier(address::UInt32) = Barrier(reinterpret(Core.LLVMPtr{UInt64,3},UInt64(address)))

"""
    initialize!(barrier, arrivals)

Initialize a barrier whose phases complete after `arrivals` arrivals (and
the transaction bytes they announce). One thread initializes; make it
visible with [`fence_barrier_init`](@ref) and a thread barrier before use.
"""
function initialize! end

"""
    arrive!(barrier)
    arrive!(barrier, bytes)
    arrive!(barrier; cta)

Arrive on the barrier's current phase. With `bytes`, first announce that
many bytes of asynchronous transactions (a TMA load's
[`transfer_bytes`](@ref)) that must also land before the phase completes.
With `cta`, arrive on the barrier at the same address in the cluster's CTA
of that rank ([`cluster_rank`](@ref)): a consumer releasing a stage that a
peer's multicast loads write.
"""
function arrive! end

"""
    expect_bytes!(barrier, bytes)

Announce `bytes` more of asynchronous transactions on the current phase,
without arriving.
"""
function expect_bytes! end

"""
    wait(barrier, parity)

Block until the barrier's phase of the given parity (0 or 1) completes. A
kernel waiting on successive phases alternates the parity: 0, 1, 0, ...
With a [`wait_timeout`](@ref), a wait that outlasts it reports the barrier
and traps.
"""
Base.wait(::Barrier,::Integer)

"""
    wait_timeout() -> Union{Nothing,Integer}

The nanoseconds a barrier [`wait`](@ref) may spin before it prints the
barrier's shared address, the parity it waited for and the thread, and
traps: a hang becomes a launch error. `nothing` (the default) waits without
limit and costs nothing. Redefine it to find a hang, then recompile:

```julia
Tylo.wait_timeout() = 1_000_000_000    # one second
```
"""
wait_timeout() = nothing
# Reports a wait that timed out (printing needs CUDACore, which overrides it).
_report_hang(::Barrier,parity) = nothing

"""
    invalidate!(barrier)

Invalidate an initialized barrier before its memory is used for anything
else.
"""
function invalidate! end

"""
    fence_barrier_init()

Make barrier initializations visible to the asynchronous proxy (TMA,
tcgen05), before the thread barrier that publishes them to other threads.
"""
function fence_barrier_init end

"""
    fence_async_shared()

Order this thread's preceding shared-memory writes before subsequent
asynchronous-proxy operations: before a `tma_store!` or an MMA reads data
that ordinary stores wrote (followed by a thread barrier when other
threads wrote it).
"""
function fence_async_shared end

"""
    cluster_rank()

The calling CTA's rank within its cluster (`%cluster_ctarank`), from 0: the
bit of it in a multicast mask, and the rank that `arrive!(barrier, rank)`
names.
"""
function cluster_rank end

"""
    cluster_sync()

All threads of every CTA in the cluster (`barrier.cluster`): after it, the
cluster's barriers are initialized (with [`fence_barrier_init`](@ref)
before), and before a CTA exits, no peer still writes into its shared
memory or arrives on its barriers.
"""
function cluster_sync end
