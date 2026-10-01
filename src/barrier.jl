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
    arrive!(barrier) -> Arrival
    arrive!(barrier, bytes) -> Arrival
    arrive!(barrier; cta)

Arrive on the barrier's current phase. With `bytes`, first announce that
many bytes of asynchronous transactions (a TMA load's
[`transfer_bytes`](@ref)) that must also land before the phase completes.
With `cta`, arrive on the barrier at the same address in the cluster's CTA
of that rank ([`cluster_rank`](@ref)): a consumer releasing a stage that a
peer's multicast loads write. A local arrival returns its [`Arrival`](@ref),
which [`after`](@ref) orders later work behind.
"""
function arrive! end

"""
    Arrival

A local arrival on a [`Barrier`](@ref): the barrier's state as the arrival
left it. Its use is [`after`](@ref).
"""
struct Arrival
    state::UInt64
end

"""
    after(arrival::Arrival, x)

`x` (a number), as a value that the compilers take to depend on `arrival`:
work computed from it cannot move above the arrival. Nothing else orders
arithmetic after an arrival, which is a store as far as the compilers know:
a role that publishes part of its results and then computes the rest would
otherwise find the rest computed first. Apply it to the one value the later
work starts from (a running maximum, a zero to sum onto); it costs a
comparison and a selection.

```julia
done = arrive!(published)                 # the first chunks are visible
rest = exp2.(scores .- after(done, m))    # computed only after the arrival
```
"""
@inline after(a::Arrival,x::Number) = ifelse(a.state == typemax(UInt64),x+one(x),x)

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

"""
    NamedBarrier

One of the CTA's 16 hardware barriers (`bar.sync`/`bar.arrive`), for
`threads` threads (a multiple of 32): [`wait`](@ref)`(b)` arrives and waits
until `threads` threads have arrived, [`arrive!`](@ref)`(b)` arrives without
waiting. Barrier 0 is `sync_threads`'. Declare a kernel's named barriers
together with [`named_barriers`](@ref), which numbers them.

Two arrivals in a row from one side can complete a barrier on their own: a
barrier paces one handoff per phase, as an [`Barrier`](@ref)'s phases do
not need to.
"""
struct NamedBarrier
    id::Int32
    threads::Int32
end

"""
    named(count=1; threads)

A group of `count` named barriers of `threads` threads each, in
[`named_barriers`](@ref).
"""
struct NamedGroup
    count::Int
    threads::Int
end
Base.@constprop :aggressive @inline function named(count::Integer=1;threads::Integer)
    threads % 32 == 0 && threads > 0 || throw(ArgumentError("a named barrier synchronizes whole warps"))
    NamedGroup(Int(count),Int(threads))
end

"""
    named_barriers(groups::NamedTuple; first=1) -> NamedBarriers

Number a kernel's named barriers: each group of [`named`](@ref) barriers in
order, from `first` (1 by default, after `sync_threads`' 0; a megakernel
that reserves more starts later). `nb.name` is the group's single
[`NamedBarrier`](@ref), or with a count above one, `nb.name[i]` (zero-based)
is the `i`-th.

```julia
nb = named_barriers((; handoff = named(8; threads=64), merge = named(; threads=128)))
arrive!(nb.handoff[i]); wait(nb.handoff[i]); wait(nb.merge)
```
"""
struct NamedBarriers{G<:NamedTuple}
    groups::G
    first::Int
end
Base.@constprop :aggressive @inline function named_barriers(groups::NamedTuple;first::Integer=1)
    first >= 1 && first+sum(g -> g.count,values(groups);init=0) <= 16 ||
        throw(ArgumentError("named barriers are numbered 1 to 15"))
    NamedBarriers(groups,Int(first))
end
"A group of consecutive named barriers: `g[i]`, zero-based."
struct NamedBarrierRange
    first::Int32
    count::Int32
    threads::Int32
end
@inline function Base.getindex(g::NamedBarrierRange,i::Integer)
    @boundscheck 0 <= i < g.count || throw(BoundsError())
    NamedBarrier(g.first+(i % Int32),g.threads)
end
@inline Base.length(g::NamedBarrierRange) = Int(g.count)
@inline function Base.getproperty(nb::NamedBarriers{G},name::Symbol) where G
    groups = getfield(nb,:groups)
    i = Base.fieldindex(G,name)
    id = getfield(nb,:first)+sum(j -> groups[j].count,1:i-1;init=0)
    g = groups[i]
    g.count == 1 ? NamedBarrier(Int32(id),Int32(g.threads)) : NamedBarrierRange(Int32(id),Int32(g.count),Int32(g.threads))
end
Base.propertynames(nb::NamedBarriers) = keys(getfield(nb,:groups))

"""
    wait(b::NamedBarrier; aligned=true)
    arrive!(b::NamedBarrier; aligned=true)

Arrive on the named barrier and wait until its `threads` threads have
arrived (`bar.sync`), or arrive without waiting (`bar.arrive`). The
calling warps' lanes take part together; with `aligned = false` they may
arrive apart (`barrier.sync`, `barrier.arrive`).
"""
Base.wait(::NamedBarrier)
