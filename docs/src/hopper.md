```@meta
CurrentModule = Tylo
DocTestSetup = :(using Tylo)
```

# TMA and Hopper WGMMA

The Hopper path uses a different matrix instruction from the warp-MMA GEMM: TMA brings a tile
from global memory into shared memory, and a 128-thread warpgroup consumes
shared descriptors with WGMMA. Megakernels' `HopperProjection` uses these
operations for GEMM and fused gate/up projection.

## The same storage contract on both sides

```jldoctest
julia> a = TMATile(Tylo.BFloat16, Val((128,64)), Val(2));

julia> transfer_bytes(a)
16384

julia> shared_layout(a)((Int32(1), Int32(0)))
72

julia> b = TMATile(Float32, Val((16,64)), Val(1), Val(64));

julia> swizzle_bytes(b), size(b)
(64, (16, 64))
```

The first plan is logical A(M,K) with K contiguous before swizzling; logical
B(K,N) uses `TMATile(T, Val((k,n)), Val(1))`. The host arrays passed to
`prepare_tma` are physically `(inner, outer)` Julia matrices in both cases:
transpose A's storage once; B already has this storage order.

A plan names an element type of one, two or four bytes, the tile shape, the
inner axis and the swizzle row width: 128 bytes by default, or 64 or 32.
The inner extent is one row of elements; the outer extent is a multiple of
eight, at most 256. `shared_layout(plan)` maps logical coordinates to
**element offsets**, just like Tylo's other layouts: `Swizzle{B,M,3}` over
packed rows, where `M` keeps the 16-byte chunk bits and `2^B` chunks form a
row. For 128-byte rows of 16-bit elements and row r, the mapping is
`64r + xor(k,8*(r%8))`. An allocation aligned to eight rows (1024 bytes for
128-byte rows) establishes the swizzle origin. `shared_tile(plan, pointer)`
borrows that allocation. `TMALoad(T, Val(S), Val(A))` is the earlier name of
the 128-byte plan.

`prepare_tma(plan, device_array)` returns an owner of the uploaded descriptor
and the source tensor. Adaptation produces an isbits device binding. Keep the
host binding alive through launches and graph replays; changing source values
is allowed, changing the address or dimensions requires a new descriptor.
PTX.jl performs the driver encoding and descriptor upload. One binding
serves loads and stores. A three-dimensional array binds a batch of
matrices: origins gain a third coordinate selecting one, each bounded
separately, and `bounds` names logical extents smaller than the array, as
the streaming attention example does for heads and padded V.

A descriptor is not available for an arbitrary composed layout.
[`swizzled_structure`](@ref) recognizes the canonical encodings
structurally, and it is the same recognition `ldmatrix` copies,
`wgmma_operand` and `tcgen05_operand` rely on: an operand descriptor is a
pure function of the storage encoding and the origin. A general `Layout`
can describe many maps for which no such descriptor exists.

## Copy and completion

```julia
# Host, with CUDACore and PTX loaded:
a_map = prepare_tma(a_plan, a_device)

# Device, in one elected producer thread:
# Initialize/publish the barrier and arrive with the aggregate transaction
# count first, using the caller's barrier protocol.
tma_load!(shared_tile(a_map, stage_pointer), a_map, (row, k), barrier)
```

Origins are zero-based logical coordinates; the inner origin starts a
16-byte chunk. Out-of-bounds source elements are zero-filled by TMA. The
full tile byte count still contributes to the mbarrier's expected
transaction count. Tylo does not insert an arrival or a wait: a producer
may put A, B and another operand behind the same barrier.

The caller publishes barrier initialization with
`fence.proxy.async.shared::cta`, establishes visibility for preceding generic
writes as required, and waits for the matching barrier phase before consumption.
All consumers must finish before the producer overwrites the stage. These are
separate obligations; a completed TMA load does not mean WGMMA has stopped
reading that storage.

Stores go the other way through the same binding and storage:

```julia
# Every writing thread, after filling the shared tile through its layout:
ptx"fence.proxy.async.shared::cta"()
sync_threads()
# One elected thread:
tma_store!(c_map, shared_tile(c_map, stage_pointer), (row, col))
commit_tma_stores()
wait_tma_reads(Val(0))   # the shared stage may be refilled
wait_tma_stores(Val(0))  # the global writes are complete
```

Store origins are non-negative; elements past the array's far edges are
not written. Stores complete through the issuing thread's bulk groups, not
through an mbarrier: `commit_tma_stores` closes a group, `wait_tma_reads`
waits until the shared sources of all but `N` groups may be reused, and
`wait_tma_stores` until their global writes are complete. The proxy fence
and the rendezvous between writers and the issuing thread are the caller's.

## WGMMA values and ownership

```julia
plan = WGMMA64(BFloat16, Val(8), Val(64), Val(4))
acc = zero_accumulator(plan)
a = wgmma_operand(plan, OperandA(), a_shared, (row, Int32(0)))
b = wgmma_operand(plan, OperandB(), b_shared)
pending = mma_async(plan, a, b, acc)
acc = wait_mma(pending)
result = finish_mma(acc)
```

All 128 threads execute this collectively. The plan supports M=64,
N=8:8:256, K=16/32/64, and BF16/FP16 inputs with FP32 accumulators.
Each K=16 instruction updates one accumulator partition. The default has
one partition; N=8,K=64 with four partitions preserves the four independent
K chains used by Cohere's n8 projection. `finish_mma` sums those partitions.
The total register payload is limited to 128 FP32 registers per thread.

`wgmma_operand` checks role, logical geometry and alignment. It accepts only
canonical TMA shared layouts, with non-K origins divisible by eight and K
origins divisible by 16. The whole instruction plan must fit.
`validate_wgmma` checks geometry on the host. `@inbounds` may remove dynamic
bounds/alignment checks when the caller has established those facts.
A descriptor carries its operand role and plan in its type.

`mma_async` fences the accumulator registers, issues the plan and commits one
WGMMA group. `wait_mma` waits for **all** prior committed groups of that
warpgroup. It carries register dependencies through inline assembly so the
compiler cannot move arithmetic on the results above the wait. Pending
accumulators have no arithmetic or store methods. Julia types do not enforce
linear ownership, thread convergence, storage lifetime, or barrier correctness.

`finish_mma` returns an ordinary `Fragment` with WGMMA ownership. Its ownership
map is an ISA property, independent of storage.
It supports fused broadcast, type conversion, and `sum`/`maximum`/`minimum`
along axis 2 (axis 1 after permutation). Reduction results are replicated
within four-lane groups and broadcast back without another shuffle. Pending
and partial accumulator states remain separate until their explicit wait
and `finish_mma` steps.

## Worked consumers and current validation

[`examples/hopper/kernel.jl`](https://github.com/jool-space/Tylo.jl/blob/main/examples/hopper/kernel.jl)
is a complete 64×N GEMM pipeline with one producer warp, one consumer
warpgroup, and one or two stages. The prepared SM90 runtime tests exercise K tails, repeated slot
reuse, modified inputs, forced GC and graph replay when run on that device.
Megakernels uses two consumer warpgroups, task dependencies, a different
barrier participant count, fused gate/up epilogues and its own workspace.
Those policies live entirely in the respective consumers.

TMA loads and stores run on the local GB10 for every element width and
swizzle row. WGMMA is assembled for SM90a; GB10 cannot execute that
architecture-specific instruction. H100/H200 runtime and sanitizer results
are still required before claiming WGMMA correctness or performance.
Multicast/clusters, transposed WGMMA operands and interleaved outstanding
WGMMA groups remain future work; tcgen05 MMA is described in
[TMEM and tcgen05](tmem.md).

Contracts follow [NVIDIA's asynchronous copy guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
and [the PTX WGMMA specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions).
