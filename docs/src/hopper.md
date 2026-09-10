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
julia> a = TMALoad(Tylo.BFloat16, Val((128,64)), Val(2));

julia> transfer_bytes(a)
16384

julia> shared_layout(a)((Int32(1), Int32(0)))
72
```

This is logical A(M,K), with K contiguous before swizzling. Logical B(K,N)
uses `TMALoad(T, Val((64,n)), Val(1))`. The host arrays passed to
`prepare_tma` are physically `(K, non-K)` Julia matrices in both cases:
transpose A's storage once; B already has this storage order.

The first descriptor-compatible layout is deliberately narrow: BF16 or FP16,
64 K elements, and 128-byte swizzling. The non-K extent must be a multiple
of eight, at most 256. `shared_layout(plan)` maps logical coordinates to
**element offsets**, just like Tylo's other layouts. For row r, the mapping
is `64r + xor(k,8*(r%8))`. A 1024-byte-aligned allocation establishes the
swizzle origin. `shared_tile(plan, pointer)` borrows that allocation.

`prepare_tma(plan, device_array)` returns an owner of the uploaded descriptor
and the source tensor. Adaptation produces an isbits device binding. Keep the
host binding alive through launches and graph replays; changing source values
is allowed, changing the address or dimensions requires a new descriptor.
PTX.jl performs the driver encoding and descriptor upload.

A descriptor is not available for an arbitrary composed layout. This subset
has an exact relationship to the hardware descriptor. A general `Layout`
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

Origins are zero-based logical coordinates. Out-of-bounds source elements
are zero-filled by TMA. The full tile byte count still contributes to the
mbarrier's expected transaction count. Tylo does not insert an arrival or a
wait: a producer may put A, B and another operand behind the same barrier.

The caller publishes barrier initialization with
`fence.proxy.async.shared::cta`, establishes visibility for preceding generic
writes as required, and waits for the matching barrier phase before consumption.
All consumers must finish before the producer overwrites the stage. These are
separate obligations; a completed TMA load does not mean WGMMA has stopped
reading that storage.

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

`finish_mma` returns an immutable distributed fragment. Its ownership map is
an ISA property, independent of storage. FP32 `map`, `scale`, and `store!` compose
an epilogue without a separate kernel. `WGMMAFragment` currently remains separate
from the generic `Fragment` broadcast/reduction API: `finish_mma` does not make
`exp.(result)` or `sum(result; dims=2)` supported automatically. There is no automatic cross-warp
redistribution or hidden scratch allocation.

## Worked consumers and current validation

[`examples/hopper/kernel.jl`](https://github.com/jool-space/Tylo.jl/blob/main/examples/hopper/kernel.jl)
is a complete 64×N GEMM pipeline with one producer warp, one consumer
warpgroup, and one or two stages. The prepared SM90 runtime tests exercise K tails, repeated slot
reuse, modified inputs, forced GC and graph replay when run on that device.
Megakernels uses two consumer warpgroups, task dependencies, a different
barrier participant count, fused gate/up epilogues and its own workspace.
Those policies live entirely in the respective consumers.

TMA runs on the local GB10. WGMMA is assembled for SM90a; GB10 cannot execute
that architecture-specific instruction. H100/H200 runtime and sanitizer
results are still required before claiming WGMMA correctness or performance.
Multicast/clusters, TMA stores, other swizzles, transposed WGMMA operands,
interleaved outstanding WGMMA groups, and tcgen05 MMA remain future work.

Contracts follow [NVIDIA's asynchronous copy guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
and [the PTX WGMMA specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions).
