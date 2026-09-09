# Representation and completion

Tylo's first experiment has two consumers in mind: standalone attention
kernels and the operations scheduled by Megakernels. Work scheduling belongs
to those consumers. Tylo describes data and the operations on that data.

## Three independent questions

**Ownership:** which logical values does a thread hold? The initial
`Layouts.LaneRows{N}` mapping gives each of 32 lanes a row of N elements.
This mapping is deliberately specific. An MMA accumulator with a different
thread/value distribution needs a different representation.

**Storage:** where do those values reside? A `TmemTile{T,N}` borrows 128
TMEM rows; `warp_rows` selects the 32-row band accessible to one warp.
An FP32 element occupies one physical column, while two BF16 elements share
one physical column. The warp-band field is separate from the column field.

**Completion:** when can another operation use the storage or registers?
`load_async` returns a pending value. `wait_load` makes its register
fragment available; the GPU implementation ties the registers through the
wait at LLVM level. Store completion and thread-synchronization fences
remain explicit.

The hardware address and completion rules follow the
[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).
All lanes of a warp must participate in the aligned collective operations
with the required uniform addresses. Julia's types do not prove collective
participation or unique ownership.

## Views and conversion

`reinterpret_tile(BFloat16, scores)` describes the same TMEM footprint with
twice as many logical columns. It changes no values. A subsequent column
slice can name the part used for probabilities. The kernel must ensure that
overwriting those scores is permitted.

`pack_bf16(values)` numerically converts FP32 register values, with adjacent
BF16 elements packed low-first. It preserves the lane/row distribution.

Register column offsets are static because dynamic tuple indexing can
materialize registers in local memory. TMEM column offsets may be runtime
values: they are address arithmetic, and forcing them to be static can
unnecessarily unroll the kernel's loops. Both kinds of view retain a static
width.

The raw constructors borrow storage; they do not allocate or free it.
A caller must supply an address within a sufficiently large allocation.
The kernel remains responsible for its allocation lifetime and reuse.

## Completion remains visible

Waiting for a load and releasing its source are separate decisions. In the
attention epilogue the final read completes before normalization and global
stores. Releasing the readout barriers at that point allows the next work
item to overlap that remaining arithmetic and output traffic.

Likewise, a sequence of TMEM stores may complete together. The kernel calls
`wait_stores()`, then `fence_before_thread_sync()`, then its barrier
arrival. Those operations are not hidden inside each individual store.

Pending values prevent accidental use through the fragment API. They are
ordinary Julia structs, not linear resources. A load wait completes all
prior loads of the executing threads; it is not an independently scoped
hardware event for one object.

## Layout module boundary

`Tylo.Layouts` currently holds the row ownership mapping and common static
column-interval validation. It is pure and independently testable. It also
supports hierarchical affine layouts, composition, XOR swizzles,
factorization and parent-relative windows; see [Layouts, storage and ownership](@ref).

The shared-memory GEMM path now uses those operations alongside distinct
MMA thread/value ownership mappings. A coherent mathematical API can later
move into Laythe.
Instruction compatibility, descriptors, and synchronization stay in Tylo.

## Current implementation limits

Device operations currently support FP32 TMEM row loads/stores of 16, 32,
or 64 values per thread and packed BF16 stores of 32, 64, or 128 values.
Global BF16 stores require 16-byte alignment and a multiple of eight values.
There is no bounds mask or partial-warp participation.

Shared/global tiles, cp.async copy plans, and warp MMA atoms/tiling are now
implemented in the complete GEMM example. TMA and Hopper WGMMA are also
implemented in a producer/consumer GEMM and used by Megakernels; see
[the Hopper path](hopper.md). TMA has GB10 runtime coverage, while WGMMA
execution still requires H100/H200 validation.

Tcgen05 MMA, arbitrary register redistribution, and allocation management
remain future work. The attention experiment still replaces only correction
and epilogue; its TMEM execution requires datacenter Blackwell validation.
