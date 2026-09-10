module Tylo

using BFloat16s: BFloat16

export SoftmaxState, softmax_update, softmax_merge, softmax_normalize, softmax_logsumexp

export WarpRowFragment, RowValues, row_ownership, row_coordinate, row_sum, row_max, row_map

export TMALoad, prepare_tma, shared_layout, shared_tile, transfer_bytes, tma_load!,
       WGMMA64, validate_wgmma, wgmma_operand, mma_async, wait_mma, finish_mma

export GlobalTile, SharedTile, window, CopyPlan, validate_copy, copy_async!,
       commit_copies, wait_copies, MMA16x8x16, TiledMMA, zero_accumulator,
       load_a, load_b, pack_operand_a, mma, store!, operand_layout, OperandA, OperandB, Accumulator,
       Fragment, RowFragment, PackedBF16, columns, scale, pack_bf16,
       TmemTile, TmemTransfer, partition, reinterpret_tile,
       load_async, wait_load, store_async!, wait_stores,
       fence_after_thread_sync, fence_before_thread_sync

include("layouts/layouts.jl")
include("fragments.jl")
include("tmem.jl")
include("memory.jl")
include("copy.jl")
include("mma.jl")
include("rows.jl")
include("arrayops.jl")
include("online.jl")
include("tma.jl")
include("wgmma.jl")

# GPU implementations load with PTX.jl. CPU layout/fragment operations have
# no dependency on a CUDA compiler, device, or instruction implementation.
"""
    pack_bf16(fragment::Fragment{Float32})

GPU operation: round adjacent FP32 values to BF16 and pack low element first.
Returns a `PackedBF16` fragment with the same logical ownership.
"""
function pack_bf16 end

"""
    load_async(partition)

Issue a warp-collective FP32 TMEM load using an explicit transfer partition.
All 32 lanes must execute with the same partition and the correct warp band. The result is pending; call `wait_load` before using its values.
"""
function load_async end

"""
    wait_load(pending)

Complete prior TMEM loads and return this load's FP32 register fragment.
The wait also carries a compiler dependency through the returned registers.
It waits for all prior loads of the executing threads, not just this handle.
"""
function wait_load end

"""
    store_async!(partition, fragment)

Issue a warp-collective store to TMEM. FP32 fragments store to FP32 partitions;
packed BF16 fragments store to BF16 partitions with matching ownership.
Storage remains in use until `wait_stores()`.
"""
function store_async! end

"Complete all prior TMEM stores of the executing threads; warp collective."
function wait_stores end

"Order subsequent TMEM operations after a preceding thread synchronization."
function fence_after_thread_sync end

"Order preceding TMEM operations before a following thread synchronization."
function fence_before_thread_sync end

end
