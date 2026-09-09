module Tylo

using BFloat16s: BFloat16

export TMALoad, prepare_tma, shared_layout, shared_tile, transfer_bytes, tma_load!,
       WGMMA64, validate_wgmma, wgmma_operand, mma_async, wait_mma, finish_mma

export GlobalTile, SharedTile, window, CopyPlan, validate_copy, copy_async!,
       commit_copies, wait_copies, MMA16x8x16, TiledMMA, zero_accumulator,
       load_a, load_b, mma, store!, operand_layout, OperandA, OperandB, Accumulator,
       RowFragment, PackedBF16, columns, scale, pack_bf16,
       TmemTile, TmemRows, warp_rows, reinterpret_tile,
       load_async, wait_load, store_async!, wait_stores, store_row!,
       fence_after_thread_sync, fence_before_thread_sync

include("layouts.jl")
include("fragments.jl")
include("tmem.jl")
include("memory.jl")
include("copy.jl")
include("mma.jl")
include("tma.jl")
include("wgmma.jl")

# GPU implementations load with PTX.jl. CPU layout/fragment operations have
# no dependency on a CUDA compiler, device, or instruction implementation.
"""
    pack_bf16(fragment::RowFragment{Float32})

GPU operation: round adjacent FP32 values to BF16 and pack low element first.
Returns a `PackedBF16` fragment with the same logical column count.
"""
function pack_bf16 end

"""
    load_async(rows::TmemRows{Float32})

Issue a warp-collective TMEM load. All 32 lanes must execute with the same
address. The result is pending; call `wait_load` before using its values.
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
    store_async!(rows, fragment)

Issue a warp-collective store to TMEM. FP32 fragments store to FP32 views;
packed BF16 fragments store to BF16 views of the same logical width.
Storage remains in use until `wait_stores()`.
"""
function store_async! end

"Complete all prior TMEM stores of the executing threads; warp collective."
function wait_stores end

"""
    store_row!(ptr, packed::PackedBF16)

Store this thread's packed BF16 row to a 16-byte-aligned global UInt16 pointer.
The caller supplies a valid, sufficiently large, distinct row per thread.
"""
function store_row! end

"Order subsequent TMEM operations after a preceding thread synchronization."
function fence_after_thread_sync end

"Order preceding TMEM operations before a following thread synchronization."
function fence_before_thread_sync end

end
