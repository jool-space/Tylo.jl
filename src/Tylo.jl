module Tylo

using BFloat16s: BFloat16

# The kernel-writing vocabulary is exported: layouts and ownerships, register
# fragments, memory tiles, atoms with their operands and verbs, and the
# per-hardware copy, MMA and completion operations.
export Layouts, Layout, @Layout, Ownership, distribute, coordinate, compose, window,
       Fragment, coordinates, relayout, GlobalTile, SharedTile, TmemTile, tile, bound, load_fragment, store!,
       Atom, OperandA, OperandB, Accumulator, ScaleA, ScaleB, operand_layout,
       MMAAtom, ScaledMMAAtom, TiledMMA, zero_accumulator, mma, pack_operand_a,
       CopyAtom, copy_async!, commit_copies, wait_copies,
       TMATile, prepare_tma, shared_layout, shared_tile, transfer_bytes, swizzle_bytes, tma_load!, tma_store!, commit_tma_stores, wait_tma_reads, wait_tma_stores,
       WGMMAAtom, operand, mma_async, wait_mma, finish_mma,
       Tcgen05Atom, accumulator, commit_mma,
       reinterpret_tile, load_async, wait_load, store_async!, wait_stores,
       SoftmaxState, softmax_update, softmax_merge, softmax_normalize,
       Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU

# Inspection, validation and less common operations are public, used qualified.
@static if VERSION >= v"1.11.0-DEV.469"
    eval(Expr(:public,:Registers,:Addresses,:instruction_atoms,:copy_atoms,:validate_copy,
        :TMABinding,:swizzled_structure,:validate_wgmma,
        :Tcgen05Operand,:tmem_allocate!,:tmem_deallocate!,:tmem_relinquish_permit,
        :fence_after_thread_sync,:fence_before_thread_sync,:softmax_logsumexp,:convert_element,:threads))
end

include("tuples.jl")
include("elements.jl")
include("layouts/layouts.jl")
import .Layouts: Layout, @Layout, Ownership, distribute, coordinate, compose, window, tile
include("fragments.jl")
include("tmem.jl")
include("memory.jl")
include("copy.jl")
include("mma.jl")
include("copyatoms.jl")
include("rows.jl")
include("arrayops.jl")
include("online.jl")
include("tma.jl")
include("wgmma.jl")
include("tcgen05.jl")
include("derive.jl")

# GPU implementations load with PTX.jl. CPU layout/fragment operations have
# no dependency on a CUDA compiler, device, or instruction implementation.
"""
    wait_load(pending)

Complete prior TMEM loads and return this load's typed registers.
Returns a `Fragment` of the partition's element type.
The wait also carries a compiler dependency through the returned registers.
It waits for all prior loads of the executing threads, not just this handle.
"""
function wait_load end

"Complete all prior TMEM stores of the executing threads; warp collective."
function wait_stores end

"Order subsequent TMEM operations after a preceding thread synchronization."
function fence_after_thread_sync end

"Order preceding TMEM operations before a following thread synchronization."
function fence_before_thread_sync end

include("ptx/ptx.jl")

end
