using Tylo.Layouts: @Layout
# Included into the FlashAttention definitions module by comparison.jl.
# The kernel's barrier protocol and role assignment remain explicit here.

# One 128x128 BF16 tile is two 128-row by 64-column B128 stripes, which is
# Tylo's canonical TMA storage. The reference kernel passes the raw descriptor
# pointer; the plan carries no runtime state, so binding it here costs nothing.
const FAB_TMA_PLAN = Tylo.TMALoad(BFloat16,Val((128,64)),Val(2))
@inline function fab_load_tile(dst_ptr, byteoff::Int, tma, row::UInt32, mbar)
    binding = Tylo.DeviceTMA(FAB_TMA_PLAN,tma)
    for stripe in 0:1
        tile = Tylo.shared_tile(binding,dst_ptr+byteoff+stripe*FAB_STRIPE_BYTES)
        @inbounds Tylo.tma_load!(tile,binding,(reinterpret(Int32,row),Int32(64stripe)),mbar)
    end
end

@inline function fab_corr_tile(d::FabDbg, ::Val{STAGE}, bars::BarrierSet,
                                stats, alpha_idx::Int,
                                stats_bar::UInt32, o_addr::UInt32,
                                ponc::UInt32, ph_pv::UInt32) where STAGE
    ptx"bar.sync"(stats_bar,UInt32(64))
    alpha = @inbounds stats[alpha_idx]
    barrier_arrive(bars.stats_free[STAGE])
    ph_pv = fab_wait(d,bars.pv_done[STAGE],ph_pv,Val(10))
    needs_correction = nvvm"vote.ballot.sync"(0xffffffff,alpha < 1f0)
    if needs_correction != UInt32(0)
        Tylo.fence_after_thread_sync()
        output = Tylo.TmemTile(Float32,o_addr,@Layout((32,128),(1,128)))
        for half in 0:1
            chunk = @inbounds Tylo.partition(Tylo.TmemTransfer{(32,64),2}(),
                Tylo.window(output,(UInt32(0),UInt32(64half)),Val((32,64))))
            values = Tylo.wait_load(Tylo.load_async(chunk))
            Tylo.store_async!(chunk,values .* alpha)
        end
        Tylo.wait_stores()
        Tylo.fence_before_thread_sync()
        barrier_arrive(bars.o_resc[STAGE,0] + Int(ponc))
    end
    ph_pv
end

@inline function fab_epi_stage(::Val{STAGE}, bars::BarrierSet,
                                stats, alpha_idx::Int,
                                stats_bar::UInt32, o_addr::UInt32,
                                out_row::UInt32, po) where STAGE
    ptx"bar.sync"(stats_bar,UInt32(64))
    inv_sum = ptx"rcp.approx.f32"(@inbounds stats[alpha_idx+256])
    row = out_row + UInt32(STAGE*FAB_BM)
    dst = po + Int(row)*(FAB_HD*2)
    output = Tylo.TmemTile(Float32,o_addr,@Layout((32,128),(1,128)))

    for half in 0:1
        chunk = @inbounds Tylo.partition(Tylo.TmemTransfer{(32,64),2}(),
                Tylo.window(output,(UInt32(0),UInt32(64half)),Val((32,64))))
        values = Tylo.wait_load(Tylo.load_async(chunk))
        if STAGE == 1 && half == 1
            # Release immediately after the final TMEM read; the next item's
            # work can overlap this item's register arithmetic/global stores.
            barrier_arrive(bars.stats_free[1])
            barrier_arrive(bars.epi)
            barrier_arrive(bars.o_resc[0,0])
            barrier_arrive(bars.o_resc[1,0])
        end
        @unroll for quarter in 0:1
            part = Tylo.window(values,Val((0,32quarter)),Val((32,32)))
            packed = Tylo.pack(Tylo.BFloat16,part .* inv_sum)
            Tylo.store!(dst + 128half + 64quarter,packed)
        end
    end
    STAGE == 0 && barrier_arrive(bars.stats_free[0])
    nothing
end

# The MMA issue. Both products are one atom each. TMA stores every tile as
# two 64-column stripes of 128-byte rows, 16 KiB apart, so a buffer of such
# tiles is one stack of 128-byte rows in which a tile occupies rows
# [256t, 256t+128) and its second column stripe sits 128 rows later. The
# layouts say exactly that; origins select the stage, the K/V slot and the
# K range of each publish group, and every offset folds to a constant.
const FAB_QK_ATOM = Tylo.Tcgen05MMA((128,FAB_BN,16),BFloat16)
const FAB_PV_ATOM = Tylo.Tcgen05MMA((128,FAB_HD,16),BFloat16)
const FAB_ROWS_LAYOUT = Tylo.Layouts.compose(Tylo.Layouts.Swizzle{3,3,3}(),
    @Layout((512,(64,2)),(64,(1,8192))))                  # (rows, columns): Q stages, V slots
const FAB_ROWS_T_LAYOUT = Tylo.Layouts.compose(Tylo.Layouts.Swizzle{3,3,3}(),
    @Layout(((64,2),512),((1,8192),64)))                  # (columns, rows): K slots as B(K,N)
@inline fab_bf16_tile(ptr,layout) = Tylo.SharedTile(reinterpret(Core.LLVMPtr{BFloat16,3},ptr),layout)
@inline function fab_mma_operands(q_ptr, kv_ptr)
    # The kernel's dynamic shared memory is 1024-byte aligned by construction.
    q = @inbounds Tylo.tcgen05_operand(FAB_QK_ATOM,Tylo.OperandA(),fab_bf16_tile(q_ptr,FAB_ROWS_LAYOUT))
    k = @inbounds Tylo.tcgen05_operand(FAB_QK_ATOM,Tylo.OperandB(),fab_bf16_tile(kv_ptr,FAB_ROWS_T_LAYOUT))
    v = @inbounds Tylo.tcgen05_operand(FAB_PV_ATOM,Tylo.OperandB(),fab_bf16_tile(kv_ptr + FAB_TILE_BYTES,FAB_ROWS_LAYOUT))
    (; q, k, v)
end

@inline function fab_qk_mma(::Val{STAGE}, ::Val{KIDX}, tmem::UInt32, ops,
                             bars::BarrierSet) where {STAGE, KIDX}
    s = Tylo.accumulator(FAB_QK_ATOM,tmem + UInt32(STAGE*128))
    a = @inbounds Tylo.tcgen05_operand(ops.q,(Int32(256STAGE),Int32(0)))
    b = @inbounds Tylo.tcgen05_operand(ops.k,(Int32(0),Int32(512KIDX)))
    @inbounds Tylo.mma(FAB_QK_ATOM,s,a,b,Val(FAB_HD),false)
    Tylo.commit_mma(bars.s_full[STAGE])
end

@inline function fab_pv_mma(dbg::FabDbg, ::Val{SPLITP}, ::Val{STAGE}, ::Val{VIDX},
                             first_accum::Bool,
                             ::Val{PAR}, tmem::UInt32, ops,
                             bars::BarrierSet, ph_pq::UInt32, ph_or::UInt32) where {SPLITP, STAGE, VIDX, PAR}
    o = Tylo.accumulator(FAB_PV_ATOM,tmem + UInt32(256 + STAGE*128))
    p = Tylo.TmemTile(BFloat16,tmem + UInt32(STAGE*128),@Layout((128,FAB_BN),(1,128)))
    NG = SPLITP ? 2 : 4        # publish groups per tile (halves / quarters)
    W = FAB_BN ÷ NG            # key rows per group
    for g in 0:(NG - 1)
        fab_wait(dbg, bars.p_q[STAGE, g], ph_pq, Val(5))
        if g == 0
            ph_or = fab_wait(dbg, bars.o_resc[STAGE, PAR], ph_or, Val(6))
        end
        Tylo.fence_after_thread_sync()
        a = @inbounds Tylo.window(p,(Int32(0),Int32(W*g)),Val((128,W)))
        b = @inbounds Tylo.tcgen05_operand(ops.v,(Int32(512VIDX + W*g),Int32(0)))
        @inbounds Tylo.mma(FAB_PV_ATOM,o,a,b,Val(W),!(first_accum & (g == 0)))
    end
    Tylo.commit_mma(bars.pv_done[STAGE])
    return ph_pq ⊻ UInt32(1), ph_or
end
