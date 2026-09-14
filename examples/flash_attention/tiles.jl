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
