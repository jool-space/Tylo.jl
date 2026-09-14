# Blackwell tcgen05 FlashAttention forward (bf16, head_dim 128).
#
# Reference kernel for the Tylo integration experiment in this directory.
# Vendored from PTX.jl (test/gpu/blackwell/flash_attention_defs.jl), itself an
# algorithmic port of pyptx examples/blackwell/flash_attention_blackwell.py
# `build_flash_attention_blackwell` (the 1-CTA variant), Copyright 2026
# Patrick Toulmé, Apache License 2.0. The architecture follows
# FlashAttention-4 / CUTLASS 77_blackwell_fmha.
#
# comparison.jl loads this file into two isolated modules and replaces only
# the correction and epilogue helpers in the second one with tiles.jl. The
# kernel is parameterized by a compile-time config NamedTuple passed as
# `Val(cfg)`; each config compiles a fresh specialization:
#
#   scoreboard :: Bool  — drop the explicit `tcgen05.wait::ld` in the
#       softmax stream and lean on the SASS register scoreboard for the
#       RAW hazard (FA4 ships this: it emits 0 wait::ld). Default false
#       (spec-conservative, B200-validated).
#   beacon     :: Bool  — hang forensics: every mbarrier wait becomes a
#       bounded try_wait; on timeout the thread records
#       0xBEAC0000 | site<<8 | phase into its dbg slot (first blame only)
#       and PRETENDS SUCCESS so the pipeline drains and the kernel exits
#       instead of hanging. Decode with FAB_SITES. Default false.
#   nreg       :: NTuple{3,Int} — setmaxnreg split (softmax, correction,
#       mma/load warpgroup). Constraint: 256a + 128b + 128c ≤ 65536.
#       Default (144, 80, 144) — the hardware-swept optimum (+6% over
#       pyptx's (136, 96, 144); producer floor is 72 — n2=64 fails ptxas).
#   emu        :: NTuple{N,Int} — softmax exp2 pairs (0-based, of 16 per
#       32-column chunk) computed by the FMA-pipe f32x2 polynomial
#       emulation instead of ex2.approx (pyptx `emu_pairs`; rebalances
#       SFU vs FMA pipe pressure — pyptx's headline config uses
#       (1, 3, 5, 7)). Default () — ex2 everywhere: the emulation
#       measured negative at every density on B200 and B300, alone and
#       stacked on splitp; the knob remains for reproduction. It changes
#       numerics (poly + round-to-int exponent inject), so fab_check
#       gates each swept config.
#   splitp     :: Bool — half-granular split-P (pyptx's 2-CTA "stage E"
#       mechanism, commit 4d2aa00, mapped onto the 1-CTA kernel): the
#       softmax still stores P per 32-column chunk, but the publish
#       (wait::st → fence → mbarrier arrive) happens per 64-column HALF
#       — half the wait::st round trips — and the PV consumer waits per
#       half instead of per quarter: kk 0-3 issue under half 0 while
#       the softmax still streams half 1 (which gates kk 4-7). The P
#       barrier group reshapes [stage, quarter] → [stage, half].
#       Default true — same-box A/B winner at the saturated shape on
#       B200 (+4.5%) and B300 (+14.7%), non-overlapping bands; fab_check
#       green incl. the rescale-heavy gate. splitp=false is the pre-port
#       quarter-granular protocol (byte-identical PTX to the
#       pre-adoption tree), kept as the fallback; numerics are
#       UNCHANGED either way (same stores, same values — only publish
#       granularity moves).
#
# Structure (faithful to pyptx):
#   * 256 query rows per CTA as two 128-row subtiles ("stages"); BN=128 KV
#     tiles staged through a 4-slot SMEM ring (K,V alternating) fed by TMA.
#   * tcgen05.mma computes S = Q@K^T into TMEM; P@V accumulates onto O in
#     TMEM with the A operand sourced directly FROM TMEM (the new
#     `tcgen05.mma` UInt32-A method → llvm.nvvm.tcgen05.mma.tensor).
#   * Warp-specialized, 16 warps: softmax (w0-7, two stage-parallel
#     warpgroups; thread t owns row t&127 and streams its 128 columns in
#     four 32-column chunks, double-buffered), correction (w8-11), MMA
#     dispatch (w12 lane 0), TMA load (w14 lane 0); w13/w15 idle.
#   * Stale-basis softmax: each tile exps against the PREVIOUS tile's
#     basis immediately; the row max / basis decision defers one tile and
#     runs in the next tile's TMEM-load shadow. P publishes per
#     64-column half (default `splitp = true`), so PV's kk 0-3 start
#     while the softmax still streams the second half; `splitp = false`
#     restores the per-32-column-quarter publish (see the knob above).
#   * FA4 "skip correction": O is rescaled in TMEM only when the running
#     row max moved by more than RESCALE_THRESHOLD in scaled-log2 units.
#     In the common no-rescale case the softmax warps release the O
#     barrier themselves (warp ballot); the correction warps re-derive
#     the same ballot from the alphas in shared memory, so barrier
#     arrival counts stay exact.
#   * BAR_O_RESC is split by KV-tile parity ([stage][tile&1]): the alpha
#     stream can run up to two tiles ahead of the MMA warp's consumption,
#     which would alias a single barrier's phase parity.
#   * ONE cross-work-item gate: the softmax pool's item start waits for
#     the previous epilogue's O readout (BAR_EPI) — softmax TMEM traffic
#     concurrent with the epilogue's O reads corrupts rows.
#   * Persistent scheduling: grid = one wave of CTAs, each looping over
#     (m_block, bh) work items; m_block varies fastest so consecutive
#     CTAs share K/V in L2.
#
# TMEM layout (512 columns of 32-bit):
#     cols   0-127   S0 (f32) — P0 (packed bf16) overwrites cols 0-63
#     cols 128-255   S1 (f32) — P1 overwrites cols 128-191
#     cols 256-383   O0 (f32 accumulator)
#     cols 384-511   O1 (f32 accumulator)
#
# Tensors are passed 2D as (batch*heads*seqlen, head_dim) bf16 row-major.
#
# Port notes vs pyptx (deliberate, all diagnosed — not casual):
#   * Explicit `tcgen05.wait::ld` in the softmax stream (pyptx default
#     `dbg_scoreboard_ld=True` drops it and leans on the SASS register
#     scoreboard, matching FA4's 0 wait::ld). Correct-by-spec first; the
#     scoreboard variant is a measured-perf follow-up.
#   * The FMA-pipe f32x2 exp2 emulation (`emu_pairs`) is ported behind
#     the `emu` config knob but defaults OFF (all 16 pairs use
#     `ex2.approx.ftz.f32`) until a B200 session sweeps it; the trick
#     rebalances SFU/FMA pipe pressure.
#   * Half-granular split-P (pyptx ships it only in the 2-CTA variant,
#     commit 4d2aa00) is ported onto this 1-CTA kernel behind the
#     `splitp` knob and adopted as the default after same-box B200 and
#     B300 A/Bs. `emu` was retested ON TOP of split-P (pyptx's pairing)
#     and stayed negative — emu is falsified terminally; the knob
#     remains for reproduction only.
#   * Debug scaffolding (waitmap / beacons / lockstep / s_f16) not ported.
#   * Shape parameters (seqlen, n_tiles, total_work, ...) are runtime
#     kernel arguments — one compiled kernel serves every shape (pyptx
#     rebuilds per shape). Static loop structure is unchanged: the pyptx
#     build-time `if trips > 0` becomes a runtime while that trips 0×.

using PTX: smem_addr_u32, tcgen05_descriptor, tcgen05_instr_desc_f16bf16_f32,
           BlackwellLayout, tensor_map_tile_2d, bf16x2_pack
using PTX.Utils: @unroll, strided_reduce
using PTX.MBarriers: BarrierSet, barrierset_init!, barrier_bytes,
                     barrier_arrive, barrier_arrive_expect_tx,
                     barrier_try_wait
using PTX.NVVM                    # @nvvm_str for vote.ballot.sync
using CUDACore
using Tylo: BFloat16
using Random

const FAB_BM       = 128            # rows per Q subtile
const FAB_QSTAGE   = 2              # subtiles per CTA → 256 rows
const FAB_BN       = 128            # KV tile size
const FAB_HD       = 128            # head dim (fixed)
const FAB_KV_SLOTS = 4              # SMEM ring slots, each one K or V tile
const FAB_THREADS  = 512
const FAB_LOG2E    = 1.4426950408889634
const FAB_THRESH   = 256.0f0        # 2^8: FA4 skip-correction gate (post-exp2 domain)

const FAB_TILE_BYTES   = FAB_BN * FAB_HD * 2       # 32 KiB per K/V/Q tile
const FAB_STRIPE_BYTES = 16384                     # TMA 128B-swizzle stripe
const FAB_Q_BYTES      = FAB_QSTAGE * FAB_TILE_BYTES
const FAB_SLOT_UNITS   = FAB_TILE_BYTES ÷ 16       # ring-slot stride, 16B desc units

# SMEM map (byte offsets into dynamic SMEM)
const FAB_SMEM_Q     = 0
const FAB_SMEM_KV    = FAB_SMEM_Q + FAB_Q_BYTES                     # 65536
const FAB_SMEM_STATS = FAB_SMEM_KV + FAB_KV_SLOTS * FAB_TILE_BYTES  # 196608
#   alpha[2][128] @ +0 (f32, stage-major), sum[2][128] @ +1024
const FAB_SMEM_BARS  = FAB_SMEM_STATS + 2048

# The synchronization plan as one BarrierSet declaration: shapes and
# arrival counts here; byte offsets, the thread-0 init sequence, and the
# arena footprint are all derived. Layout order matches the historical
# hand-maintained offset table (see PTX.MBarriers.BarrierSet).
const FAB_BARS = (
    q          = ((2,),   1),    # per Q subtile; TMA tx
    kv_full    = ((4,),   1),    # per ring slot; TMA tx
    kv_free    = ((4,),   1),    # per ring slot; tcgen05.commit
    s_full     = ((2,),   1),    # per stage; tcgen05.commit
    p_q        = ((2, 4), 128),  # [stage, quarter]; softmax publishes P
    o_resc     = ((2, 2), 128),  # [stage, tile parity]; gates PV
    o_done     = ((),     1),    # tcgen05.commit
    stats_free = ((2,),   128),  # per stage; paces the alpha/sum slots
    pv_done    = ((2,),   1),    # per stage; tcgen05.commit
    q_free     = ((),     1),    # tcgen05.commit
    epi        = ((),     128),  # softmax item-start gate
)
const FAB_BAR_BYTES = barrier_bytes(BarrierSet{FAB_BARS})

# splitp=true table: the P publish group reshapes to [stage, half]
# (pyptx 4d2aa00's B2_P_FULL [stage][half]); every other group is
# unchanged. The arena SHRINKS by 4 barriers, so the splitp=false
# footprint (FAB_BAR_BYTES) is the layout maximum — the SMEM map keeps
# the splitp=false offsets for both configs and the splitp=true arena
# simply leaves its last 32 bytes unused.
const FAB_BARS_SP = (; FAB_BARS..., p_q = ((2, 2), 128))

# Dispatch-based table selection (kernel-path validation must be
# dispatch, not runtime membership — see CLAUDE.md's 1.10 note).
@inline fab_barset(::Val{false}, base) = BarrierSet{FAB_BARS}(base)
@inline fab_barset(::Val{true},  base) = BarrierSet{FAB_BARS_SP}(base)

const FAB_SMEM_TMEM  = FAB_SMEM_BARS + FAB_BAR_BYTES
const FAB_SMEM_BYTES = FAB_SMEM_TMEM + 8

# TMEM columns: S/P at stage*128, O at 256 + stage*128.

const FAB_MMA_TID  = UInt32(384)   # warp 12 lane 0
const FAB_LOAD_TID = UInt32(448)   # warp 14 lane 0

# ── compile-time config ─────────────────────────────────────────────────

# Default nreg = the hardware-swept optimum (setmaxnreg is
# warpgroup-collective): 144*256 + 80*128 + 144*128 = 65536, +6% over
# pyptx's original (136, 96, 144) split. The sweep bracketed the optimum
# and found the producer's register floor: every n2=64 candidate fails
# ptxas (the CORRECTION warpgroup floors at 72). Correctness re-gated by
# fab_check at each candidate (rescale-heavy input, scale=2.0).
# splitp = true: same-box A/Bs win at the saturated shape with
# non-overlapping bands on B200 (+4.5%) and B300 (+14.7%).
# fab_check green (incl. the rescale-heavy scale=2.0 gate) for splitp
# and splitp+emu; emu on top of splitp read 1128–1146 — negative, so
# emu stays falsified even in pyptx's pairing.
const FAB_CFG_DEFAULT = (scoreboard = false, beacon = false,
                         nreg = (144, 80, 144), emu = (), splitp = true)

function fab_check_cfg(cfg)
    a, b, c = cfg.nreg
    pool = 256a + 128b + 128c
    pool <= 65536 ||
        error("nreg $(cfg.nreg): 256*$a + 128*$b + 128*$c = $pool > 65536")
    all(x -> 24 <= x <= 256 && x % 8 == 0, cfg.nreg) ||
        error("nreg values must be multiples of 8 in [24, 256]")
    all(x -> x isa Int && 0 <= x <= 15, cfg.emu) && allunique(cfg.emu) ||
        error("emu pairs must be unique Ints in 0:15")
    cfg.splitp isa Bool || error("splitp must be a Bool")
    cfg
end

# setmaxnreg through a Val function barrier: the immediate operand must
# be a compile-time constant, and the type parameter guarantees it
# survives any constprop weather between the config NamedTuple and the
# immarg.
@inline fab_maxnreg_inc(::Val{N}) where {N} =
    ptx"setmaxnreg.inc.sync.aligned.u32"(Val(N))
@inline fab_maxnreg_dec(::Val{N}) where {N} =
    ptx"setmaxnreg.dec.sync.aligned.u32"(Val(N))

# ── beacon plumbing (hang forensics) ────────────────────────────────────
# pyptx's `bwait` idea: waits are bounded; on timeout, record a
# first-blame code and pretend success so the pipeline drains and the
# host gets its buffer back instead of a hung kernel. One UInt32 slot
# per thread; 0 = never timed out.

struct FabDbg{BEACON}
    ptr::Core.LLVMPtr{UInt32, 1}
    tid::UInt32
end

const FAB_SITES = Dict(
    1  => "load:KV_FREE",   2  => "load:Q_FREE",
    3  => "mma:Q",          4  => "mma:KV_FULL",
    5  => "pv:P_Q",         6  => "pv:O_RESC",
    7  => "softmax:S_FULL", 8  => "softmax:STATS_FREE",
    9  => "softmax:EPI",    10 => "corr:PV_DONE",
    11 => "corr:O_DONE",
)

# Waits return the barrier's NEXT phase parity, so a per-barrier phase
# variable updates as `ph = fab_wait(d, bar, ph, site)`. Callers whose
# phase is shared across a barrier group (p_q) discard the return and
# flip once per group.
@inline function fab_wait(::FabDbg{false}, mbar, ph::UInt32, ::Val)
    barrier_try_wait(mbar, ph)
    ph ⊻ UInt32(1)
end
@inline function fab_wait(d::FabDbg{true}, mbar, ph::UInt32,
                           ::Val{SITE}) where {SITE}
    tries = UInt32(0)
    while !ptx"mbarrier.try_wait.parity.shared.b64"(mbar, ph)
        tries += UInt32(1)
        if tries > UInt32(3_000_000)
            slot = d.ptr + Int(d.tid) * 4
            if ptx"ld.global.b32"(slot) == UInt32(0)      # first blame only
                ptx"st.global.b32"(slot,
                    UInt32(0xBEAC0000) | (UInt32(SITE) << 8) | ph)
            end
            return ph ⊻ UInt32(1)   # pretend success: let the pipeline drain
        end
    end
    ph ⊻ UInt32(1)
end

# Per-work-item Q row base: m_block = w & mb_mask varies fastest,
# bh = w >> mb_shift.
@inline fab_q_row0(w::UInt32, seqlen::UInt32, mb_shift::UInt32, mb_mask::UInt32) =
    (w >> mb_shift) * seqlen + (w & mb_mask) * UInt32(FAB_QSTAGE * FAB_BM)

# One 128×128 bf16 tile = two TMA 128-row × 64-col swizzle stripes.
@inline function fab_load_tile(dst_ptr, byteoff::Int, tma, row::UInt32, mbar)
    for stripe in 0:1
        ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            dst_ptr + byteoff + stripe * FAB_STRIPE_BYTES, tma,
            Int32(stripe * 64), reinterpret(Int32, row), mbar)
    end
end

# Ring-slot load: wait FREE (uniform accounting — the ring is self-credited
# once at role start), expect_tx, two stripe loads. Returns advanced row.
@inline function fab_load_kv(d::FabDbg, ::Val{SLOT}, kv_ptr, tma, row::UInt32,
                              bars::BarrierSet,
                              ph::UInt32) where {SLOT}
    ph = fab_wait(d, bars.kv_free[SLOT], ph, Val(1))
    barrier_arrive_expect_tx(bars.kv_full[SLOT], FAB_TILE_BYTES)
    fab_load_tile(kv_ptr, SLOT * FAB_TILE_BYTES, tma, row, bars.kv_full[SLOT])
    return row + UInt32(FAB_BN), ph
end

@inline fab_commit(mbar_ptr) =
    ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64"(
        smem_addr_u32(mbar_ptr))

# Descriptor offset (16B units) for k-atom kk of a K-major 128-col bf16
# tile: k-atoms 0-3 live in stripe 0 (+32 B each), 4-7 in stripe 1 (+16 KiB).
@inline fab_kmaj_off(kk::Int) = UInt64((kk ÷ 4) * 1024 + (kk % 4) * 2)

# S(stage) = Q(stage) @ K(k_idx)^T — 8 k16 UMMAs, accumulate within the tile.
@inline function fab_qk_mma(::Val{STAGE}, ::Val{KIDX}, tmem::UInt32,
                             dq0::UInt64, dk0::UInt64, idesc::UInt32,
                             bars::BarrierSet) where {STAGE, KIDX}
    d = tmem + UInt32(STAGE * 128)
    for kk in 0:7
        da = dq0 + UInt64(STAGE * FAB_SLOT_UNITS) + fab_kmaj_off(kk)
        db = dk0 + UInt64(KIDX * 2 * FAB_SLOT_UNITS) + fab_kmaj_off(kk)
        ptx"tcgen05.mma.cta_group::1.kind::f16"(d, da, db, idesc, kk != 0)
    end
    fab_commit(bars.s_full[STAGE])
end

# O(stage) += P(stage) @ V(v_idx) — A sourced from TMEM, consumed per
# publish group as the softmax releases it: 64-column halves under
# splitp (kk 0-3 issue under half 0 while the softmax still streams
# half 1, which gates kk 4-7 — pyptx 4d2aa00's pv_mma), 32-column
# quarters otherwise. PV(tile) is gated on O_RESC[stage][tile&1] at the
# first group.
@inline function fab_pv_mma(dbg::FabDbg, ::Val{SPLITP}, ::Val{STAGE}, ::Val{VIDX},
                             first_accum::Bool,
                             ::Val{PAR}, tmem::UInt32, dv0::UInt64,
                             idesc::UInt32, bars::BarrierSet,
                             ph_pq::UInt32, ph_or::UInt32) where {SPLITP, STAGE,
                                                                  VIDX, PAR}
    d = tmem + UInt32(256 + STAGE * 128)
    NG = SPLITP ? 2 : 4        # publish groups per tile (halves / quarters)
    W  = 8 ÷ NG                # k-atoms per group
    for g in 0:(NG - 1)
        # the group's barriers share ph_pq: flip once per tile, at return
        fab_wait(dbg, bars.p_q[STAGE, g], ph_pq, Val(5))
        if g == 0
            ph_or = fab_wait(dbg, bars.o_resc[STAGE, PAR], ph_or, Val(6))
        end
        ptx"tcgen05.fence::after_thread_sync"()
        for kk in (W * g):(W * g + W - 1)
            # `% UInt32`: the unchecked conversion is exact by construction
            # (kk ∈ 0:7); the checked UInt32() truncation would leave a
            # live InexactError branch in the loop-carried form.
            a = tmem + ((STAGE * 128 + kk * 8) % UInt32)
            db = dv0 + ((VIDX * 2 * FAB_SLOT_UNITS + kk * 128) % UInt64)
            ptx"tcgen05.mma.cta_group::1.kind::f16"(d, a, db, idesc,
                                                    !(first_accum & (kk == 0)))
        end
    end
    # deferred-rescale handshake: correction may only touch O after this
    # tile's PV has fully accumulated
    fab_commit(bars.pv_done[STAGE])
    return ph_pq ⊻ UInt32(1), ph_or
end

# ── softmax helpers (per-thread full row, 32-column chunks) ─────────────

# FMA-pipe exp2 emulation (pyptx `exp_pair_emu`, ported bit-for-bit):
# clamp → packed fma (scale + stale-basis bias) → round-to-int via the
# 2^23+2^22 add/sub trick (add.rm keeps the fraction in [-0.5, 0.5)) →
# degree-3 poly in the fraction → inject the integer part into the
# exponent field with a bit-shift/add. Constants are pyptx's exact bit
# patterns (examples/blackwell/flash_attention_blackwell.py:106-111).
const FAB_EX2_CLAMP = reinterpret(Float32, 0xC2FE0000)   # -127.0
@inline fab_pack2(lo::Float32, hi::Float32) =
    (UInt64(reinterpret(UInt32, hi)) << 32) | UInt64(reinterpret(UInt32, lo))
const FAB_EX2_BIG2 = fab_pack2(reinterpret(Float32, 0x4B400000),
                                reinterpret(Float32, 0x4B400000))
const FAB_EX2_C32  = fab_pack2(reinterpret(Float32, 0x3D9DF09D),
                                reinterpret(Float32, 0x3D9DF09D))
const FAB_EX2_C22  = fab_pack2(reinterpret(Float32, 0x3E6906A4),
                                reinterpret(Float32, 0x3E6906A4))
const FAB_EX2_C12  = fab_pack2(reinterpret(Float32, 0x3F31F519),
                                reinterpret(Float32, 0x3F31F519))
const FAB_EX2_C02  = fab_pack2(1.0f0, 1.0f0)

@inline function fab_exp_pair_emu(v0::UInt32, v1::UInt32,
                                   qk2::UInt64, m2::UInt64)
    a0 = ptx"max.ftz.f32"(reinterpret(Float32, v0), FAB_EX2_CLAMP)
    a1 = ptx"max.ftz.f32"(reinterpret(Float32, v1), FAB_EX2_CLAMP)
    x  = ptx"fma.rn.ftz.f32x2"(fab_pack2(a0, a1), qk2, m2)
    bi = ptx"add.rm.ftz.f32x2"(x, FAB_EX2_BIG2)
    f  = ptx"sub.rn.ftz.f32x2"(x, ptx"sub.rn.ftz.f32x2"(bi, FAB_EX2_BIG2))
    pl = ptx"fma.rn.ftz.f32x2"(f, FAB_EX2_C32, FAB_EX2_C22)
    pl = ptx"fma.rn.ftz.f32x2"(pl, f, FAB_EX2_C12)
    pl = ptx"fma.rn.ftz.f32x2"(pl, f, FAB_EX2_C02)
    r0 = ((bi % UInt32) << 23) + (pl % UInt32)
    r1 = (((bi >> 32) % UInt32) << 23) + ((pl >> 32) % UInt32)
    reinterpret(Float32, r0), reinterpret(Float32, r1)
end

# @generated: the emu/ex2 split is per-pair compile-time state; explicit
# unrolled assignments keep every lane in registers (a 32-wide closure
# would escape the inliner and become a real device CALL).
@generated function fab_exp_chunk(::Val{EMU}, v::NTuple{32, UInt32},
                                   qk_scale::Float32, mneg::Float32,
                                   qk2::UInt64) where {EMU}
    stmts = Expr[]
    isempty(EMU) || push!(stmts, :(m2 = fab_pack2(mneg, mneg)))
    lanes = Symbol[]
    for pair in 0:15
        i = 2pair + 1
        w0 = Symbol(:w, i); w1 = Symbol(:w, i + 1)
        if pair in EMU
            push!(stmts, :(($w0, $w1) =
                fab_exp_pair_emu(v[$i], v[$(i + 1)], qk2, m2)))
        else
            push!(stmts, :($w0 = ptx"ex2.approx.ftz.f32"(
                fma(reinterpret(Float32, v[$i]), qk_scale, mneg))))
            push!(stmts, :($w1 = ptx"ex2.approx.ftz.f32"(
                fma(reinterpret(Float32, v[$(i + 1)]), qk_scale, mneg))))
        end
        push!(lanes, w0, w1)
    end
    quote
        Base.@inline
        $(stmts...)
        tuple($(lanes...))
    end
end

# exp → pack bf16 → store P quarter → publish BAR_P_Q[C] → fold accs.
# Chunk folds use `strided_reduce` (pyptx chunk_reduce shape): closure-free
# unrolled pairwise trees — an `ntuple ... do` closure of that size escapes
# the inliner and becomes a real device-side call.
# The wait::st round trip hides under the next chunk's in-flight load.
# Under splitp the store stays per-chunk but the publish coarsens to the
# 64-column half: chunks 0/2 store and move on (their st completes under
# the half-mate's exp/pack), chunks 1/3 wait::st (covering BOTH stores),
# fence, and arrive the half barrier — pyptx 4d2aa00's `c == 1 or c == 3`
# publish, with [stage, half] granularity on the consumer side.
@inline function fab_softmax_chunk(::Val{C}, ::Val{SPLITP}, emu::Val,
                                    v::NTuple{32, UInt32},
                                    macc::NTuple{4, Float32}, sacc::NTuple{4, Float32},
                                    sp_base::UInt32, pq_ptr,
                                    qk_scale::Float32, qk2::UInt64,
                                    mneg::Float32) where {C, SPLITP}
    w = fab_exp_chunk(emu, v, qk_scale, mneg, qk2)
    pk = ntuple(k -> bf16x2_pack(w[2k - 1], w[2k]), Val(16))
    ptx"tcgen05.st.sync.aligned.32x32b.x16.b32"(sp_base + UInt32(C * 16), pk)
    if SPLITP
        if C == 1 || C == 3
            ptx"tcgen05.wait::st.sync.aligned"()
            ptx"tcgen05.fence::before_thread_sync"()
            barrier_arrive(pq_ptr + 8 * (C >> 1))
        end
    else
        ptx"tcgen05.wait::st.sync.aligned"()
        ptx"tcgen05.fence::before_thread_sync"()
        barrier_arrive(pq_ptr + 8 * C)
    end
    # fanout 3 on the max fold: NVPTX fuses the ternary groups into
    # sm_100's 3-input max.NaN.f32 (max is exactly associative, so the
    # association change is value-neutral; + keeps the pairwise default)
    cm = strided_reduce(max, w, Val(4), Val(3))
    cs = strided_reduce(+, w, Val(4))
    if C == 0
        return cm, cs
    else
        return ntuple(a -> max(macc[a], cm[a]), Val(4)),
               ntuple(a -> sacc[a] + cs[a], Val(4))
    end
end

# One KV tile of the softmax stream. Returns the updated per-thread state
# (ph_s, ph_stats, mneg, pmax_pend, sums_pend, l_run).
@inline function fab_softmax_body(d::FabDbg, sb::Bool, sp::Val, emu::Val,
                                   first::Bool,
                                   oresc_par_ptr,
                                   sp_base::UInt32, sfull, sfree, pq_ptr,
                                   stats, alpha_idx::Int, stats_bar::UInt32,
                                   qk_scale::Float32, qk2::UInt64,
                                   ph_s::UInt32, ph_stats::UInt32, mneg::Float32,
                                   pmax::Float32, sums::Float32, l_run::Float32)
    ph_s = fab_wait(d, sfull, ph_s, Val(7))
    ptx"tcgen05.fence::after_thread_sync"()

    # first chunk's load flies under the deferred tail
    v0 = ptx"tcgen05.ld.sync.aligned.32x32b.x32.b32"(sp_base)

    if !first
        # ── deferred tail of tile k-1: basis decision from this thread's
        # own row max on the exp'd values (exp2 is monotonic; pmax > 2^T
        # means the basis drifted by more than T scaled-log2 units) ──
        alpha = 1.0f0
        if pmax > FAB_THRESH
            alpha = ptx"rcp.approx.f32"(pmax)
            mneg -= ptx"lg2.approx.f32"(pmax)
        end
        l_run = (l_run + sums) * alpha

        # publish alpha_{k-1}: gates PV(k) at parity k&1. The common-case
        # O_RESC release goes first; the stats-slot publish (paced by
        # correction) follows.
        needs = alpha < 1.0f0
        blt = nvvm"vote.ballot.sync"(0xffffffff, needs)
        if blt == UInt32(0)
            barrier_arrive(oresc_par_ptr)
        end
        ph_stats = fab_wait(d, sfree, ph_stats, Val(8))
        @inbounds stats[alpha_idx] = alpha
        ptx"bar.arrive"(stats_bar, UInt32(64))
    end

    # ── stream 4 chunks: ld(c+1) issues as soon as ld(c) lands (or, in
    # scoreboard mode, immediately — up to two loads in flight, the RAW
    # hazard left to the SASS register scoreboard as FA4 does), and flies
    # under exp/pack/store(c) ──
    v0 = sb ? v0 : PTX.wait_registers(ptx"tcgen05.wait::ld.sync.aligned", v0)
    v1 = ptx"tcgen05.ld.sync.aligned.32x32b.x32.b32"(sp_base + UInt32(32))
    macc, sacc = fab_softmax_chunk(Val(0), sp, emu, v0, ntuple(_ -> 0.0f0, Val(4)),
                                    ntuple(_ -> 0.0f0, Val(4)),
                                    sp_base, pq_ptr, qk_scale, qk2, mneg)
    v1 = sb ? v1 : PTX.wait_registers(ptx"tcgen05.wait::ld.sync.aligned", v1)
    v2 = ptx"tcgen05.ld.sync.aligned.32x32b.x32.b32"(sp_base + UInt32(64))
    macc, sacc = fab_softmax_chunk(Val(1), sp, emu, v1, macc, sacc,
                                    sp_base, pq_ptr, qk_scale, qk2, mneg)
    v2 = sb ? v2 : PTX.wait_registers(ptx"tcgen05.wait::ld.sync.aligned", v2)
    v3 = ptx"tcgen05.ld.sync.aligned.32x32b.x32.b32"(sp_base + UInt32(96))
    macc, sacc = fab_softmax_chunk(Val(2), sp, emu, v2, macc, sacc,
                                    sp_base, pq_ptr, qk_scale, qk2, mneg)
    v3 = sb ? v3 : PTX.wait_registers(ptx"tcgen05.wait::ld.sync.aligned", v3)
    macc, sacc = fab_softmax_chunk(Val(3), sp, emu, v3, macc, sacc,
                                    sp_base, pq_ptr, qk_scale, qk2, mneg)

    # horizontal reduce into the pending carries
    pmax = max(max(macc[1], macc[2], macc[3]), macc[4])
    sums = (sacc[1] + sacc[2]) + (sacc[3] + sacc[4])
    return ph_s, ph_stats, mneg, pmax, sums, l_run
end

# ── correction helpers ──────────────────────────────────────────────────

# Element-wise O ops as top-level helpers: a closure over a variable that
# is reassigned in its enclosing loop gets boxed (dynamic dispatch in
# device code) — arguments don't.
@inline fab_scale64(v::NTuple{64, UInt32}, s::Float32) =
    ntuple(i -> reinterpret(UInt32, reinterpret(Float32, v[i]) * s), Val(64))
# Q as a type parameter: `v[32q + ...]` with a runtime q is a dynamic
# NTuple{64} getindex, which lowers through an alloca that SROA cannot
# remove — the whole 64-register tuple demotes to local memory.
@inline fab_pack16(v::NTuple{64, UInt32}, ::Val{Q}, s::Float32) where {Q} =
    ntuple(Val(16)) do k
        bf16x2_pack(reinterpret(Float32, v[32Q + 2k - 1]) * s,
                    reinterpret(Float32, v[32Q + 2k]) * s)
    end

# One tile of the correction stream for one stage: consume the alpha,
# rescale O in TMEM iff any row in this warp's band needs it. The PV_DONE
# wait happens EVERY tile (correction idles here anyway) so the phase var
# tracks the true completion count — a parity-only wait inside the rare
# branch could alias an older same-parity completion and race the
# in-flight PV.
@inline function fab_corr_tile(d::FabDbg, ::Val{STAGE}, bars::BarrierSet,
                                stats, alpha_idx::Int,
                                stats_bar::UInt32, o_addr::UInt32,
                                ponc::UInt32, ph_pv::UInt32) where {STAGE}
    ptx"bar.sync"(stats_bar, UInt32(64))
    alpha = @inbounds stats[alpha_idx]
    barrier_arrive(bars.stats_free[STAGE])
    ph_pv = fab_wait(d, bars.pv_done[STAGE], ph_pv, Val(10))
    need = alpha < 1.0f0
    blt = nvvm"vote.ballot.sync"(0xffffffff, need)
    # softmax already released O_RESC when nothing rescales
    if blt != UInt32(0)
        ptx"tcgen05.fence::after_thread_sync"()
        for half in 0:1
            addr = o_addr + UInt32(half * 64)
            ov = ptx"tcgen05.ld.sync.aligned.32x32b.x64.b32"(addr)
            ov = PTX.wait_registers(ptx"tcgen05.wait::ld.sync.aligned", ov)
            ptx"tcgen05.st.sync.aligned.32x32b.x64.b32"(addr, fab_scale64(ov, alpha))
        end
        ptx"tcgen05.wait::st.sync.aligned"()
        ptx"tcgen05.fence::before_thread_sync"()
        barrier_arrive(bars.o_resc[STAGE, 0] + Int(ponc))
    end
    return ph_pv
end

# Epilogue for one stage: normalize O by the row sum, pack bf16, store.
@inline function fab_epi_stage(::Val{STAGE}, bars::BarrierSet,
                                stats, alpha_idx::Int,
                                stats_bar::UInt32, o_addr::UInt32,
                                out_row::UInt32, po) where {STAGE}
    ptx"bar.sync"(stats_bar, UInt32(64))          # streaming softmax's row sum
    l = @inbounds stats[alpha_idx + 256]
    inv_l = ptx"rcp.approx.f32"(l)

    row = out_row + UInt32(STAGE * FAB_BM)
    gptr = po + Int(row) * (FAB_HD * 2)
    for half in 0:1
        addr = o_addr + UInt32(half * 64)
        ov = ptx"tcgen05.ld.sync.aligned.32x32b.x64.b32"(addr)
        ov = PTX.wait_registers(ptx"tcgen05.wait::ld.sync.aligned", ov)
        if STAGE == 1 && half == 1
            # the LAST TMEM read is complete: everything the gates protect
            # (the readout) is done, and the remaining normalize/pack/store
            # is register→global work. Release the next item now so it
            # overlaps the store tail.
            barrier_arrive(bars.stats_free[1])
            barrier_arrive(bars.epi)
            barrier_arrive(bars.o_resc[0, 0])
            barrier_arrive(bars.o_resc[1, 0])
        end
        # @unroll (not plain for): pk_q[4vec + 1] must be a constant
        # tuple index by construction, not by trusting LLVM's unroller
        @unroll for q in 0:1
            pk_q = fab_pack16(ov, Val(q), inv_l)
            @unroll for vec in 0:3
                ptx"st.global.v4.b32"(gptr + half * 128 + 64q + vec * 16,
                    (pk_q[4vec + 1], pk_q[4vec + 2],
                     pk_q[4vec + 3], pk_q[4vec + 4]))
            end
        end
    end
    if STAGE == 0
        # stage-0 sums slot consumed: return the credit
        barrier_arrive(bars.stats_free[0])
    end
end

# ── the kernel ──────────────────────────────────────────────────────────

function fab_kernel!(
        O::CuDeviceVector{BFloat16, 1},
        tma_Q::PTX.TMADescriptorPtr,
        tma_K::PTX.TMADescriptorPtr,
        tma_V::PTX.TMADescriptorPtr,
        seqlen::UInt32, mb_shift::UInt32, mb_mask::UInt32,
        n_tiles::UInt32, total_work::UInt32,
        qk_scale::Float32,
        dbg::CuDeviceVector{UInt32, 1},   # 512 slots; only written by beacon
        ::Val{CFG}) where {CFG}

    # @inbounds: CuDynamicSharedArray's @boundscheck plants
    # gpu_report_exception cold paths (and their register save/restore) in
    # the entry region. Not launch-critical — the kernel passes on
    # hardware under Pkg.test's --check-bounds=yes, i.e. with these calls
    # present, so `.reqntid` alone is what makes ptxas honor setmaxnreg —
    # but the checks are dead weight in a hand-verified SMEM layout.
    smem_q    = @inbounds CuDynamicSharedArray(BFloat16, FAB_Q_BYTES ÷ 2, FAB_SMEM_Q)
    smem_kv   = @inbounds CuDynamicSharedArray(BFloat16, FAB_KV_SLOTS * FAB_TILE_BYTES ÷ 2,
                                               FAB_SMEM_KV)
    stats     = @inbounds CuDynamicSharedArray(Float32, 512, FAB_SMEM_STATS)
    bar_mem   = @inbounds CuDynamicSharedArray(UInt64, FAB_BAR_BYTES ÷ 8, FAB_SMEM_BARS)
    tmem_slot = @inbounds CuDynamicSharedArray(UInt32, 1, FAB_SMEM_TMEM)

    q_ptr  = pointer(smem_q)
    kv_ptr = pointer(smem_kv)
    bars   = fab_barset(Val(CFG.splitp), pointer(bar_mem))

    tid      = ptx"mov.u32"(sreg"tid.x")
    cta_id   = ptx"mov.u32"(sreg"ctaid.x")
    num_ctas = ptx"mov.u32"(sreg"nctaid.x")
    warp   = tid >> UInt32(5)
    wig    = warp & UInt32(3)                     # warp in group
    row128 = tid & UInt32(127)
    lane_bits = (tid & UInt32(96)) << UInt32(16)  # TMEM lane bits: (warp%4)*32

    d  = FabDbg{CFG.beacon}(pointer(dbg), tid)
    sb = CFG.scoreboard

    # ── init barriers, allocate TMEM ──
    if tid == UInt32(0)
        barrierset_init!(bars)
    end
    if warp == UInt32(12)
        ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(
            smem_addr_u32(pointer(tmem_slot)), UInt32(512))
    end
    ptx"bar.sync"(Val(0))
    tmem = @inbounds tmem_slot[1]

    # =====================================================================
    # TMA load warp (single lane; warpgroup 12-15 takes NREG_OTHER)
    # =====================================================================
    if tid >= UInt32(384)
        fab_maxnreg_inc(Val(CFG.nreg[3]))
    end

    if tid == FAB_LOAD_TID
        # self-credit the ring and the Q buffer once: every load below
        # (including each work item's prologue) waits FREE first, so the
        # accounting is uniform across persistent iterations
        for s in 0:(FAB_KV_SLOTS - 1)
            barrier_arrive(bars.kv_free[s])
        end
        barrier_arrive(bars.q_free)
        @unroll for s in 0:3
            ph_s = UInt32(0)
        end
        q_free_ph = UInt32(0)

        lw = cta_id
        while lw < total_work
            q_row0 = fab_q_row0(lw, seqlen, mb_shift, mb_mask)
            kv_row = (lw >> mb_shift) * seqlen
            krow = kv_row
            vrow = kv_row

            # prologue, ordered by first use: QK(stage 0) needs only
            # Q0+K0, so K0 goes ahead of the second Q tile
            q_free_ph = fab_wait(d, bars.q_free, q_free_ph, Val(2))
            barrier_arrive_expect_tx(bars.q[0], FAB_TILE_BYTES)
            fab_load_tile(q_ptr, 0, tma_Q, q_row0, bars.q[0])
            krow, ph_0 = fab_load_kv(d, Val(0), kv_ptr, tma_K, krow, bars, ph_0)
            barrier_arrive_expect_tx(bars.q[1], FAB_TILE_BYTES)
            fab_load_tile(q_ptr, FAB_TILE_BYTES, tma_Q, q_row0 + UInt32(FAB_BM),
                           bars.q[1])
            @unroll for (s, tma, row) in ((1, tma_V, vrow), (2, tma_K, krow),
                                            (3, tma_V, vrow))
                row, ph_s = fab_load_kv(d, Val(s), kv_ptr, tma, row, bars, ph_s)
            end

            # steady state: 4 loads per trip into slots 0..3, K/V alternating
            trips = (n_tiles - UInt32(2)) >> 1
            trip = UInt32(0)
            while trip < trips
                @unroll for (s, tma, row) in ((0, tma_K, krow), (1, tma_V, vrow),
                                                (2, tma_K, krow), (3, tma_V, vrow))
                    row, ph_s = fab_load_kv(d, Val(s), kv_ptr, tma, row, bars, ph_s)
                end
                trip += UInt32(1)
            end
            lw += num_ctas
        end
    end

    # =====================================================================
    # MMA dispatch warp (single thread)
    # =====================================================================
    if tid == FAB_MMA_TID
        idesc_qk = tcgen05_instr_desc_f16bf16_f32(
            m = 128, n = FAB_BN, ab_dtype = :bf16, a_major = :K, b_major = :K)
        idesc_pv = tcgen05_instr_desc_f16bf16_f32(
            m = 128, n = FAB_HD, ab_dtype = :bf16, a_major = :K, b_major = :MN)
        # K-major operands: pyptx masked_descriptor ≡ leading 16 / stride
        # 1024 / B128 (see tcgen05_descriptor.jl's tested equivalence).
        dq0 = tcgen05_descriptor(smem_addr_u32(q_ptr);
                                 leading_bytes = 16, stride_bytes = 1024,
                                 swizzle = BlackwellLayout.B128)
        dk0 = tcgen05_descriptor(smem_addr_u32(kv_ptr);
                                 leading_bytes = 16, stride_bytes = 1024,
                                 swizzle = BlackwellLayout.B128)
        dv0 = tcgen05_descriptor(smem_addr_u32(kv_ptr) + UInt32(FAB_TILE_BYTES);
                                 leading_bytes = 16384, stride_bytes = 1024,
                                 swizzle = BlackwellLayout.B128)

        sp = Val(CFG.splitp)
        kph0 = UInt32(0); kph1 = UInt32(0); kph2 = UInt32(0); kph3 = UInt32(0)
        ph_qa = UInt32(0); ph_qb = UInt32(0)
        ph_pq0 = UInt32(0); ph_pq1 = UInt32(0)      # all quarters/halves flip together
        ph_or00 = UInt32(0); ph_or01 = UInt32(0)    # [stage][parity]
        ph_or10 = UInt32(0); ph_or11 = UInt32(0)

        mw = cta_id
        while mw < total_work
            # ── prologue: S(0) for stage 0 as soon as Q0+K0 land; the
            # second Q tile's TMA overlaps the first QK ──
            ph_qa = fab_wait(d, bars.q[0], ph_qa, Val(3))
            kph0 = fab_wait(d, bars.kv_full[0], kph0, Val(4))
            fab_qk_mma(Val(0), Val(0), tmem, dq0, dk0, idesc_qk, bars)
            ph_qb = fab_wait(d, bars.q[1], ph_qb, Val(3))
            fab_qk_mma(Val(1), Val(0), tmem, dq0, dk0, idesc_qk, bars)
            fab_commit(bars.kv_free[0])

            # ── peeled j=0: PV(0) with V slot 1, S(1) with K slot 2 ──
            kph1 = fab_wait(d, bars.kv_full[1], kph1, Val(4))
            kph2 = fab_wait(d, bars.kv_full[2], kph2, Val(4))
            ph_pq0, ph_or00 = fab_pv_mma(d, sp, Val(0), Val(0), true, Val(0),
                                          tmem, dv0, idesc_pv, bars, ph_pq0, ph_or00)
            fab_qk_mma(Val(0), Val(1), tmem, dq0, dk0, idesc_qk, bars)
            ph_pq1, ph_or10 = fab_pv_mma(d, sp, Val(1), Val(0), true, Val(0),
                                          tmem, dv0, idesc_pv, bars, ph_pq1, ph_or10)
            fab_qk_mma(Val(1), Val(1), tmem, dq0, dk0, idesc_qk, bars)
            fab_commit(bars.kv_free[2])
            fab_commit(bars.kv_free[1])

            # ── steady: j = 1 .. n_tiles-2, two iterations per trip. KV
            # waits are hoisted ahead of the PV/QK pair: KV normally
            # arrives early while P quarters are the tight resource, so
            # the QK UMMAs queue behind the PVs and keep the tensor core
            # fed through quarter stalls. ──
            trips = (n_tiles - UInt32(2)) >> 1
            trip = UInt32(0)
            while trip < trips
                # j odd: V in slot 3, next K in slot 0
                kph3 = fab_wait(d, bars.kv_full[3], kph3, Val(4))
                kph0 = fab_wait(d, bars.kv_full[0], kph0, Val(4))
                ph_pq0, ph_or01 = fab_pv_mma(d, sp, Val(0), Val(1), false, Val(1),
                                              tmem, dv0, idesc_pv, bars, ph_pq0, ph_or01)
                fab_qk_mma(Val(0), Val(0), tmem, dq0, dk0, idesc_qk, bars)
                ph_pq1, ph_or11 = fab_pv_mma(d, sp, Val(1), Val(1), false, Val(1),
                                              tmem, dv0, idesc_pv, bars, ph_pq1, ph_or11)
                fab_qk_mma(Val(1), Val(0), tmem, dq0, dk0, idesc_qk, bars)
                fab_commit(bars.kv_free[0])
                fab_commit(bars.kv_free[3])
                # j even: V in slot 1, next K in slot 2
                kph1 = fab_wait(d, bars.kv_full[1], kph1, Val(4))
                kph2 = fab_wait(d, bars.kv_full[2], kph2, Val(4))
                ph_pq0, ph_or00 = fab_pv_mma(d, sp, Val(0), Val(0), false, Val(0),
                                              tmem, dv0, idesc_pv, bars, ph_pq0, ph_or00)
                fab_qk_mma(Val(0), Val(1), tmem, dq0, dk0, idesc_qk, bars)
                ph_pq1, ph_or10 = fab_pv_mma(d, sp, Val(1), Val(0), false, Val(0),
                                              tmem, dv0, idesc_pv, bars, ph_pq1, ph_or10)
                fab_qk_mma(Val(1), Val(1), tmem, dq0, dk0, idesc_qk, bars)
                fab_commit(bars.kv_free[2])
                fab_commit(bars.kv_free[1])
                trip += UInt32(1)
            end

            # all QKs for this work item are issued: release the Q buffer
            # so the next item's Q TMA overlaps the final PV and the
            # epilogue
            fab_commit(bars.q_free)

            # ── peeled j = n_tiles-1 (odd): PV only, V in slot 3 ──
            kph3 = fab_wait(d, bars.kv_full[3], kph3, Val(4))
            ph_pq0, ph_or01 = fab_pv_mma(d, sp, Val(0), Val(1), false, Val(1),
                                          tmem, dv0, idesc_pv, bars, ph_pq0, ph_or01)
            ph_pq1, ph_or11 = fab_pv_mma(d, sp, Val(1), Val(1), false, Val(1),
                                          tmem, dv0, idesc_pv, bars, ph_pq1, ph_or11)
            fab_commit(bars.kv_free[3])
            fab_commit(bars.o_done)

            mw += num_ctas
        end

        # tail padding: correction's PV_DONE parity waits may lag the
        # completion stream while idling through skip tiles; extra
        # completions per stage guarantee every pending parity test still
        # sees a phase flip after the last real PV
        for _ in 1:8
            fab_commit(bars.pv_done[0])
            fab_commit(bars.pv_done[1])
        end
    end

    # =====================================================================
    # Softmax: two stage-parallel warpgroups (warps 0-3 → stage 0, warps
    # 4-7 → stage 1). Thread t owns row t&127 and streams its 128 columns
    # in four 32-column chunks, ping-ponging two register buffers so each
    # chunk's TMEM load flies under the previous chunk's exp.
    # =====================================================================
    if tid < UInt32(256)
        fab_maxnreg_inc(Val(CFG.nreg[1]))
        stage = (tid & UInt32(128)) >> UInt32(7)

        sp_base = tmem + lane_bits + (stage << UInt32(7))
        sfull   = bars.s_full[stage]
        sfree   = bars.stats_free[stage]
        pq_ptr  = bars.p_q[stage, 0]
        oresc   = bars.o_resc[stage, 0]
        stats_bar = UInt32(1) + wig + (stage << UInt32(2))  # named bar 1+w / 5+w
        alpha_idx = Int(stage) * 128 + Int(row128) + 1
        sum_idx   = alpha_idx + 256

        sp  = Val(CFG.splitp)
        emu = Val(CFG.emu)
        qk2 = fab_pack2(qk_scale, qk_scale)
        ph_s = UInt32(0); ph_stats = UInt32(0); epi_ph = UInt32(0)
        pmax = 0.0f0; sums = 0.0f0

        sw = cta_id
        while sw < total_work
            # gate the item start on the previous epilogue's O readout
            epi_ph = fab_wait(d, bars.epi, epi_ph, Val(9))

            mneg = 0.0f0
            l_run = 0.0f0
            ph_s, ph_stats, mneg, pmax, sums, l_run = fab_softmax_body(
                d, sb, sp, emu, true, oresc, sp_base, sfull, sfree, pq_ptr,
                stats, alpha_idx,
                stats_bar, qk_scale, qk2, ph_s, ph_stats, mneg, pmax, sums, l_run)
            it = UInt32(1)
            while it < n_tiles
                po = (it & UInt32(1)) << UInt32(3)
                ph_s, ph_stats, mneg, pmax, sums, l_run = fab_softmax_body(
                    d, sb, sp, emu, false, oresc + Int(po), sp_base, sfull, sfree,
                    pq_ptr,
                    stats, alpha_idx, stats_bar, qk_scale, qk2,
                    ph_s, ph_stats, mneg, pmax, sums, l_run)
                it += UInt32(1)
            end

            # drain: fold the last tile's sums into l (its basis decision
            # never runs — the pending rescale cancels between O and l in
            # the epilogue normalize), publish the row sum,
            # STATS_FREE-gated like every body publish.
            l_run += sums
            @inbounds stats[sum_idx] = l_run
            ph_stats = fab_wait(d, sfree, ph_stats, Val(8))
            ptx"bar.arrive"(stats_bar, UInt32(64))

            sw += num_ctas
        end
    end

    # =====================================================================
    # Correction warps (tids 256-383): O rescale + epilogue
    # =====================================================================
    if UInt32(256) <= tid < UInt32(384)
        fab_maxnreg_dec(Val(CFG.nreg[2]))
        # per-stage state (suffix-minted): named bars 1+w / 5+w, alpha
        # slots at stage*128, O at TMEM cols 256 + stage*128
        @unroll for st in 0:1
            stats_bar_st = UInt32(1 + 4st) + wig
            alpha_idx_st = 128st + Int(row128) + 1
            o_addr_st    = tmem + UInt32(256 + 128st) + lane_bits
            ph_pv_st     = UInt32(0)
        end
        done_ph = UInt32(0)
        po = pointer(O)

        # first KV tile needs no correction: release both stages once
        # (parity-0 barrier gates PV(0)); also pre-arrive STATS_FREE so
        # the softmax's pre-write wait at tile 1 is already satisfied
        @unroll for st in 0:1
            barrier_arrive(bars.o_resc[st, 0])
            barrier_arrive(bars.stats_free[st])
        end
        barrier_arrive(bars.epi)

        cw = cta_id
        while cw < total_work
            # alpha published at tile i (i = 0..n_tiles-2; the last tile
            # publishes nothing) rescales O between PV(i) and PV(i+1)
            it = UInt32(0)
            while it < n_tiles - UInt32(1)
                ponc = ((it + UInt32(1)) & UInt32(1)) << UInt32(3)
                @unroll for st in 0:1
                    ph_pv_st = fab_corr_tile(d, Val(st), bars, stats, alpha_idx_st,
                                              stats_bar_st, o_addr_st, ponc, ph_pv_st)
                end
                it += UInt32(1)
            end

            # drain: consume the last tile's PV_DONE completion so the
            # per-work-item wait/completion counts stay exactly balanced
            @unroll for st in 0:1
                ph_pv_st = fab_wait(d, bars.pv_done[st], ph_pv_st, Val(10))
            end

            # ── epilogue: wait all PV done, normalize, store bf16 ──
            done_ph = fab_wait(d, bars.o_done, done_ph, Val(11))
            ptx"tcgen05.fence::after_thread_sync"()

            out_row = fab_q_row0(cw, seqlen, mb_shift, mb_mask) + row128
            @unroll for st in 0:1
                fab_epi_stage(Val(st), bars, stats, alpha_idx_st, stats_bar_st,
                               o_addr_st, out_row, po)
            end

            cw += num_ctas
        end
    end

    ptx"bar.sync"(Val(0))
    if warp == UInt32(12)
        ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(tmem, UInt32(512))
        ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    end
    return nothing
end


# ── host-side helpers (shared by the tests and the bench workbench) ─────

# f32 reference with bf16-quantized inputs, per (batch*head) slice.
function fab_cpu_ref(Q::Matrix{Float32}, K::Matrix{Float32},
                      V::Matrix{Float32}, sm_scale::Float32)
    Qb = Float32.(BFloat16.(Q))
    Kb = Float32.(BFloat16.(K))
    Vb = Float32.(BFloat16.(V))
    scores = (Qb * Kb') .* sm_scale
    rmax = maximum(scores; dims = 2)
    ex = exp.(scores .- rmax)
    return (ex ./ sum(ex; dims = 2)) * Vb
end

# (rows, 128) f32 → BFloat16, row-major (HD-fast) for TMA/global IO.
function fab_pack(X::Matrix{Float32})
    rows, hd = size(X)
    out = Array{BFloat16}(undef, hd, rows)
    for r in 1:rows, h in 1:hd
        @inbounds out[h, r] = BFloat16(X[r, h])
    end
    out
end

# Decode a beacon buffer (one UInt32 per tid; 0 = never timed out).
function fab_decode_beacons(dbg::Vector{UInt32})
    hits = [(tid = t - 1, site = Int((v >> 8) & 0xff), phase = Int(v & 0xff))
            for (t, v) in enumerate(dbg) if v != 0]
    isempty(hits) && return nothing
    by_site = Dict{String, Vector{Int}}()
    for h in hits
        push!(get!(by_site, get(FAB_SITES, h.site, "site$(h.site)"), Int[]), h.tid)
    end
    by_site
end
