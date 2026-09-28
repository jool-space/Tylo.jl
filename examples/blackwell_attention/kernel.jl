using Tylo, PTX, CUDACore, BFloat16s
using Tylo.Layouts: Layout, static
using Tylo: tmem_allocate!, tmem_deallocate!, fence_after_thread_sync, fence_before_thread_sync

# FlashAttention's forward pass on datacenter Blackwell, O = softmax(Q Kᵀ ·
# scale) V, in the style of FlashAttention-4. Q, K, V and O are
# d×seq×(batch·heads) BF16 arrays (d contiguous), d = 128, as for the Hopper
# kernel.
#
# A persistent CTA of sixteen warps computes 256 query rows at a time, as two
# tiles of 128 whose work interleaves on the tensor cores:
#
#   warps 0-3, 4-7   softmax of tile 0, tile 1: thread t owns row t, reads its
#                    128 scores from TMEM, and writes the probabilities P back
#                    over them in BF16;
#   warps 8-11       correction: rescales O in TMEM when a row's maximum has
#                    grown, and finally normalizes and stores O;
#   warp 12          MMA: one thread issues S = Q Kᵀ and O += P V;
#   warp 13          loads: one thread streams Q, then K and V blocks of 128
#                    keys through a ring of shared stages.
#
# TMEM holds S of both tiles (columns 0-255, P over the second half of each)
# and O of both (256-511). Per tile, the MMA thread issues O += P V for the
# previous block, then S = Q Kᵀ for this one into the same columns (the
# tensor pipe runs in order), and commits `sfull`: once S is ready and O
# final up to the previous block. The softmax takes the new row maxima,
# hands the rescale factors to the correction warps through shared memory
# and a named barrier, writes P in four 32-key chunks, and arrives on
# `pready` after three of them, so that most of the next P V runs while it
# writes the last (`plast`). A row's maximum only moves when it grows by
# more than 2^8 (the probabilities stay below 256), so rescales are rare.
#
# With `CTAs = 2`, the two CTAs of a cluster compute 512 rows together, each
# its own two tiles, and every product is one CTA-pair instruction (M = 256)
# that the even CTA's MMA thread issues: each CTA holds half of a K block (64
# keys) and half of a V block (64 of d), so a stage is half as large and the
# ring twice as deep. Both CTAs' loads complete on the even CTA's barriers;
# commits arrive in both; the softmax and correction warps of both CTAs
# release P and O to the MMA thread through the even CTA's `paired` and
# `pairlast` barriers.
# A thread's row of a fragment in the row-per-lane ownership.
@inline _rowmax(x) = maximum(x;dims=2).data[1]
@inline _rowsum(x) = sum(x;dims=2).data[1]
@inline _exponentials(x,c,m) = Tylo._ex2.(muladd.(x,c,-m))
# `x`, once `token` exists: a data dependence the compilers cannot see through.
@inline _after(token::UInt64,x) = ifelse(token == typemax(UInt64),zero(x),x)
# A row's sum as one chain from `init`: every addition waits for `init`.
# Pairs of values add at once (Tylo's FP32 pairs), then the two lanes.
@generated function _chainsum(init,x::Fragment{Float32,N}) where N
    chain = :(Tylo.F32x2(init,0f0))
    for k in 1:N÷2
        chain = :($chain+Tylo.F32x2(d[$(2k-1)],d[$(2k)]))
    end
    :(Base.@inline; d = x.data; (a,b) = Tylo._lanes($chain); a+b)
end
# A chunk of a row's scores, 32 keys from `first`, with keys past `row` masked.
@inline _mask(x,first::Int32,row::Int32) = Fragment(ntuple(e -> ifelse(first+Int32(e-1) <= row,x.data[e],-Inf32),Val(32)),Tylo.Layouts.layout(x))
# P's first `Split` chunks and the rest, from the row's four.
# exp2 of a pair on the FMA pipe (pyptx's degree-3 polynomial with the
# integer part rounded off and added into the exponent), as FlashAttention-4
# computes some of each row's to relieve MUFU.
@inline _pack2(lo::Float32,hi::Float32) = (UInt64(reinterpret(UInt32,hi)) << 32) | UInt64(reinterpret(UInt32,lo))
@inline function _ex2_pair(x0::Float32,x1::Float32)
    big = _pack2(reinterpret(Float32,0x4B400000),reinterpret(Float32,0x4B400000))
    x = _pack2(ptx"max.ftz.f32"(x0,-127f0),ptx"max.ftz.f32"(x1,-127f0))
    bi = ptx"add.rm.ftz.f32x2"(x,big)
    f = ptx"sub.rn.ftz.f32x2"(x,ptx"sub.rn.ftz.f32x2"(bi,big))
    c3, c2, c1 = reinterpret(Float32,0x3D9DF09D), reinterpret(Float32,0x3E6906A4), reinterpret(Float32,0x3F31F519)
    q = ptx"fma.rn.ftz.f32x2"(f,_pack2(c3,c3),_pack2(c2,c2))
    q = ptx"fma.rn.ftz.f32x2"(q,f,_pack2(c1,c1))
    q = ptx"fma.rn.ftz.f32x2"(q,f,_pack2(1f0,1f0))
    reinterpret(Float32,((bi % UInt32) << 23)+(q % UInt32)), reinterpret(Float32,(((bi >> 32) % UInt32) << 23)+((q >> 32) % UInt32))
end
# A chunk's exponentials, the pairs in `Emulated` (0-15) on the FMA pipe.
@generated function _exponentials(x::Fragment{Float32,32},c::Float32,m::Float32,::Val{Emulated}) where Emulated
    isempty(Emulated) && return :(Base.@inline; _exponentials(x,c,m))
    stmts = Any[:(y = muladd.(x,c,-m)), :(v = y.data)]
    names = Symbol[]
    for pair in 0:15
        a, b = Symbol(:e,2pair+1), Symbol(:e,2pair+2)
        push!(stmts,pair in Emulated ? :(($a,$b) = _ex2_pair(v[$(2pair+1)],v[$(2pair+2)])) :
                                       :(($a,$b) = (Tylo._ex2(v[$(2pair+1)]),Tylo._ex2(v[$(2pair+2)]))))
        push!(names,a,b)
    end
    :(Base.@inline; $(stmts...); Fragment(($(names...),),Tylo.Layouts.layout(x)))
end
# P's first `Split` chunks and the rest: the middle two chunks with
# `Emulated` pairs on the FMA pipe.
@inline _chunk(s,q,c,m,e) = _exponentials(s[q],c,m,q in (2,3) ? e : Val(()))
@inline _first(s,c,m,::Val{1},e) = (_chunk(s,1,c,m,e),)
@inline _first(s,c,m,::Val{2},e) = (_chunk(s,1,c,m,e),_chunk(s,2,c,m,e))
@inline _first(s,c,m,::Val{3},e) = (_chunk(s,1,c,m,e),_chunk(s,2,c,m,e),_chunk(s,3,c,m,e))
@inline _rest(s,c,m,::Val{1},e) = (_chunk(s,2,c,m,e),_chunk(s,3,c,m,e),_chunk(s,4,c,m,e))
@inline _rest(s,c,m,::Val{2},e) = (_chunk(s,3,c,m,e),_chunk(s,4,c,m,e))
@inline _rest(s,c,m,::Val{3},e) = (_chunk(s,4,c,m,e),)
# Chunks of P into TMEM, 32 keys each, the first at key 32`First`.
@generated function _store_chunks!(t,band,chunks::NTuple{N,Any},::Val{First}) where {N,First}
    stores = [:(@inbounds store_async!(window(t,(band,Int32($(32(First+q)))),(32,32)),Fragment{BFloat16}(chunks[$(q+1)]))) for q in 0:N-1]
    :(Base.@inline; $(stores...); nothing)
end

function blackwell_attention_kernel!(o,tq,tk,tv,counter,seq_kv::Int32,heads::Int32,scale::Float32,::Val{Stages},::Val{CTAs}=Val(1),
                                     ::Val{Split}=Val(3),::Val{Causal}=Val(false),::Val{Dynamic}=Val(Causal),
                                     ::Val{Emulated}=Val(()),timeline=nothing) where {Stages,CTAs,Split,Causal,Dynamic,Emulated}
    qk = Tcgen05Atom((128CTAs,128,16),BFloat16)             # S = Q Kᵀ: 128 rows per CTA × 128 keys
    pv = Tcgen05Atom((128CTAs,128,16),BFloat16)             # O += P V: 128 rows per CTA × d
    qbytes, kvbytes = transfer_bytes(tq), transfer_bytes(tk)  # a Q tile; this CTA's share of a K or V block
    workspace = @inbounds CuDynamicSharedArray(UInt8,192+16Stages+3072+1024+2qbytes+Stages*kvbytes)
    raw = pointer(workspace)
    @inline qfull(i) = Barrier(raw,i)                       # tile i's queries have arrived
    @inline qempty(i) = Barrier(raw+16,i)                   # tile i's queries may be replaced
    @inline sfull(i) = Barrier(raw+32,i)                    # S of tile i is ready (and O to the previous block)
    @inline pready(i) = Barrier(raw+48,i)                   # P's first 96 keys are written and O rescaled
    @inline plast(i) = Barrier(raw+64,i)                    # P's last 32 keys are written
    @inline ofull(i) = Barrier(raw+80,i)                    # O of tile i is final
    @inline summed(i,b) = Barrier(raw+96,i+2b)              # the row sums of tile i in buffer b are written
    @inline paired(i) = Barrier(raw+128,i)                  # pready of both CTAs (the even CTA's)
    @inline pairlast(i) = Barrier(raw+144,i)                # plast of both CTAs (the even CTA's)
    @inline itemready(b) = Barrier(raw+160,b)               # item slot b holds the next item (`Dynamic`)
    @inline kvfull(s) = Barrier(raw+176,s)                  # stage s holds a K or V block
    @inline kvempty(s) = Barrier(raw+176+8Stages,s)         # stage s may be refilled
    slot = reinterpret(Core.LLVMPtr{UInt32,PTX.AS.Shared},raw+176+16Stages)
    ids = reinterpret(Core.LLVMPtr{Int32,PTX.AS.Shared},raw+180+16Stages)            # [slot]: an item, or -1
    factors = reinterpret(Core.LLVMPtr{Float32,PTX.AS.Shared},raw+192+16Stages)      # [row, tile]: rescale factor
    # [row, tile, item parity]: row sums, two items' worth, since the softmax
    # may finish an item of one block before the correction has read the last.
    sums = factors+1024
    start = raw+192+16Stages+3072
    base = start+((UInt32(0)-PTX.smem_addr_u32(start)) & UInt32(1023))
    @inline qstage(i) = base+i*Int32(qbytes)
    @inline kvstage(s) = base+Int32(2qbytes)+s*Int32(kvbytes)
    tid = Int32(threadIdx().x)-Int32(1)
    warp = warp_uniform(tid ÷ Int32(32))
    lane = tid%Int32(32)
    if tid == Int32(0)
        for i in Int32(0):Int32(1)
            initialize!(qfull(i),1)
            initialize!(qempty(i),1)
            initialize!(sfull(i),1)
            initialize!(pready(i),CTAs == 1 ? 128+4 : 128)  # the softmax threads (and correction warps)
            initialize!(plast(i),128)
            initialize!(ofull(i),1)
            initialize!(summed(i,0),128)
            initialize!(summed(i,1),128)
            initialize!(paired(i),8CTAs)                    # softmax and correction warps of both CTAs
            initialize!(pairlast(i),4CTAs)
            initialize!(itemready(i),1)
        end
        for s in Int32(0):Int32(Stages-1)
            initialize!(kvfull(s),1)
            initialize!(kvempty(s),1)
        end
        fence_barrier_init()
    end
    warp == Int32(12) && tmem_allocate!(qk,slot,512)
    fence_before_thread_sync()
    CTAs == 1 ? sync_threads() : cluster_sync()
    fence_after_thread_sync()
    tmem = unsafe_load(slot)
    rank = CTAs == 1 ? Int32(0) : cluster_rank()
    # Commits arrive in both CTAs of a pair; loads complete on the even CTA's
    # barriers; P and O are released to the even CTA's MMA thread.
    @inline commit(b) = CTAs == 1 ? commit_mma(b) : commit_mma(qk,b,UInt16(0b11))
    @inline completion() = CTAs == 1 ? nothing : 0
    @inline function release_to_mma(local_barrier,pair_barrier)
        sync_warp()
        lane == Int32(0) && (CTAs == 1 ? arrive!(local_barrier) : arrive!(pair_barrier;cta=0))
    end
    # The whole allocation, as FP32 columns and as BF16 halves of them; S, P
    # and O of each tile are windows of it, so that one register holds TMEM.
    words = TmemTile(Float32,tmem,Layout((static(128),static(512)),(static(1),static(128))))
    halves = TmemTile(BFloat16,tmem,Layout((static(128),static(1024)),(static(1),static(128))))
    @inline scores(i) = @inbounds window(words,(0,128i),(128,128))
    @inline probabilities(i) = @inbounds window(halves,(0,256i+128),(128,128))    # over S's second half
    @inline output(i) = @inbounds window(words,(0,256+128i),(128,128))
    seq = size(o,2) % Int32
    nq = seq ÷ Int32(256CTAs)                               # query blocks of a CTA (pair)
    tiles = nq*heads
    first_tile = (Int32(blockIdx().x)-Int32(1)) ÷ Int32(CTAs)
    stride = gridDim().x ÷ Int32(CTAs)
    blocks = seq_kv ÷ Int32(128)
    # Item `t`: its head, first query row and key blocks. Causal items walk
    # each head's query blocks longest first, and stop at their diagonal:
    # the last block is the second tile's alone.
    # Causal items go longest first across heads, not within each: heads in
    # sections whose K and V fit in L2 (about 50 MB), and within a section
    # every head's longest block before any shorter one. (Head by head, a
    # long item would still be handed out near the end.)
    fit = max(Int32(1),unchecked_div(Int32(50_000_000),Int32(512)*seq_kv))
    section = min(Int32(1) << (Int32(31)-leading_zeros(fit) % Int32),heads)
    @inline function item(t)
        if Causal
            per = section*nq
            s = unchecked_div(t,per)
            within = t-s*per
            width = min(section,heads-s*section)
            head = s*section+unchecked_rem(within,width)
            block = nq-Int32(1)-unchecked_div(within,width)
        else
            head, block = unchecked_div(t,nq), unchecked_rem(t,nq)
        end
        head, block*Int32(256CTAs)+rank*Int32(256)
    end
    @inline count(q0) = Causal ? (q0 ÷ Int32(128))+Int32(2) : blocks
    # Tile i's blocks: a causal item's first tile ends a block before the
    # second, at its own diagonal.
    @inline tile_count(q0,i) = count(q0)-(Causal && i == 0 ? Int32(1) : Int32(0))
    # The items: every `stride`th from `first_tile`, or (`Dynamic`) handed out
    # by the load thread from a global counter, longest first, through two
    # item slots; -1 ends the work.
    @inline next_item(t,n) = Dynamic ? (wait(itemready(n&Int32(1)),(n>>1)&Int32(1)); unsafe_load(ids,(n&Int32(1))+Int32(1),Val(4))) : t
    @inline more(t) = Int32(0) <= t < tiles
    c2 = scale*1.442695f0                                   # scores to base-2 exponents (log2 e)
    band = (warp%Int32(4))*Int32(32)                        # the TMEM lanes this warp may access
    row = band+lane
    # Named barriers 1-8 pair softmax warp (tile i, band b) with correction warp b.
    @inline handoff(i) = Int32(1)+Int32(4)*i+warp%Int32(4)
    rows = distribute((32,32),32;vector=32,axis=2)          # TMEM transfers: lane t holds row t, 32 columns
    # With a `Timeline`, each role's first lane records its steps (recorder =
    # warp): a softmax warp at slots 6k.. of its kth block, events 1 waiting
    # for S, 2 S ready, 3 S read, 4 P published but the last chunk, 5 all,
    # 6 summed; a correction warp at 3s.. of its sth step, 10+i waiting for
    # tile i's O, 12+i the factor, 14+i O released; the MMA thread at
    # 8k+4i.. of tile i's kth P V, 20+i waiting for P, 22+i the first part
    # issued, 24+i the rest, 26+i the next S issued; the load thread at 2x..
    # of stage fill x, 30 waiting for the stage, 31 free.
    @inline mark(slot,event) = (timeline === nothing || (lane == Int32(0) && record!(timeline,warp,slot,event)); nothing)
    if warp < Int32(8)
        acquire_registers(Val(192))
        i = warp ÷ Int32(4)
        k = Int32(0); n = Int32(0)                          # blocks and items so far
        t = first_tile
        while true
            t = next_item(t,n)
            more(t) || break
            m = -Inf32; l = 0f0
            _, q0 = item(t)
            nb = tile_count(q0,i)
            for j in Int32(0):nb-Int32(1)
                mark(6k,1)
                wait(sfull(i),k&Int32(1))
                mark(6k+Int32(1),2)
                fence_after_thread_sync()
                # The row in four loads and one wait.
                pending = @inbounds (load_async(rows,window(scores(i),(band,Int32(0)),(32,32))),
                                     load_async(rows,window(scores(i),(band,Int32(32)),(32,32))),
                                     load_async(rows,window(scores(i),(band,Int32(64)),(32,32))),
                                     load_async(rows,window(scores(i),(band,Int32(96)),(32,32))))
                s = (wait_load(pending[1]),wait_load(pending[2]),wait_load(pending[3]),wait_load(pending[4]))
                mark(6k+Int32(2),3)
                if Causal && j >= count(q0)-Int32(2)        # keys past this row's diagonal do not count
                    r = q0+Int32(128)*i+row
                    key = Int32(128)*j
                    s = (_mask(s[1],key,r),_mask(s[2],key+Int32(32),r),_mask(s[3],key+Int32(64),r),_mask(s[4],key+Int32(96),r))
                end
                top = max(_rowmax(s[1]),_rowmax(s[2]),_rowmax(s[3]),_rowmax(s[4]))*c2
                factor = 1f0
                if j == Int32(0) || top-m > 8f0             # the maximum moves: rescale O and the sum
                    j > Int32(0) && (factor = Tylo._ex2(m-top))
                    m = top
                end
                p = _first(s,c2,m,Val(Split),Val(Emulated))
                if j > Int32(0)
                    unsafe_store!(factors,factor,row+Int32(128)*i+Int32(1),Val(4))
                    ptx"bar.arrive"(handoff(i)%UInt32,UInt32(64))
                    l *= factor
                end
                # `Split` chunks of P, published; then the rest and the row
                # sum. Nothing orders pure arithmetic after a store or an
                # arrival, so the work that must wait takes an input through
                # the arrival's token: the compilers cannot hoist it above.
                _store_chunks!(probabilities(i),band,p,Val(0))
                wait_stores()
                fence_before_thread_sync()
                token = ptx"mbarrier.arrive.shared.b64"(pready(i).ptr)
                CTAs == 1 || release_to_mma(pready(i),paired(i))
                mark(6k+Int32(3),4)
                basis = _after(token,m)
                rest = _rest(s,c2,basis,Val(Split),Val(Emulated))
                _store_chunks!(probabilities(i),band,rest,Val(Split))
                wait_stores()
                fence_before_thread_sync()
                token = ptx"mbarrier.arrive.shared.b64"(plast(i).ptr)
                CTAs == 1 || release_to_mma(plast(i),pairlast(i))
                mark(6k+Int32(4),5)
                zero = _after(token,0f0)
                chunks = (p...,rest...)
                l += (_chainsum(zero,chunks[1])+_chainsum(zero,chunks[2]))+(_chainsum(zero,chunks[3])+_chainsum(zero,chunks[4]))
                mark(6k+Int32(5),6)
                k += Int32(1)
            end
            unsafe_store!(sums,l,row+Int32(128)*(i+Int32(2)*(n&Int32(1)))+Int32(1),Val(4))
            arrive!(summed(i,n&Int32(1)))
            n += Int32(1)
            Dynamic || (t += stride)
        end
    elseif warp < Int32(12)
        release_registers(Val(64))
        k0 = Int32(0); k1 = Int32(0); n = Int32(0)          # each tile's blocks, and items so far
        t = first_tile
        while true
            t = next_item(t,n)
            more(t) || break
            head, q0 = item(t)
            for j in Int32(0):count(q0)-Int32(1)
                for i in Int32(0):Int32(1)
                    j < tile_count(q0,i) || continue
                    k = i == Int32(0) ? k0 : k1
                    step = Int32(3)*(k0+k1)
                    if j > Int32(0)
                        # O includes the previous block. (At an item's first
                        # block there is nothing to wait for: the tile may be
                        # a block ahead while the other's epilogue runs, and
                        # a wait two phases behind would not end.)
                        mark(step,10+i)
                        wait(sfull(i),k&Int32(1))
                        ptx"bar.sync"(handoff(i)%UInt32,UInt32(64))
                        mark(step+Int32(1),12+i)
                        factor = unsafe_load(factors,row+Int32(128)*i+Int32(1),Val(4))
                        if vote_any_sync(0xffffffff,factor < 1f0)
                            fence_after_thread_sync()
                            for q in 0:3
                                part = @inbounds window(output(i),(band,Int32(32q)),(32,32))
                                values = wait_load(@inbounds load_async(rows,part))
                                @inbounds store_async!(part,values .* factor)
                            end
                            wait_stores()
                        end
                        fence_before_thread_sync()
                    end
                    # At an item's first block O is free once the previous
                    # item's epilogue has read it, which released it then.
                    (j > Int32(0) || n == Int32(0)) && release_to_mma(pready(i),paired(i))
                    mark(step+Int32(2),14+i)
                    i == Int32(0) ? (k0 += Int32(1)) : (k1 += Int32(1))
                end
            end
            dst = @inbounds GlobalTile(pointer(o,Int32(1)+head*seq*Int32(128)),Layout((seq,static(128)),(static(128),static(1))),16)
            for i in Int32(0):Int32(1)
                wait(summed(i,n&Int32(1)),(n>>1)&Int32(1))
                inverse = 1f0/unsafe_load(sums,row+Int32(128)*(i+Int32(2)*(n&Int32(1)))+Int32(1),Val(4))
                wait(ofull(i),n&Int32(1))
                fence_after_thread_sync()
                for q in 0:3
                    values = wait_load(@inbounds load_async(rows,window(output(i),(band,Int32(32q)),(32,32))))
                    @inbounds store!(window(dst,(q0+Int32(128)*i+band,Int32(32q)),(32,32)),Fragment{BFloat16}(values .* inverse),lane)
                end
                fence_before_thread_sync()
                release_to_mma(pready(i),paired(i))         # the next item's first P V of this tile may begin
            end
            n += Int32(1)
            Dynamic || (t += stride)
        end
    else
        release_registers(Val(64))
        if tid == Int32(384) && rank == Int32(0)            # warp 12: the MMA thread (of the pair)
            c = Int32(0); k0 = Int32(0); k1 = Int32(0); n = Int32(0)   # stages, each tile's blocks, items
            @inline ready(s) = (wait(kvfull(s%Int32(Stages)),(s÷Int32(Stages))&Int32(1)); fence_after_thread_sync())
            @inline stage_of(s) = kvstage(s%Int32(Stages))
            @inline release(s) = commit(kvempty(s%Int32(Stages)))
            @inline function pv_step(i,vb,k,accumulate)
                mark(Int32(8)*k+Int32(4i),20+i)
                wait(CTAs == 1 ? pready(i) : paired(i),k&Int32(1))
                fence_after_thread_sync()
                @inbounds mma(pv,output(i),window(probabilities(i),(Int32(0),Int32(0)),(128,32Split)),vb,32Split,accumulate)
                mark(Int32(8)*k+Int32(4i+1),22+i)
                wait(CTAs == 1 ? plast(i) : pairlast(i),k&Int32(1))
                fence_after_thread_sync()
                @inbounds mma(pv,output(i),window(probabilities(i),(Int32(0),Int32(32Split)),(128,128-32Split)),operand(vb,(Int32(32Split),Int32(0))),128-32Split,true)
                mark(Int32(8)*k+Int32(4i+2),24+i)
            end
            t = first_tile
            while true
                t = next_item(t,n)
                more(t) || break
                _, q0 = item(t)
                blocks_t = count(q0)
                wait(qfull(0),n&Int32(1)); wait(qfull(1),n&Int32(1))
                fence_after_thread_sync()
                qa = @inbounds (operand(qk,OperandA(),shared_tile(tq,qstage(Int32(0)))),operand(qk,OperandA(),shared_tile(tq,qstage(Int32(1)))))
                ready(c)
                kb = @inbounds operand(qk,OperandB(),shared_tile(tk,stage_of(c)))
                PTX.Utils.@unroll for i in 0:1
                    @inbounds mma(qk,scores(i),qa[i+1],kb,128,false)
                    commit(sfull(i))
                end
                release(c)
                c += Int32(1)
                for j in Int32(1):blocks_t-Int32(1)
                    # Each block is awaited where it is first read: V before
                    # the first P V, K before the first Q Kᵀ, behind it. A
                    # causal item's first tile finishes a block early.
                    ready(c)
                    vb = @inbounds operand(pv,OperandB(),shared_tile(tv,stage_of(c)))
                    kb = @inbounds operand(qk,OperandB(),shared_tile(tk,stage_of(c+Int32(1))))
                    PTX.Utils.@unroll for i in 0:1
                        pv_step(i,vb,i == 0 ? k0 : k1,j > Int32(1))
                        if i == 0 && Causal && j == blocks_t-Int32(1)
                            commit(ofull(0))
                            ready(c+Int32(1))               # the second tile's Q Kᵀ still reads K
                        else
                            i == 0 && ready(c+Int32(1))
                            @inbounds mma(qk,scores(i),qa[i+1],kb,128,false)
                            commit(sfull(i))
                            mark(Int32(8)*(i == 0 ? k0 : k1)+Int32(4i+3),26+i)
                        end
                        i == 0 ? (k0 += Int32(1)) : (k1 += Int32(1))
                    end
                    release(c); release(c+Int32(1))
                    c += Int32(2)
                end
                commit(qempty(0)); commit(qempty(1))
                ready(c)
                vb = @inbounds operand(pv,OperandB(),shared_tile(tv,stage_of(c)))
                PTX.Utils.@unroll for i in 0:1
                    if !(i == 0 && Causal)                  # done already
                        pv_step(i,vb,i == 0 ? k0 : k1,blocks_t > Int32(1))
                        commit(ofull(i))
                        i == 0 ? (k0 += Int32(1)) : (k1 += Int32(1))
                    end
                end
                release(c)
                c += Int32(1); n += Int32(1)
                Dynamic || (t += stride)
            end
        elseif tid == Int32(416)                            # warp 13: the load thread
            c = Int32(0); n = Int32(0)
            # Stage `x` of the ring: returns the next.
            @inline function load(x,binding,origin)
                r = x%Int32(Stages)
                mark(Int32(2)*x,30)
                x >= Int32(Stages) && wait(kvempty(r),((x÷Int32(Stages))-Int32(1))&Int32(1))
                mark(Int32(2)*x+Int32(1),31)
                rank == Int32(0) && arrive!(kvfull(r),CTAs*kvbytes)
                @inbounds tma_load!(shared_tile(binding,kvstage(r)),binding,origin,kvfull(r);cta=completion())
                x+Int32(1)
            end
            t = first_tile
            while true
                if Dynamic
                    unsafe_store!(ids,more(t) ? t : Int32(-1),(n&Int32(1))+Int32(1),Val(4))
                    arrive!(itemready(n&Int32(1)))
                end
                more(t) || break
                head, q0 = item(t)
                blocks_t = count(q0)
                half = rank*Int32(128÷CTAs)                 # this CTA's keys of K, columns of V
                for i in Int32(0):Int32(1)
                    n >= Int32(1) && wait(qempty(i),(n-Int32(1))&Int32(1))
                    rank == Int32(0) && arrive!(qfull(i),CTAs*qbytes)
                    @inbounds tma_load!(shared_tile(tq,qstage(i)),tq,(q0+Int32(128)*i,Int32(0),head),qfull(i);cta=completion())
                    i == Int32(0) && (c = load(c,tk,(Int32(0),half,head)))
                end
                for j in Int32(1):blocks_t-Int32(1)
                    c = load(c,tv,(Int32(128)*(j-Int32(1)),half,head))
                    c = load(c,tk,(Int32(0),Int32(128)*j+half,head))
                end
                c = load(c,tv,(Int32(128)*(blocks_t-Int32(1)),half,head))
                n += Int32(1)
                t = Dynamic ? gridDim().x % Int32+CUDACore.atomic_add!(pointer(counter),Int32(1)) : t+stride
            end
        end
    end
    fence_before_thread_sync()
    CTAs == 1 ? sync_threads() : cluster_sync()             # no TMEM access remains in either CTA
    fence_after_thread_sync()
    warp == Int32(12) && tmem_deallocate!(qk,tmem,512)
    nothing
end

"""
    blackwell_attention(q, k, v; stages, ctas=1) -> (kernel, launch, o)

O = softmax(Q Kᵀ / √d) V for BF16 `q`, `k`, `v` (d×seq×(batch·heads)), d =
128, the key length a multiple of 128 and the query length of 256 per CTA:
`ctas = 2` makes CTA pairs compute 512 rows with 256-row products. `stages`
counts ring stages of a K or V block's share (4 by default for one CTA, 8
for pairs, whose stages are half as large). With a `timeline`
(`Timeline(CUDACore.zeros(UInt64,capacity,14,blocks))`), each warp role
records its steps (see the kernel for the events).
"""
function blackwell_attention(q::CuArray{T,3},k::CuArray{T,3},v::CuArray{T,3};ctas::Integer=1,stages::Integer=4ctas,split::Integer=3,causal::Bool=false,dynamic::Bool=causal,emulate::Tuple=(),o=nothing,timeline=nothing) where T
    d, sq, bh = size(q); skv = size(k,2)
    ctas in (1,2) || throw(ArgumentError("one CTA or a CTA pair"))
    split in (1,2,3) || throw(ArgumentError("P is published after one to three of its four chunks"))
    causal && !(ctas == 1 && size(q,2) == size(k,2)) && throw(ArgumentError("causal attention runs in one CTA, with equal sequence lengths"))
    dynamic && ctas != 1 && throw(ArgumentError("dynamic scheduling hands items to single CTAs"))
    counter = CuArray(Int32[0])
    d == 128 && size(k) == size(v) == (d,skv,bh) && sq % (256ctas) == 0 && skv % 128 == 0 ||
        throw(ArgumentError("d must be 128, the query length a multiple of $(256ctas) and the key length of 128"))
    tq = prepare_tma(TMATile(T,(128,128),2),q)                 # 128 query rows × d
    tk = prepare_tma(TMATile(T,(128,128÷ctas),1),k)            # d × this CTA's keys of a block
    tv = prepare_tma(TMATile(T,(128,128÷ctas),2),v)            # a block's keys × this CTA's columns of d
    shmem = 192+16stages+3072+1024+2transfer_bytes(tq)+stages*transfer_bytes(tk)
    shmem <= 227*1024 || throw(ArgumentError("$stages stages need $(cld(shmem,1024)) KiB of shared memory"))
    o = o === nothing ? similar(q) : o
    scale = Float32(1/sqrt(d))
    kernel = @cuda launch=false minthreads=512 maxregs=128 blackwell_attention_kernel!(o,tq,tk,tv,counter,Int32(skv),Int32(bh),scale,Val(stages),Val(ctas),Val(split),Val(causal),Val(dynamic),Val(emulate),timeline)
    CUDACore.attributes(kernel.fun)[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem
    sms = CUDACore.attribute(CUDACore.device(),CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    blocks = ctas*min((sq÷(256ctas))*bh,sms÷ctas)
    launch() = GC.@preserve tq tk tv (dynamic && fill!(counter,Int32(0)); kernel(o,tq,tk,tv,counter,Int32(skv),Int32(bh),scale,Val(stages),Val(ctas),Val(split),Val(causal),Val(dynamic),Val(emulate),timeline;threads=512,blocks,shmem,clustersize=ctas))
    kernel, launch, o
end
