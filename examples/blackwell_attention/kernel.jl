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
# A thread's row of a fragment in the row-per-lane ownership.
@inline _rowmax(x) = maximum(x;dims=2).data[1]
@inline _rowsum(x) = sum(x;dims=2).data[1]
@inline _exponentials(x,c,m) = Tylo._ex2.(muladd.(x,c,-m))
# `x`, once `token` exists: a data dependence the compilers cannot see through.
@inline _after(token::UInt64,x) = ifelse(token == typemax(UInt64),zero(x),x)
# A row's sum as one chain from `init`: every addition waits for `init`.
@inline _chainsum(init,x) = foldl(+,x.data;init)

function blackwell_attention_kernel!(o,tq,tk,tv,seq_kv::Int32,heads::Int32,scale::Float32,::Val{Stages}) where Stages
    qk = Tcgen05Atom((128,128,16),BFloat16)                 # S = Q Kᵀ: 128 rows × 128 keys
    pv = Tcgen05Atom((128,128,16),BFloat16)                 # O += P V: 128 rows × d
    qbytes, kvbytes = transfer_bytes(tq), transfer_bytes(tk)
    workspace = @inbounds CuDynamicSharedArray(UInt8,128+16Stages+2048+1024+(2+Stages)*qbytes)
    raw = pointer(workspace)
    @inline qfull(i) = Barrier(raw,i)                       # tile i's queries have arrived
    @inline qempty(i) = Barrier(raw+16,i)                   # tile i's queries may be replaced
    @inline sfull(i) = Barrier(raw+32,i)                    # S of tile i is ready (and O to the previous block)
    @inline pready(i) = Barrier(raw+48,i)                   # P's first 96 keys are written and O rescaled
    @inline plast(i) = Barrier(raw+64,i)                    # P's last 32 keys are written
    @inline ofull(i) = Barrier(raw+80,i)                    # O of tile i is final
    @inline summed(i) = Barrier(raw+96,i)                   # the row sums of tile i are written
    @inline kvfull(s) = Barrier(raw+112,s)                  # stage s holds a K or V block
    @inline kvempty(s) = Barrier(raw+112+8Stages,s)         # stage s may be refilled
    slot = reinterpret(Core.LLVMPtr{UInt32,PTX.AS.Shared},raw+112+16Stages)
    factors = reinterpret(Core.LLVMPtr{Float32,PTX.AS.Shared},raw+128+16Stages)      # [row, tile]: rescale factor
    sums = factors+1024                                                              # [row, tile]: row sum
    start = raw+128+16Stages+2048
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
            initialize!(pready(i),128+4)                    # the softmax threads and four correction warps
            initialize!(plast(i),128)
            initialize!(ofull(i),1)
            initialize!(summed(i),128)
        end
        for s in Int32(0):Int32(Stages-1)
            initialize!(kvfull(s),1)
            initialize!(kvempty(s),1)
        end
        fence_barrier_init()
    end
    warp == Int32(12) && tmem_allocate!(slot,512)
    fence_before_thread_sync()
    sync_threads()
    fence_after_thread_sync()
    tmem = unsafe_load(slot)
    # The whole allocation, as FP32 columns and as BF16 halves of them; S, P
    # and O of each tile are windows of it, so that one register holds TMEM.
    words = TmemTile(Float32,tmem,Layout((static(128),static(512)),(static(1),static(128))))
    halves = TmemTile(BFloat16,tmem,Layout((static(128),static(1024)),(static(1),static(128))))
    @inline scores(i) = @inbounds window(words,(0,128i),(128,128))
    @inline probabilities(i) = @inbounds window(halves,(0,256i+128),(128,128))    # over S's second half
    @inline output(i) = @inbounds window(words,(0,256+128i),(128,128))
    seq = size(o,2) % Int32
    nq = seq ÷ Int32(256)
    tiles = nq*heads
    blocks = seq_kv ÷ Int32(128)
    c2 = scale*1.442695f0                                   # scores to base-2 exponents (log2 e)
    band = (warp%Int32(4))*Int32(32)                        # the TMEM lanes this warp may access
    row = band+lane
    # Named barriers 1-8 pair softmax warp (tile i, band b) with correction warp b.
    @inline handoff(i) = Int32(1)+Int32(4)*i+warp%Int32(4)
    rows = distribute((32,32),32;vector=32,axis=2)          # TMEM transfers: lane t holds row t, 32 columns
    if warp < Int32(8)
        acquire_registers(Val(192))
        i = warp ÷ Int32(4)
        k = Int32(0)                                        # blocks so far
        t = Int32(blockIdx().x)-Int32(1)
        while t < tiles
            m = -Inf32; l = 0f0
            for j in Int32(0):blocks-Int32(1)
                wait(sfull(i),k&Int32(1))
                fence_after_thread_sync()
                # The row in four loads and one wait.
                pending = @inbounds (load_async(rows,window(scores(i),(band,Int32(0)),(32,32))),
                                     load_async(rows,window(scores(i),(band,Int32(32)),(32,32))),
                                     load_async(rows,window(scores(i),(band,Int32(64)),(32,32))),
                                     load_async(rows,window(scores(i),(band,Int32(96)),(32,32))))
                s = (wait_load(pending[1]),wait_load(pending[2]),wait_load(pending[3]),wait_load(pending[4]))
                top = max(_rowmax(s[1]),_rowmax(s[2]),_rowmax(s[3]),_rowmax(s[4]))*c2
                if j == Int32(0)
                    m = top
                else
                    factor = 1f0
                    if top-m > 8f0                          # the maximum moves: rescale O and the sum
                        factor = Tylo._ex2(m-top)
                        m = top
                    end
                    unsafe_store!(factors,factor,row+Int32(128)*i+Int32(1),Val(4))
                    ptx"bar.arrive"(handoff(i)%UInt32,UInt32(64))
                    l *= factor
                end
                # Three chunks of P, published; then the last and the row sum.
                # Nothing orders pure arithmetic after a store or an arrival,
                # so the work that must wait takes an input through the
                # arrival's token: the compilers cannot hoist it above.
                p = (_exponentials(s[1],c2,m),_exponentials(s[2],c2,m),_exponentials(s[3],c2,m))
                PTX.Utils.@unroll for q in 0:2
                    @inbounds store_async!(window(probabilities(i),(band,Int32(32q)),(32,32)),Fragment{BFloat16}(p[q+1]))
                end
                wait_stores()
                fence_before_thread_sync()
                token = ptx"mbarrier.arrive.shared.b64"(pready(i).ptr)
                last = _exponentials(s[4],c2,_after(token,m))
                @inbounds store_async!(window(probabilities(i),(band,Int32(96)),(32,32)),Fragment{BFloat16}(last))
                wait_stores()
                fence_before_thread_sync()
                token = ptx"mbarrier.arrive.shared.b64"(plast(i).ptr)
                zero = _after(token,0f0)
                l += (_chainsum(zero,p[1])+_chainsum(zero,p[2]))+(_chainsum(zero,p[3])+_chainsum(zero,last))
                k += Int32(1)
            end
            unsafe_store!(sums,l,row+Int32(128)*i+Int32(1),Val(4))
            arrive!(summed(i))
            t += gridDim().x
        end
    elseif warp < Int32(12)
        release_registers(Val(72))
        k = Int32(0); n = Int32(0)                          # blocks and tiles so far
        t = Int32(blockIdx().x)-Int32(1)
        while t < tiles
            for j in Int32(0):blocks-Int32(1)
                for i in Int32(0):Int32(1)
                    wait(sfull(i),k&Int32(1))               # O includes the previous block
                    if j > Int32(0)
                        ptx"bar.sync"(handoff(i)%UInt32,UInt32(64))
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
                    sync_warp()
                    lane == Int32(0) && arrive!(pready(i))
                end
                k += Int32(1)
            end
            head = unchecked_div(t,nq)
            q0 = unchecked_rem(t,nq)*Int32(256)
            dst = @inbounds GlobalTile(pointer(o,Int32(1)+head*seq*Int32(128)),Layout((seq,static(128)),(static(128),static(1))),16)
            for i in Int32(0):Int32(1)
                wait(summed(i),n&Int32(1))
                inverse = 1f0/unsafe_load(sums,row+Int32(128)*i+Int32(1),Val(4))
                wait(ofull(i),n&Int32(1))
                fence_after_thread_sync()
                for q in 0:3
                    values = wait_load(@inbounds load_async(rows,window(output(i),(band,Int32(32q)),(32,32))))
                    @inbounds store!(window(dst,(q0+Int32(128)*i+band,Int32(32q)),(32,32)),Fragment{BFloat16}(values .* inverse),lane)
                end
                fence_before_thread_sync()
            end
            n += Int32(1)
            t += gridDim().x
        end
    else
        release_registers(Val(56))
        if tid == Int32(384)                                # warp 12: the MMA thread
            c = Int32(0); k = Int32(0); n = Int32(0)        # stages, blocks and tiles so far
            @inline ready(s) = (wait(kvfull(s%Int32(Stages)),(s÷Int32(Stages))&Int32(1)); fence_after_thread_sync())
            @inline stage_of(s) = kvstage(s%Int32(Stages))
            @inline release(s) = commit_mma(kvempty(s%Int32(Stages)))
            @inline function pv_step(i,vb,k,accumulate)
                wait(pready(i),k&Int32(1))
                fence_after_thread_sync()
                @inbounds mma(pv,output(i),window(probabilities(i),(Int32(0),Int32(0)),(128,96)),vb,96,accumulate)
                wait(plast(i),k&Int32(1))
                fence_after_thread_sync()
                @inbounds mma(pv,output(i),window(probabilities(i),(Int32(0),Int32(96)),(128,32)),operand(vb,(Int32(96),Int32(0))),32,true)
            end
            t = Int32(blockIdx().x)-Int32(1)
            while t < tiles
                wait(qfull(0),n&Int32(1)); wait(qfull(1),n&Int32(1))
                fence_after_thread_sync()
                qa = @inbounds (operand(qk,OperandA(),shared_tile(tq,qstage(Int32(0)))),operand(qk,OperandA(),shared_tile(tq,qstage(Int32(1)))))
                ready(c)
                kb = @inbounds operand(qk,OperandB(),shared_tile(tk,stage_of(c)))
                PTX.Utils.@unroll for i in 0:1
                    @inbounds mma(qk,scores(i),qa[i+1],kb,128,false)
                    commit_mma(sfull(i))
                end
                release(c)
                c += Int32(1)
                for j in Int32(1):blocks-Int32(1)
                    ready(c); ready(c+Int32(1))
                    vb = @inbounds operand(pv,OperandB(),shared_tile(tv,stage_of(c)))
                    kb = @inbounds operand(qk,OperandB(),shared_tile(tk,stage_of(c+Int32(1))))
                    PTX.Utils.@unroll for i in 0:1
                        pv_step(i,vb,k,j > Int32(1))
                        @inbounds mma(qk,scores(i),qa[i+1],kb,128,false)
                        commit_mma(sfull(i))
                    end
                    release(c); release(c+Int32(1))
                    c += Int32(2); k += Int32(1)
                end
                commit_mma(qempty(0)); commit_mma(qempty(1))
                ready(c)
                vb = @inbounds operand(pv,OperandB(),shared_tile(tv,stage_of(c)))
                PTX.Utils.@unroll for i in 0:1
                    pv_step(i,vb,k,blocks > Int32(1))
                    commit_mma(ofull(i))
                end
                release(c)
                c += Int32(1); k += Int32(1); n += Int32(1)
                t += gridDim().x
            end
        elseif tid == Int32(416)                            # warp 13: the load thread
            c = Int32(0); n = Int32(0)
            # Stage `x` of the ring: returns the next.
            @inline function load(x,binding,origin)
                r = x%Int32(Stages)
                x >= Int32(Stages) && wait(kvempty(r),((x÷Int32(Stages))-Int32(1))&Int32(1))
                arrive!(kvfull(r),kvbytes)
                @inbounds tma_load!(shared_tile(binding,kvstage(r)),binding,origin,kvfull(r))
                x+Int32(1)
            end
            t = Int32(blockIdx().x)-Int32(1)
            while t < tiles
                head = unchecked_div(t,nq)
                q0 = unchecked_rem(t,nq)*Int32(256)
                for i in Int32(0):Int32(1)
                    n >= Int32(1) && wait(qempty(i),(n-Int32(1))&Int32(1))
                    arrive!(qfull(i),qbytes)
                    @inbounds tma_load!(shared_tile(tq,qstage(i)),tq,(q0+Int32(128)*i,Int32(0),head),qfull(i))
                    i == Int32(0) && (c = load(c,tk,(Int32(0),Int32(0),head)))
                end
                for j in Int32(1):blocks-Int32(1)
                    c = load(c,tv,(Int32(128)*(j-Int32(1)),Int32(0),head))
                    c = load(c,tk,(Int32(0),Int32(128)*j,head))
                end
                c = load(c,tv,(Int32(128)*(blocks-Int32(1)),Int32(0),head))
                n += Int32(1)
                t += gridDim().x
            end
        end
    end
    fence_before_thread_sync()
    sync_threads()
    fence_after_thread_sync()
    warp == Int32(12) && tmem_deallocate!(tmem,512)
    nothing
end

"""
    blackwell_attention(q, k, v; stages=4) -> (kernel, launch, o)

O = softmax(Q Kᵀ / √d) V for BF16 `q`, `k`, `v` (d×seq×(batch·heads)), d =
128, the query length a multiple of 256 and the key length of 128.
"""
function blackwell_attention(q::CuArray{T,3},k::CuArray{T,3},v::CuArray{T,3};stages::Integer=4,o=nothing) where T
    d, sq, bh = size(q); skv = size(k,2)
    d == 128 && size(k) == size(v) == (d,skv,bh) && sq % 256 == 0 && skv % 128 == 0 ||
        throw(ArgumentError("d must be 128, the query length a multiple of 256 and the key length of 128"))
    tq = prepare_tma(TMATile(T,(128,128),2),q)                 # 128 query rows × d
    tk = prepare_tma(TMATile(T,(128,128),1),k)                 # d × 128 keys
    tv = prepare_tma(TMATile(T,(128,128),2),v)                 # 128 keys × d
    shmem = 128+16stages+2048+1024+(2+stages)*transfer_bytes(tq)
    shmem <= 227*1024 || throw(ArgumentError("$stages stages need $(cld(shmem,1024)) KiB of shared memory"))
    o = o === nothing ? similar(q) : o
    scale = Float32(1/sqrt(d))
    kernel = @cuda launch=false minthreads=512 maxregs=128 blackwell_attention_kernel!(o,tq,tk,tv,Int32(skv),Int32(bh),scale,Val(stages))
    CUDACore.attributes(kernel.fun)[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem
    blocks = min((sq÷256)*bh,CUDACore.attribute(CUDACore.device(),CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
    launch() = GC.@preserve tq tk tv kernel(o,tq,tk,tv,Int32(skv),Int32(bh),scale,Val(stages);threads=512,blocks,shmem)
    kernel, launch, o
end
