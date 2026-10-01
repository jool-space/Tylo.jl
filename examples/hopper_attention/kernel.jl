using Tylo, PTX, CUDACore, BFloat16s
using Tylo.Layouts: Layout, static
include(joinpath(@__DIR__,"..","launch.jl"))

# FlashAttention's forward pass on Hopper, O = softmax(Q Kᵀ · scale) V, in the
# style of FlashAttention-3. Q, K, V and O are d×seq×(batch·heads) BF16
# arrays (d contiguous), d = 128. Each CTA computes 128 query rows of one
# head with three warpgroups: the first loads Q once and streams K and V
# blocks of 128 rows through a ring of shared stages with TMA; the other two
# each own 64 query rows. For every block a consumer computes S = Q Kᵀ with
# WGMMA, updates the online softmax in registers, converts the probabilities
# to BF16 in place (the accumulator's layout is the A operand's), and adds
# P V with P from registers and V transposed from shared memory.
# With `overlap`, a consumer issues block j's S = Q Kᵀ and block j-1's P V
# back to back, waits for S only, and computes block j's softmax while P V
# runs, as FlashAttention-3 does within a warpgroup. Both products complete
# within the iteration: an accumulator still in flight across the loop's
# back edge would be copied there, and ptxas would serialize every WGMMA.
# With `pingpong` as well, the two consumers take turns issuing their products
# through named barriers, so that one's softmax runs while the other's
# products occupy the tensor cores (FlashAttention-3's warpgroup ping-pong).
# The non-persistent kernel's shared memory: Q's barrier and the ring's, Q,
# and the ring of K and V stages.
@inline attention_shared(cfg,tq,tk,tv) = SharedLayout((;
    qfull = barriers(1;arrivals=1),
    full = barriers(cfg.stages;arrivals=1),                 # K and V of stage s have landed
    empty = barriers(cfg.stages;arrivals=256),              # stage s may be refilled: every consumer thread
    q = buffers(transfer_bytes(tq),1;align=1024),
    kv = buffers(transfer_bytes(tk)+transfer_bytes(tv),cfg.stages;align=1024)))

function attention_kernel!(o,tq,tk,tv,seq_kv::Int32,scale::Float32,cfg::Config)
    qk = WGMMAAtom(BFloat16,128,64)             # S: 64 query rows × 128 keys, half of d per issue
    pv = WGMMAAtom(BFloat16,128,64)             # O: 64 query rows × d, half of a block per issue
    qbytes, kbytes, vbytes = transfer_bytes(tq), transfer_bytes(tk), transfer_bytes(tv)
    layout = attention_shared(cfg,tq,tk,tv)
    smem = bind(layout,pointer(@inbounds CuDynamicSharedArray(UInt8,shared_bytes(layout))))
    stages = Ring(smem.full,smem.empty)
    tid = Int32(threadIdx().x)-Int32(1)
    warpgroup = warp_uniform(tid ÷ Int32(128))
    if tid == Int32(0)
        initialize!(smem)
        fence_barrier_init()
    end
    sync_threads()
    q0 = (Int32(blockIdx().x)-Int32(1))*Int32(128)
    head = Int32(blockIdx().y)-Int32(1)
    blocks = seq_kv ÷ Int32(128)
    if warpgroup == Int32(0)
        release_registers(40)
        if tid == Int32(0)
            arrive!(smem.qfull[0],qbytes)
            @inbounds tma_load!(shared_tile(tq,smem.q[0]),tq,(q0,Int32(0),head),smem.qfull[0])
            for j in Int32(0):blocks-Int32(1)
                s = stages[j]
                acquire(s)
                at = smem.kv[s.slot]
                arrive!(s.full,kbytes+vbytes)
                @inbounds tma_load!(shared_tile(tk,at),tk,(Int32(0),Int32(128)*j,head),s.full)
                @inbounds tma_load!(shared_tile(tv,at+kbytes),tv,(Int32(128)*j,Int32(0),head),s.full)
            end
        end
    else
        acquire_registers(232)
        rows = (warpgroup-Int32(1))*Int32(64)
        thread = tid%Int32(128)
        wait(smem.qfull[0],0)
        q = shared_tile(tq,smem.q[0])
        @inbounds qa = (operand(qk,OperandA(),q,(rows,Int32(0))),operand(qk,OperandA(),q,(rows,Int32(64))))
        state = SoftmaxState(finish_mma(zero_accumulator(qk));dims=2)
        # S for block j: two groups, one per half of d.
        @inline function scores_async(j)
            k = shared_tile(tk,smem.kv[stages[j].slot])
            # (A name of the enclosing body assigned here would be boxed.)
            @inbounds half = mma_async(qk,qa[1],operand(qk,OperandB(),k,(Int32(0),Int32(0))))   # fresh: no zeroing
            @inbounds mma_async(qk,qa[2],operand(qk,OperandB(),k,(Int32(64),Int32(0))),half)
        end
        # O plus P V for block j: two groups, one per half of the block, P
        # (the softmax weights) from registers and V transposed.
        @inline function output_async(out,weights,j)
            probabilities = Fragment{BFloat16}(weights)
            v = shared_tile(tv,smem.kv[stages[j].slot]+kbytes)
            @inbounds out = mma_async(pv,window(probabilities,(0,0),(64,64)),operand(pv,OperandB(),v,(Int32(0),Int32(0))),out)
            @inbounds mma_async(pv,window(probabilities,(0,64),(64,64)),operand(pv,OperandB(),v,(Int32(64),Int32(0))),out)
        end
        if !cfg.overlap || blocks == Int32(1)
            out = zero_accumulator(pv)
            for j in Int32(0):blocks-Int32(1)
                wait(stages[j])
                update = softmax_update(state,finish_mma(wait_mma(scores_async(j)));scale)
                state = update.state
                out = out .* update.rescale                   # the output so far, to the new maxima
                out = wait_mma(output_async(out,update.weights,j))
                arrive!(stages[j].empty)
            end
        else
            # Turns: consumer w (1 or 2) waits on its turn and passes the
            # other's; the second primes the first turn and skips its last pass.
            turns = named_barriers((;turn=named(2;threads=256))).turn
            @inline turn() = cfg.pingpong && wait(@inbounds(turns[warpgroup-Int32(1)]);aligned=false)
            @inline pass(final) = cfg.pingpong && !(final && warpgroup == Int32(2)) &&
                arrive!(@inbounds(turns[Int32(2)-warpgroup]);aligned=false)
            cfg.pingpong && warpgroup == Int32(2) && arrive!(turns[0];aligned=false)
            wait(stages[0])
            turn()
            first_scores = scores_async(Int32(0))
            pass(false)
            update = softmax_update(state,finish_mma(wait_mma(first_scores));scale)
            state, p = update.state, update.weights
            out = zero_accumulator(pv)
            for j in Int32(1):blocks-Int32(1)
                wait(stages[j])
                turn()
                scores = scores_async(j)
                pending = output_async(out,p,j-Int32(1))
                pass(false)
                update = softmax_update(state,finish_mma(wait_mma(scores;newer=2));scale)
                state = update.state
                out = wait_mma(pending) .* update.rescale
                p = update.weights
                arrive!(stages[j-Int32(1)].empty)
            end
            turn()
            last_output = output_async(out,p,blocks-Int32(1))
            pass(true)
            out = wait_mma(last_output)
            arrive!(stages[blocks-Int32(1)].empty)
        end
        result = Fragment{BFloat16}(softmax_normalize(finish_mma(out),state))
        seq = size(o,2) % Int32
        dst = @inbounds GlobalTile(pointer(o,Int32(1)+head*seq*Int32(128)),Layout((seq,static(128)),(static(128),static(1))),16)
        @inbounds store!(window(dst,(q0+rows,Int32(0)),(64,128)),result,thread)
    end
    nothing
end

# The persistent form: one CTA per SM walks (query block, head) tiles, query
# blocks first so that the CTAs running at once share a head's K and V
# through L2. Q is double-buffered and K/V stages are counted across tiles,
# so the producer loads the next tile's Q and first blocks while the
# consumers finish and store the current one. With `causal` (queries and keys
# aligned), query block b attends to key blocks 0 through b only: the others
# are never loaded and the diagonal block is masked. Causal tiles differ in
# length, so CTAs take them from a global counter (`counter`, zero at launch),
# longest first within each head; the producer passes each tile to the
# consumers in shared memory, published by the barrier that announces its Q,
# and -1 when none remain. (Taking them longest first across all heads
# balances no better and loses the L2 sharing of a head's K and V.)
# The persistent kernel's shared memory: double-buffered Q with the tile in
# each buffer, and the ring of K and V stages, counted across tiles.
@inline persistent_shared(cfg,tq,tk,tv) = SharedLayout((;
    qfull = barriers(2;arrivals=1),                         # Q buffer b holds a tile's queries
    qempty = barriers(2;arrivals=256),                      # Q buffer b may be refilled
    full = barriers(cfg.stages;arrivals=1),
    empty = barriers(cfg.stages;arrivals=256),
    tile = scalars(Int32,2),                                # the tile in Q buffer b, or -1
    q = buffers(transfer_bytes(tq),2;align=1024),
    kv = buffers(transfer_bytes(tk)+transfer_bytes(tv),cfg.stages;align=1024)))

function persistent_attention_kernel!(o,tq,tk,tv,counter,seq_kv::Int32,heads::Int32,scale::Float32,cfg::Config)
    qk = WGMMAAtom(BFloat16,128,64)
    pv = WGMMAAtom(BFloat16,128,64)
    qbytes, kbytes, vbytes = transfer_bytes(tq), transfer_bytes(tk), transfer_bytes(tv)
    layout = persistent_shared(cfg,tq,tk,tv)
    smem = bind(layout,pointer(@inbounds CuDynamicSharedArray(UInt8,shared_bytes(layout))))
    queries = Ring(smem.qfull,smem.qempty)
    stages = Ring(smem.full,smem.empty)
    tid = Int32(threadIdx().x)-Int32(1)
    warpgroup = warp_uniform(tid ÷ Int32(128))
    if tid == Int32(0)
        initialize!(smem)
        fence_barrier_init()
    end
    sync_threads()
    seq = size(o,2) % Int32
    nq = seq ÷ Int32(128)
    tiles = nq*heads
    blocks = seq_kv ÷ Int32(128)
    # Head and query block of tile t: heads outer, so concurrent CTAs share K
    # and V in L2, or (`heads_inner`) query blocks outer, for causal tiles
    # longest first across all heads. Causal tiles run longest first.
    @inline function tile_of(t)
        head, block = cfg.heads_inner ? (unchecked_rem(t,heads), unchecked_div(t,heads)) : (unchecked_div(t,nq), unchecked_rem(t,nq))
        head, cfg.causal ? nq-Int32(1)-block : block
    end
    # The tile after `t`: the next in the static order, or (`dynamic`) from
    # the counter.
    @inline next_tile(t) = cfg.dynamic ? gridDim().x+CUDACore.atomic_add!(pointer(counter),Int32(1)) : t+gridDim().x
    if warpgroup == Int32(0)
        release_registers(40)
        if tid == Int32(0)
            i = Int32(0); c = Int32(0)                      # tiles and K/V blocks so far
            t = Int32(blockIdx().x)-Int32(1)
            while true
                b = queries[i]
                acquire(b)
                smem.tile[b.slot] = t < tiles ? t : Int32(-1)
                if t >= tiles
                    arrive!(b.full)                         # no tile: the consumers stop
                    break
                end
                head, qblock = tile_of(t)
                q0 = qblock*Int32(128)
                arrive!(b.full,qbytes)
                @inbounds tma_load!(shared_tile(tq,smem.q[b.slot]),tq,(q0,Int32(0),head),b.full)
                for j in Int32(0):(cfg.causal ? qblock : blocks-Int32(1))
                    s = stages[c]
                    acquire(s)
                    at = smem.kv[s.slot]
                    arrive!(s.full,kbytes+vbytes)
                    @inbounds tma_load!(shared_tile(tk,at),tk,(Int32(0),Int32(128)*j,head),s.full)
                    @inbounds tma_load!(shared_tile(tv,at+kbytes),tv,(Int32(128)*j,Int32(0),head),s.full)
                    c += Int32(1)
                end
                i += Int32(1)
                t = next_tile(t)
            end
        end
    else
        acquire_registers(232)
        rows = (warpgroup-Int32(1))*Int32(64)
        thread = tid%Int32(128)
        i = Int32(0); c = Int32(0)
        while true
            b = queries[i]
            wait(b)
            t = smem.tile[b.slot]
            t < Int32(0) && break
            q = shared_tile(tq,smem.q[b.slot])
            @inbounds qa = (operand(qk,OperandA(),q,(rows,Int32(0))),operand(qk,OperandA(),q,(rows,Int32(64))))
            state = SoftmaxState(finish_mma(zero_accumulator(qk));dims=2)
            out = zero_accumulator(pv)
            head, qblock = tile_of(t)
            last = cfg.causal ? qblock : blocks-Int32(1)
            for j in Int32(0):last
                s = stages[c]
                wait(s)
                k = shared_tile(tk,smem.kv[s.slot]); v = shared_tile(tv,smem.kv[s.slot]+kbytes)
                @inbounds sa = mma_async(qk,qa[1],operand(qk,OperandB(),k,(Int32(0),Int32(0))))
                @inbounds sa = mma_async(qk,qa[2],operand(qk,OperandB(),k,(Int32(64),Int32(0))),sa)
                scores = wait_mma(sa)
                if cfg.causal && j == last
                    # The diagonal block: key c of the block attends to query r
                    # of this warpgroup's rows when c ≤ rows + r.
                    at = coordinates(scores,thread)
                    scores = ifelse.((x -> x[2] <= x[1]+rows).(at),scores,-Inf32)
                end
                update = softmax_update(state,scores;scale)
                state = update.state
                out = out .* update.rescale                   # the output so far, to the new maxima
                p = Fragment{BFloat16}(update.weights)
                @inbounds out = mma_async(pv,window(p,(0,0),(64,64)),operand(pv,OperandB(),v,(Int32(0),Int32(0))),out)
                @inbounds out = wait_mma(mma_async(pv,window(p,(0,64),(64,64)),operand(pv,OperandB(),v,(Int32(64),Int32(0))),out))
                arrive!(s.empty)
                c += Int32(1)
            end
            arrive!(b.empty)
            q0 = qblock*Int32(128)
            result = Fragment{BFloat16}(softmax_normalize(finish_mma(out),state))
            dst = @inbounds GlobalTile(pointer(o,Int32(1)+head*seq*Int32(128)),Layout((seq,static(128)),(static(128),static(1))),16)
            @inbounds store!(window(dst,(q0+rows,Int32(0)),(64,128)),result,thread)
            i += Int32(1)
        end
    end
    nothing
end

"""
    hopper_attention_launch(q, k, v; stages=2, persistent=true, causal=false, overlap=false, pingpong=overlap, sms) -> Launch

Prepare O = softmax(Q Kᵀ / √d) V for BF16 `q`, `k`, `v` of size
d×seq×(batch·heads), d = 128, sequence lengths multiples of 128. With
`persistent`, one CTA per SM walks the tiles (with the simple schedule);
`causal` masks keys after each query (sequences of equal length).
With `overlap`, each block's softmax runs while the tensor cores multiply the
previous block's probabilities by V, and with `pingpong` the two consumers
take turns at the tensor cores, which needs three stages to pay. On an H100
NVL the persistent form is fastest for short sequences (2048) and the
overlapped ping-pong schedule for long ones (4096 and up), each within a
few percent of the other; run.jl tunes between them.
"""
function hopper_attention_launch(q::CuArray{T,3},k::CuArray{T,3},v::CuArray{T,3};stages::Integer=2,persistent::Bool=true,causal::Bool=false,
                                 dynamic::Bool=causal,heads_inner::Bool=false,overlap::Bool=false,pingpong::Bool=overlap,o=nothing,
                                 sms::Integer=multiprocessors()) where T
    d, sq, bh = size(q); skv = size(k,2)
    d == 128 && size(k) == size(v) == (d,skv,bh) && sq % 128 == 0 && skv % 128 == 0 ||
        throw(ArgumentError("d must be 128 and sequence lengths multiples of 128"))
    tq = prepare_tma(TMATile(T,(128,128),2),q)                 # 128 query rows × d
    tk = prepare_tma(TMATile(T,(128,128),1),k)                 # d × 128 keys
    tv = prepare_tma(TMATile(T,(128,128),2),v)                 # 128 keys × d
    o = o === nothing ? similar(q) : o
    scale = Float32(1/sqrt(d))
    pingpong && !overlap && throw(ArgumentError("ping-pong turns apply to the overlapped schedule"))
    persistent && overlap && throw(ArgumentError("the persistent form has the simple schedule"))
    causal && !(persistent && sq == skv) && throw(ArgumentError("causal attention is persistent, with equal sequence lengths"))
    if persistent
        cfg = Config(;stages,causal,dynamic,heads_inner)
        shmem = shared_bytes(persistent_shared(cfg,tq,tk,tv))
        shmem <= 227*1024 || throw(ArgumentError("$stages stages need $(cld(shmem,1024)) KiB of shared memory"))
        counter = CuArray(Int32[0])
        return Launch(persistent_attention_kernel!,(o,tq,tk,tv,counter,Int32(skv),Int32(bh),scale,cfg);
                      threads=384,blocks=min((sq÷128)*bh,sms),shmem,minthreads=384,maxregs=168)
    end
    cfg = Config(;stages,overlap,pingpong)
    shmem = shared_bytes(attention_shared(cfg,tq,tk,tv))
    shmem <= 227*1024 || throw(ArgumentError("$stages stages need $(cld(shmem,1024)) KiB of shared memory"))
    Launch(attention_kernel!,(o,tq,tk,tv,Int32(skv),scale,cfg);
           threads=384,blocks=(sq÷128,bh),shmem,minthreads=384,maxregs=168)
end
"""
    hopper_attention(q, k, v; kwargs...) -> (kernel, launch, o)

[`hopper_attention_launch`](@ref), compiled: the kernel, a launcher (which
returns a dynamic schedule's item counter to zero first), and O.
"""
function hopper_attention(q::CuArray,k::CuArray,v::CuArray;kwargs...)
    l = hopper_attention_launch(q,k,v;kwargs...)
    kernel = compile(l)
    function launch()
        l.kernel === persistent_attention_kernel! && fill!(l.args[5],Int32(0))
        l(kernel)
    end
    kernel, launch, l.result
end
