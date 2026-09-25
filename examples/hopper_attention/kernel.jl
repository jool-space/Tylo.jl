using Tylo, PTX, CUDACore, BFloat16s
using Tylo.Layouts: Layout, static

# FlashAttention's forward pass on Hopper, O = softmax(Q Kᵀ · scale) V, in the
# style of FlashAttention-3. Q, K, V and O are d×seq×(batch·heads) BF16
# arrays (d contiguous), d = 128. Each CTA computes 128 query rows of one
# head with three warpgroups: the first loads Q once and streams K and V
# blocks of 128 rows through `Stages` shared buffers with TMA; the other two
# each own 64 query rows. For every block a consumer computes S = Q Kᵀ with
# WGMMA, updates the online softmax in registers, converts the probabilities
# to BF16 in place (the accumulator's layout is the A operand's), and adds
# P V with P from registers and V transposed from shared memory.
# With `Overlap`, a consumer issues block j's S = Q Kᵀ and block j-1's P V
# back to back, waits for S only, and computes block j's softmax while P V
# runs, as FlashAttention-3 does within a warpgroup. Both products complete
# within the iteration: an accumulator still in flight across the loop's
# back edge would be copied there, and ptxas would serialize every WGMMA.
# With `PingPong` as well, the two consumers take turns issuing their products
# through named barriers, so that one's softmax runs while the other's
# products occupy the tensor cores (FlashAttention-3's warpgroup ping-pong).
function attention_kernel!(o,tq,tk,tv,seq_kv::Int32,scale::Float32,::Val{Stages},::Val{Overlap}=Val(false),::Val{PingPong}=Val(false)) where {Stages,Overlap,PingPong}
    qk = WGMMAAtom(BFloat16,128,64)             # S: 64 query rows × 128 keys, half of d per issue
    pv = WGMMAAtom(BFloat16,128,64)             # O: 64 query rows × d, half of a block per issue
    qbytes, kbytes, vbytes = transfer_bytes(tq), transfer_bytes(tk), transfer_bytes(tv)
    workspace = @inbounds CuDynamicSharedArray(UInt8,8+16Stages+1024+qbytes+Stages*(kbytes+vbytes))
    raw = pointer(workspace)
    @inline qfull() = Barrier(raw,0)
    @inline full(s) = Barrier(raw+8,s)                      # K and V of stage s have landed
    @inline empty(s) = Barrier(raw+8+8Stages,s)             # stage s may be refilled
    start = raw+8+16Stages
    base = start+((UInt32(0)-PTX.smem_addr_u32(start)) & UInt32(1023))
    @inline kstage(s) = base+qbytes+s*Int32(kbytes+vbytes)
    @inline vstage(s) = kstage(s)+kbytes
    tid = Int32(threadIdx().x)-Int32(1)
    warpgroup = warp_uniform(tid ÷ Int32(128))
    if tid == Int32(0)
        initialize!(qfull(),1)
        for s in Int32(0):Int32(Stages-1)
            initialize!(full(s),1)
            initialize!(empty(s),256)
        end
        fence_barrier_init()
    end
    sync_threads()
    q0 = (Int32(blockIdx().x)-Int32(1))*Int32(128)
    head = Int32(blockIdx().y)-Int32(1)
    blocks = seq_kv ÷ Int32(128)
    if warpgroup == Int32(0)
        release_registers(Val(40))
        if tid == Int32(0)
            arrive!(qfull(),qbytes)
            @inbounds tma_load!(shared_tile(tq,base),tq,(q0,Int32(0),head),qfull())
            for j in Int32(0):blocks-Int32(1)
                slot = j%Int32(Stages)
                j >= Int32(Stages) && wait(empty(slot),((j÷Int32(Stages))-Int32(1))&Int32(1))
                arrive!(full(slot),kbytes+vbytes)
                @inbounds tma_load!(shared_tile(tk,kstage(slot)),tk,(Int32(0),Int32(128)*j,head),full(slot))
                @inbounds tma_load!(shared_tile(tv,vstage(slot)),tv,(Int32(128)*j,Int32(0),head),full(slot))
            end
        end
    else
        acquire_registers(Val(232))
        rows = (warpgroup-Int32(1))*Int32(64)
        thread = tid%Int32(128)
        wait(qfull(),Int32(0))
        q = shared_tile(tq,base)
        @inbounds qa = (operand(qk,OperandA(),q,(rows,Int32(0))),operand(qk,OperandA(),q,(rows,Int32(64))))
        state = SoftmaxState(finish_mma(zero_accumulator(qk));dims=2)
        # Named apart from the producer's variables: a function's closures
        # and its other locals share one scope.
        @inline slot_of(j) = j%Int32(Stages)
        @inline phase_of(j) = (j÷Int32(Stages))&Int32(1)
        # S for block j: two groups, one per half of d.
        @inline function scores_async(j)
            k = shared_tile(tk,kstage(slot_of(j)))
            # (A name of the enclosing body assigned here would be boxed.)
            @inbounds half = mma_async(qk,qa[1],operand(qk,OperandB(),k,(Int32(0),Int32(0))))   # fresh: no zeroing
            @inbounds mma_async(qk,qa[2],operand(qk,OperandB(),k,(Int32(64),Int32(0))),half)
        end
        # O plus P V for block j: two groups, one per half of the block, P
        # (the softmax weights) from registers and V transposed.
        @inline function output_async(out,weights,j)
            probabilities = Fragment{BFloat16}(weights)
            v = shared_tile(tv,vstage(slot_of(j)))
            @inbounds out = mma_async(pv,window(probabilities,(0,0),(64,64)),operand(pv,OperandB(),v,(Int32(0),Int32(0))),out)
            @inbounds mma_async(pv,window(probabilities,(0,64),(64,64)),operand(pv,OperandB(),v,(Int32(64),Int32(0))),out)
        end
        if !Overlap || blocks == Int32(1)
            out = zero_accumulator(pv)
            for j in Int32(0):blocks-Int32(1)
                wait(full(slot_of(j)),phase_of(j))
                update = softmax_update(state,finish_mma(wait_mma(scores_async(j)));scale)
                state = update.state
                out = out .* update.rescale                   # the output so far, to the new maxima
                out = wait_mma(output_async(out,update.weights,j))
                arrive!(empty(slot_of(j)))
            end
        else
            # Turns: consumer w waits on barrier w and passes to the other's;
            # the second primes the first turn and skips its last pass.
            @inline turn() = PingPong && ptx"barrier.sync"(warpgroup % UInt32,UInt32(256))
            @inline pass(final) = PingPong && !(final && warpgroup == Int32(2)) &&
                ptx"barrier.arrive"((Int32(3)-warpgroup) % UInt32,UInt32(256))
            PingPong && warpgroup == Int32(2) && ptx"barrier.arrive"(UInt32(1),UInt32(256))
            wait(full(slot_of(Int32(0))),Int32(0))
            turn()
            first_scores = scores_async(Int32(0))
            pass(false)
            update = softmax_update(state,finish_mma(wait_mma(first_scores));scale)
            state, p = update.state, update.weights
            out = zero_accumulator(pv)
            for j in Int32(1):blocks-Int32(1)
                wait(full(slot_of(j)),phase_of(j))
                turn()
                scores = scores_async(j)
                pending = output_async(out,p,j-Int32(1))
                pass(false)
                update = softmax_update(state,finish_mma(wait_mma(scores;newer=2));scale)
                state = update.state
                out = wait_mma(pending) .* update.rescale
                p = update.weights
                arrive!(empty(slot_of(j-Int32(1))))
            end
            turn()
            last_output = output_async(out,p,blocks-Int32(1))
            pass(true)
            out = wait_mma(last_output)
            arrive!(empty(slot_of(blocks-Int32(1))))
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
# consumers finish and store the current one.
function persistent_attention_kernel!(o,tq,tk,tv,seq_kv::Int32,heads::Int32,scale::Float32,::Val{Stages}) where Stages
    qk = WGMMAAtom(BFloat16,128,64)
    pv = WGMMAAtom(BFloat16,128,64)
    qbytes, kbytes, vbytes = transfer_bytes(tq), transfer_bytes(tk), transfer_bytes(tv)
    workspace = @inbounds CuDynamicSharedArray(UInt8,32+16Stages+1024+2qbytes+Stages*(kbytes+vbytes))
    raw = pointer(workspace)
    @inline qfull(b) = Barrier(raw,b)                       # Q buffer b holds a tile's queries
    @inline qempty(b) = Barrier(raw+16,b)                   # Q buffer b may be refilled
    @inline full(s) = Barrier(raw+32,s)
    @inline empty(s) = Barrier(raw+32+8Stages,s)
    start = raw+32+16Stages
    base = start+((UInt32(0)-PTX.smem_addr_u32(start)) & UInt32(1023))
    @inline qstage(b) = base+b*Int32(qbytes)
    @inline kstage(s) = base+Int32(2qbytes)+s*Int32(kbytes+vbytes)
    @inline vstage(s) = kstage(s)+kbytes
    tid = Int32(threadIdx().x)-Int32(1)
    warpgroup = warp_uniform(tid ÷ Int32(128))
    if tid == Int32(0)
        for b in Int32(0):Int32(1)
            initialize!(qfull(b),1)
            initialize!(qempty(b),256)
        end
        for s in Int32(0):Int32(Stages-1)
            initialize!(full(s),1)
            initialize!(empty(s),256)
        end
        fence_barrier_init()
    end
    sync_threads()
    seq = size(o,2) % Int32
    nq = seq ÷ Int32(128)
    tiles = nq*heads
    blocks = seq_kv ÷ Int32(128)
    if warpgroup == Int32(0)
        release_registers(Val(40))
        if tid == Int32(0)
            i = Int32(0); c = Int32(0)                      # tiles and K/V blocks so far
            t = Int32(blockIdx().x)-Int32(1)
            while t < tiles
                qb = i&Int32(1)
                i >= Int32(2) && wait(qempty(qb),((i>>1)-Int32(1))&Int32(1))
                head = unchecked_div(t,nq)
                q0 = unchecked_rem(t,nq)*Int32(128)
                arrive!(qfull(qb),qbytes)
                @inbounds tma_load!(shared_tile(tq,qstage(qb)),tq,(q0,Int32(0),head),qfull(qb))
                for j in Int32(0):blocks-Int32(1)
                    slot = c%Int32(Stages)
                    c >= Int32(Stages) && wait(empty(slot),((c÷Int32(Stages))-Int32(1))&Int32(1))
                    arrive!(full(slot),kbytes+vbytes)
                    @inbounds tma_load!(shared_tile(tk,kstage(slot)),tk,(Int32(0),Int32(128)*j,head),full(slot))
                    @inbounds tma_load!(shared_tile(tv,vstage(slot)),tv,(Int32(128)*j,Int32(0),head),full(slot))
                    c += Int32(1)
                end
                i += Int32(1)
                t += gridDim().x
            end
        end
    else
        acquire_registers(Val(232))
        rows = (warpgroup-Int32(1))*Int32(64)
        thread = tid%Int32(128)
        i = Int32(0); c = Int32(0)
        t = Int32(blockIdx().x)-Int32(1)
        while t < tiles
            qb = i&Int32(1)
            wait(qfull(qb),(i>>1)&Int32(1))
            q = shared_tile(tq,qstage(qb))
            @inbounds qa = (operand(qk,OperandA(),q,(rows,Int32(0))),operand(qk,OperandA(),q,(rows,Int32(64))))
            state = SoftmaxState(finish_mma(zero_accumulator(qk));dims=2)
            out = zero_accumulator(pv)
            for j in Int32(0):blocks-Int32(1)
                slot = c%Int32(Stages)
                wait(full(slot),(c÷Int32(Stages))&Int32(1))
                k = shared_tile(tk,kstage(slot)); v = shared_tile(tv,vstage(slot))
                @inbounds sa = mma_async(qk,qa[1],operand(qk,OperandB(),k,(Int32(0),Int32(0))))
                @inbounds sa = mma_async(qk,qa[2],operand(qk,OperandB(),k,(Int32(64),Int32(0))),sa)
                update = softmax_update(state,finish_mma(wait_mma(sa));scale)
                state = update.state
                out = out .* update.rescale                   # the output so far, to the new maxima
                p = Fragment{BFloat16}(update.weights)
                @inbounds out = mma_async(pv,window(p,(0,0),(64,64)),operand(pv,OperandB(),v,(Int32(0),Int32(0))),out)
                @inbounds out = wait_mma(mma_async(pv,window(p,(0,64),(64,64)),operand(pv,OperandB(),v,(Int32(64),Int32(0))),out))
                arrive!(empty(slot))
                c += Int32(1)
            end
            arrive!(qempty(qb))
            head = unchecked_div(t,nq)
            q0 = unchecked_rem(t,nq)*Int32(128)
            result = Fragment{BFloat16}(softmax_normalize(finish_mma(out),state))
            dst = @inbounds GlobalTile(pointer(o,Int32(1)+head*seq*Int32(128)),Layout((seq,static(128)),(static(128),static(1))),16)
            @inbounds store!(window(dst,(q0+rows,Int32(0)),(64,128)),result,thread)
            i += Int32(1)
            t += gridDim().x
        end
    end
    nothing
end

"""
    hopper_attention(q, k, v; stages=2, persistent=true, overlap=false, pingpong=overlap) -> (kernel, launch, o)

Prepare O = softmax(Q Kᵀ / √d) V for BF16 `q`, `k`, `v` of size
d×seq×(batch·heads), d = 128, sequence lengths multiples of 128. With
`persistent`, one CTA per SM walks the tiles (with the simple schedule).
With `overlap`, each block's softmax runs while the tensor cores multiply the
previous block's probabilities by V, and with `pingpong` the two consumers
take turns at the tensor cores, which needs three stages to pay. On an H100
NVL the persistent form is fastest for short sequences (2048) and the
overlapped ping-pong schedule for long ones (4096 and up), each within a
few percent of the other; run.jl tunes between them.
"""
function hopper_attention(q::CuArray{T,3},k::CuArray{T,3},v::CuArray{T,3};stages::Integer=2,persistent::Bool=true,overlap::Bool=false,pingpong::Bool=overlap,o=nothing) where T
    d, sq, bh = size(q); skv = size(k,2)
    d == 128 && size(k) == size(v) == (d,skv,bh) && sq % 128 == 0 && skv % 128 == 0 ||
        throw(ArgumentError("d must be 128 and sequence lengths multiples of 128"))
    tq = prepare_tma(TMATile(T,(128,128),2),q)                 # 128 query rows × d
    tk = prepare_tma(TMATile(T,(128,128),1),k)                 # d × 128 keys
    tv = prepare_tma(TMATile(T,(128,128),2),v)                 # 128 keys × d
    shmem = 8+16stages+1024+transfer_bytes(tq)+stages*(transfer_bytes(tk)+transfer_bytes(tv))
    shmem <= 227*1024 || throw(ArgumentError("$stages stages need $(cld(shmem,1024)) KiB of shared memory"))
    o = o === nothing ? similar(q) : o
    scale = Float32(1/sqrt(d))
    pingpong && !overlap && throw(ArgumentError("ping-pong turns apply to the overlapped schedule"))
    persistent && overlap && throw(ArgumentError("the persistent form has the simple schedule"))
    if persistent
        shmem += 16+transfer_bytes(tq)
        shmem <= 227*1024 || throw(ArgumentError("$stages stages need $(cld(shmem,1024)) KiB of shared memory"))
        kernel = @cuda launch=false minthreads=384 persistent_attention_kernel!(o,tq,tk,tv,Int32(skv),Int32(bh),scale,Val(stages))
        CUDACore.attributes(kernel.fun)[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem
        blocks = min((sq÷128)*bh,CUDACore.attribute(CUDACore.device(),CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
        launch_persistent() = GC.@preserve tq tk tv kernel(o,tq,tk,tv,Int32(skv),Int32(bh),scale,Val(stages);threads=384,blocks,shmem)
        return kernel, launch_persistent, o
    end
    kernel = @cuda launch=false minthreads=384 attention_kernel!(o,tq,tk,tv,Int32(skv),scale,Val(stages),Val(overlap),Val(pingpong))
    CUDACore.attributes(kernel.fun)[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem
    launch() = GC.@preserve tq tk tv kernel(o,tq,tk,tv,Int32(skv),scale,Val(stages),Val(overlap),Val(pingpong);threads=384,blocks=(sq÷128,bh),shmem)
    kernel, launch, o
end
