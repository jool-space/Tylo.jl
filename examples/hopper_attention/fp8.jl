# FP8 attention forward on Hopper: Q, K and V in E4M3 (d = 128), O in BF16.
# The persistent kernel of kernel.jl with FP8 products: S = Q Kᵀ is one FP8
# WGMMA over all of d (128 bytes, one swizzle row), the softmax runs in FP32,
# and P, packed to FP8 in the register layout FP8 WGMMA reads (a quad
# exchange, `pack_operand_a`), multiplies V in one FP8 WGMMA over the block.
# FP8 WGMMA takes no transposed operands, so V comes with the sequence
# contiguous: a seq×d×(batch·heads) array, Vᵀ of the BF16 kernel's layout.
include("kernel.jl")

# Double-buffered Q and a ring of K and V stages, as in the BF16 persistent
# kernel, but without the tile handoff (the tiles go in a fixed order).
@inline fp8_shared(cfg,tq,tk,tv) = SharedLayout((;
    qfull = barriers(2;arrivals=1), qempty = barriers(2;arrivals=256),
    full = barriers(cfg.stages;arrivals=1), empty = barriers(cfg.stages;arrivals=256),
    q = buffers(transfer_bytes(tq),2;align=1024),
    kv = buffers(transfer_bytes(tk)+transfer_bytes(tv),cfg.stages;align=1024)))

function fp8_attention_kernel!(o,tq,tk,tv,seq_kv::Int32,heads::Int32,scale::Float32,cfg::Config)
    T = Float8_E4M3FN
    qk = WGMMAAtom(T,128,128)                   # S: 64 query rows × 128 keys, all of d
    pv = WGMMAAtom(T,128,128)                   # O: 64 query rows × d, all of a block
    qbytes, kbytes, vbytes = transfer_bytes(tq), transfer_bytes(tk), transfer_bytes(tv)
    layout = fp8_shared(cfg,tq,tk,tv)
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
    if warpgroup == Int32(0)
        release_registers(40)
        if tid == Int32(0)
            i = Int32(0); c = Int32(0)
            t = Int32(blockIdx().x)-Int32(1)
            while t < tiles
                b = queries[i]
                acquire(b)
                head = unchecked_div(t,nq)
                q0 = unchecked_rem(t,nq)*Int32(128)
                arrive!(b.full,qbytes)
                @inbounds tma_load!(shared_tile(tq,smem.q[b.slot]),tq,(q0,Int32(0),head),b.full)
                for j in Int32(0):blocks-Int32(1)
                    s = stages[c]
                    acquire(s)
                    at = smem.kv[s.slot]
                    arrive!(s.full,kbytes+vbytes)
                    @inbounds tma_load!(shared_tile(tk,at),tk,(Int32(0),Int32(128)*j,head),s.full)
                    @inbounds tma_load!(shared_tile(tv,at+kbytes),tv,(Int32(128)*j,Int32(0),head),s.full)
                    c += Int32(1)
                end
                i += Int32(1)
                t += gridDim().x
            end
        end
    else
        acquire_registers(232)
        rows = (warpgroup-Int32(1))*Int32(64)
        thread = tid%Int32(128)
        i = Int32(0); c = Int32(0)
        t = Int32(blockIdx().x)-Int32(1)
        while t < tiles
            b = queries[i]
            wait(b)
            @inbounds qa = operand(qk,OperandA(),shared_tile(tq,smem.q[b.slot]),(rows,Int32(0)))
            state = SoftmaxState(zero_accumulator(qk);dims=2)
            out = zero_accumulator(pv)
            for j in Int32(0):blocks-Int32(1)
                s = stages[c]
                wait(s)
                at = smem.kv[s.slot]
                @inbounds scores = wait_mma(mma_async(qk,qa,operand(qk,OperandB(),shared_tile(tk,at))))
                update = softmax_update(state,scores;scale)
                state = update.state
                p = pack_operand_a(pv,update.weights,thread)
                @inbounds out = wait_mma(mma_async(pv,p,operand(pv,OperandB(),shared_tile(tv,at+kbytes)),out .* update.rescale))
                arrive!(s.empty)
                c += Int32(1)
            end
            arrive!(b.empty)
            head = unchecked_div(t,nq)
            q0 = unchecked_rem(t,nq)*Int32(128)
            result = Fragment{BFloat16}(softmax_normalize(out,state))
            dst = @inbounds GlobalTile(pointer(o,Int32(1)+head*seq*Int32(128)),Layout((seq,static(128)),(static(128),static(1))),16)
            @inbounds store!(window(dst,(q0+rows,Int32(0)),(64,128)),result,thread)
            i += Int32(1)
            t += gridDim().x
        end
    end
    nothing
end

"""
    fp8_attention_launch(q, k, vt; stages=4, o=nothing, sms) -> Launch

O = softmax(Q Kᵀ / √d) V for E4M3 `q`, `k` (d×seq×(batch·heads)) and `vt`
(seq×d×(batch·heads), V with the sequence contiguous), d = 128; O in BF16,
the launch's result.
"""
function fp8_attention_launch(q::CuArray{T,3},k::CuArray{T,3},vt::CuArray{T,3};stages::Integer=4,o=nothing,sms::Integer=multiprocessors()) where T
    d, sq, bh = size(q); skv = size(k,2)
    d == 128 && size(k) == (d,skv,bh) && size(vt) == (skv,d,bh) && sq % 128 == 0 && skv % 128 == 0 ||
        throw(ArgumentError("d must be 128, V transposed, and sequence lengths multiples of 128"))
    tq = prepare_tma(TMATile(T,(128,128),2),q)                 # 128 query rows × d
    tk = prepare_tma(TMATile(T,(128,128),1),k)                 # d × 128 keys
    tv = prepare_tma(TMATile(T,(128,128),1),vt)                # 128 keys × d, keys contiguous
    cfg = Config(;stages)
    shmem = shared_bytes(fp8_shared(cfg,tq,tk,tv))
    shmem <= 227*1024 || throw(ArgumentError("$stages stages need $(cld(shmem,1024)) KiB of shared memory"))
    o = o === nothing ? CuArray{BFloat16}(undef,d,sq,bh) : o
    scale = Float32(1/sqrt(d))
    Launch(fp8_attention_kernel!,(o,tq,tk,tv,Int32(skv),Int32(bh),scale,cfg);
           threads=384,blocks=min((sq÷128)*bh,sms),shmem,minthreads=384,maxregs=168)
end
"""
    fp8_attention(q, k, vt; kwargs...) -> (kernel, launch, o)

[`fp8_attention_launch`](@ref), compiled: the kernel, a launcher, and O.
"""
function fp8_attention(q::CuArray,k::CuArray,vt::CuArray;kwargs...)
    l = fp8_attention_launch(q,k,vt;kwargs...)
    kernel = compile(l)
    kernel, () -> l(kernel), l.result
end
