# FP8 attention forward on Hopper: Q, K and V in E4M3 (d = 128), O in BF16.
# The persistent kernel of kernel.jl with FP8 products: S = Q Kᵀ is one FP8
# WGMMA over all of d (128 bytes, one swizzle row), the softmax runs in FP32,
# and P, packed to FP8 in the register layout FP8 WGMMA reads (a quad
# exchange, `pack_operand_a`), multiplies V in one FP8 WGMMA over the block.
# FP8 WGMMA takes no transposed operands, so V comes with the sequence
# contiguous: a seq×d×(batch·heads) array, Vᵀ of the BF16 kernel's layout.
include("kernel.jl")

function fp8_attention_kernel!(o,tq,tk,tv,seq_kv::Int32,heads::Int32,scale::Float32,::Val{Stages}) where Stages
    T = Float8_E4M3FN
    qk = WGMMAAtom(T,128,128)                   # S: 64 query rows × 128 keys, all of d
    pv = WGMMAAtom(T,128,128)                   # O: 64 query rows × d, all of a block
    qbytes, kbytes, vbytes = transfer_bytes(tq), transfer_bytes(tk), transfer_bytes(tv)
    workspace = @inbounds CuDynamicSharedArray(UInt8,32+16Stages+1024+2qbytes+Stages*(kbytes+vbytes))
    raw = pointer(workspace)
    @inline qfull(b) = Barrier(raw,b)
    @inline qempty(b) = Barrier(raw+16,b)
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
            i = Int32(0); c = Int32(0)
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
            @inbounds qa = operand(qk,OperandA(),shared_tile(tq,qstage(qb)),(rows,Int32(0)))
            state = SoftmaxState(zero_accumulator(qk);dims=2)
            out = zero_accumulator(pv)
            for j in Int32(0):blocks-Int32(1)
                slot = c%Int32(Stages)
                wait(full(slot),(c÷Int32(Stages))&Int32(1))
                @inbounds scores = wait_mma(mma_async(qk,qa,operand(qk,OperandB(),shared_tile(tk,kstage(slot)))))
                update = softmax_update(state,scores;scale)
                state = update.state
                p = pack_operand_a(pv,update.weights,thread)
                @inbounds out = wait_mma(mma_async(pv,p,operand(pv,OperandB(),shared_tile(tv,vstage(slot))),out .* update.rescale))
                arrive!(empty(slot))
                c += Int32(1)
            end
            arrive!(qempty(qb))
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
    fp8_attention(q, k, vt; stages=4) -> (kernel, launch, o)

O = softmax(Q Kᵀ / √d) V for E4M3 `q`, `k` (d×seq×(batch·heads)) and `vt`
(seq×d×(batch·heads), V with the sequence contiguous), d = 128; O in BF16.
"""
function fp8_attention(q::CuArray{T,3},k::CuArray{T,3},vt::CuArray{T,3};stages::Integer=4,o=nothing) where T
    d, sq, bh = size(q); skv = size(k,2)
    d == 128 && size(k) == (d,skv,bh) && size(vt) == (skv,d,bh) && sq % 128 == 0 && skv % 128 == 0 ||
        throw(ArgumentError("d must be 128, V transposed, and sequence lengths multiples of 128"))
    tq = prepare_tma(TMATile(T,(128,128),2),q)                 # 128 query rows × d
    tk = prepare_tma(TMATile(T,(128,128),1),k)                 # d × 128 keys
    tv = prepare_tma(TMATile(T,(128,128),1),vt)                # 128 keys × d, keys contiguous
    shmem = 32+16stages+1024+2transfer_bytes(tq)+stages*(transfer_bytes(tk)+transfer_bytes(tv))
    shmem <= 227*1024 || throw(ArgumentError("$stages stages need $(cld(shmem,1024)) KiB of shared memory"))
    o = o === nothing ? CuArray{BFloat16}(undef,d,sq,bh) : o
    scale = Float32(1/sqrt(d))
    kernel = @cuda launch=false minthreads=384 fp8_attention_kernel!(o,tq,tk,tv,Int32(skv),Int32(bh),scale,Val(stages))
    CUDACore.attributes(kernel.fun)[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem
    blocks = min((sq÷128)*bh,CUDACore.attribute(CUDACore.device(),CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
    launch() = GC.@preserve tq tk tv kernel(o,tq,tk,tv,Int32(skv),Int32(bh),scale,Val(stages);threads=384,blocks,shmem)
    kernel, launch, o
end
