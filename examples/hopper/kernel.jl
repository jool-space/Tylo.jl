using Tylo, PTX, CUDACore, BFloat16s
using Tylo.Layouts: @Layout

# A small, complete pipeline, separate from Megakernels' scheduler: one producer
# warp and one consumer warpgroup. Full 64×N outputs, K tail zero-filled by TMA.
function hopper_gemm_kernel!(out,a,b,plan,total_k::Int32,::Val{Stages}) where Stages
    n = size(plan)[2]
    bytes = transfer_bytes(a)+transfer_bytes(b)
    workspace = @inbounds CuDynamicSharedArray(UInt8,1056+Stages*bytes)
    raw = pointer(workspace)
    ready(s) = Barrier(raw,s)          # stage s holds data
    done(s) = Barrier(raw+16,s)        # stage s is free again
    start = raw+32
    buffers = start+((UInt32(0)-PTX.smem_addr_u32(start)) & UInt32(1023))
    tid = Int32(threadIdx().x)-Int32(1)
    if tid == 128
        for s in Int32(0):Int32(Stages-1)
            initialize!(ready(s),1)
            initialize!(done(s),128)
        end
        fence_barrier_init()
    end
    sync_threads()
    count = cld(total_k,Int32(64))
    if tid == 128
        for i in Int32(0):count-Int32(1)
            slot = i%Int32(Stages)
            if i >= Int32(Stages)
                wait(done(slot),(i÷Int32(Stages))-Int32(1))
            end
            ptr = buffers+slot*Int32(bytes)
            arrive!(ready(slot),bytes)
            @inbounds tma_load!(shared_tile(a,ptr),a,(0,64i),ready(slot))
            @inbounds tma_load!(shared_tile(b,ptr+transfer_bytes(a)),b,(64i,0),ready(slot))
        end
    elseif tid < 128
        acc = zero_accumulator(plan)
        for i in Int32(0):count-Int32(1)
            slot = i%Int32(Stages)
            wait(ready(slot),i÷Int32(Stages))
            ptr = buffers+slot*Int32(bytes)
            @inbounds ad = operand(plan,OperandA(),shared_tile(a,ptr))
            @inbounds bd = operand(plan,OperandB(),shared_tile(b,ptr+transfer_bytes(a)))
            acc = wait_mma(mma_async(plan,ad,bd,acc))
            arrive!(done(slot))
        end
        dst = GlobalTile(pointer(out),@Layout((64, n), (1, 64)))
        store!(dst,finish_mma(acc),tid)
    end
    sync_threads()
    if tid == 128
        for s in Int32(0):Int32(Stages-1)
            invalidate!(ready(s))
            invalidate!(done(s))
        end
    end
    nothing
end
