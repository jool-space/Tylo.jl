using Tylo, PTX, CUDACore, BFloat16s
using Tylo.Layouts: @Layout

# A small, complete pipeline, separate from Megakernels' scheduler: one producer
# warp and one consumer warpgroup. Full 64×N outputs, K tail zero-filled by TMA.
function hopper_gemm_kernel!(out,a,b,plan,total_k::Int32,::Val{Stages}) where Stages
    n = size(plan)[2]
    bytes = transfer_bytes(a)+transfer_bytes(b)
    workspace = @inbounds CuDynamicSharedArray(UInt8,1056+Stages*bytes)
    raw = pointer(workspace)
    ready = reinterpret(Core.LLVMPtr{UInt64,3},raw)
    done = ready+16
    start = raw+32
    buffers = start+((UInt32(0)-PTX.smem_addr_u32(start)) & UInt32(1023))
    tid = Int32(threadIdx().x)-Int32(1)
    if tid == 128
        for s in Int32(0):Int32(Stages-1)
            ptx"mbarrier.init.shared.b64"(ready+s*Int32(8),UInt32(1))
            ptx"mbarrier.init.shared.b64"(done+s*Int32(8),UInt32(128))
        end
        ptx"fence.proxy.async.shared::cta"()
    end
    sync_threads()
    count = cld(total_k,Int32(64))
    if tid == 128
        for i in Int32(0):count-Int32(1)
            slot = i%Int32(Stages)
            if i >= Int32(Stages)
                parity = UInt32(((i÷Int32(Stages))-Int32(1))&Int32(1))
                while !ptx"mbarrier.try_wait.parity.shared.b64"(done+slot*Int32(8),parity) end
            end
            ptr = buffers+slot*Int32(bytes)
            bar = ready+slot*Int32(8)
            ptx"mbarrier.arrive.expect_tx.shared.b64"(bar,UInt32(bytes))
            @inbounds tma_load!(shared_tile(a,ptr),a,(Int32(0),i*Int32(64)),bar)
            @inbounds tma_load!(shared_tile(b,ptr+transfer_bytes(a)),b,(i*Int32(64),Int32(0)),bar)
        end
    elseif tid < 128
        acc = zero_accumulator(plan)
        for i in Int32(0):count-Int32(1)
            slot = i%Int32(Stages)
            parity = UInt32((i÷Int32(Stages))&Int32(1))
            while !ptx"mbarrier.try_wait.parity.shared.b64"(ready+slot*Int32(8),parity) end
            ptr = buffers+slot*Int32(bytes)
            @inbounds ad = wgmma_operand(plan,OperandA(),shared_tile(a,ptr))
            @inbounds bd = wgmma_operand(plan,OperandB(),shared_tile(b,ptr+transfer_bytes(a)))
            acc = wait_mma(mma_async(plan,ad,bd,acc))
            ptx"mbarrier.arrive.shared.b64"(done+slot*Int32(8))
        end
        dst = GlobalTile(pointer(out),@Layout((64, n), (1, 64)))
        store!(dst,finish_mma(acc),tid)
    end
    sync_threads()
    if tid == 128
        for s in Int32(0):Int32(Stages-1)
            ptx"mbarrier.inval.shared.b64"(ready+s*Int32(8))
            ptx"mbarrier.inval.shared.b64"(done+s*Int32(8))
        end
    end
    nothing
end
