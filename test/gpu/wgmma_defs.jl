# Kernel definitions shared by wgmma.jl and wgmma_fragments.jl; not a test.
# An independent descriptor test uses ordinary generic-proxy shared stores,
# then explicitly publishes them. Distinct coordinate values expose incorrect
# descriptor origins that a constant-filled matrix would hide.
@inline function wgmma_operand_probe!(out,p::WGMMAAtom{T,N,K},k0::Int32,::Val{Normalize}=Val(false)) where {T,N,K,Normalize}
    workspace = @inbounds CuDynamicSharedArray(UInt8,32768+1024)
    root = pointer(workspace)
    root += (UInt32(0)-PTX.smem_addr_u32(root)) & UInt32(1023)
    a = shared_tile(TMATile(T,(128,64),2),root)
    b = shared_tile(TMATile(T,(64,128),1),root+16384)
    tid = Int32(threadIdx().x)-Int32(1)
    for i in tid:Int32(128):Int32(8191)
        r,k = i÷Int32(64),i%Int32(64)
        @inbounds a[(r,k)] = T(Float32((r+Int32(3)*k)%Int32(19)-Int32(9)))
        @inbounds b[(k,r)] = T(Float32((Int32(2)*r+k)%Int32(17)-Int32(8)))
    end
    ptx"fence.proxy.async.shared::cta"()
    sync_threads()
    @inbounds ad = operand(p,OperandA(),a,(Int32(64),k0))
    @inbounds bd = operand(p,OperandB(),b,(k0,Int32(8)))
    c = finish_mma(wait_mma(mma_async(p,ad,bd,zero_accumulator(p))))
    if Normalize
        scores = c .* (1f0/128f0)
        weights = exp.(scores .- maximum(scores;dims=2))
        c = weights ./ sum(weights;dims=2)
    end
    dst = GlobalTile(pointer(out),@Layout((64, N), (1, 64)))
    @inbounds store!(dst,c,tid)
    nothing
end

# The same product as a K=64 atom, as four chained K=16 issues that keep one
# group in flight, each after an mbarrier wait loop as in a pipeline whose
# stages land one by one: ptxas must keep the WGMMA asynchronous across the
# divergent spin loops.
@inline function wgmma_pipelined_probe!(out,p::WGMMAAtom{T,N,16}) where {T,N}
    workspace = @inbounds CuDynamicSharedArray(UInt8,32768+1024+8)
    root = pointer(workspace)
    stage = Barrier(root+32768+1024)
    root += (UInt32(0)-PTX.smem_addr_u32(root)) & UInt32(1023)
    a = shared_tile(TMATile(T,(128,64),2),root)
    b = shared_tile(TMATile(T,(64,128),1),root+16384)
    tid = Int32(threadIdx().x)-Int32(1)
    tid == Int32(0) && initialize!(stage,128)
    for i in tid:Int32(128):Int32(8191)
        r,k = i÷Int32(64),i%Int32(64)
        @inbounds a[(r,k)] = T(Float32((r+Int32(3)*k)%Int32(19)-Int32(9)))
        @inbounds b[(k,r)] = T(Float32((Int32(2)*r+k)%Int32(17)-Int32(8)))
    end
    fence_async_shared()
    sync_threads()
    c = @inbounds mma_async(p,operand(p,OperandA(),a,(64,0)),operand(p,OperandB(),b,(0,8)),zero_accumulator(p))
    PTX.Utils.@unroll for k in 16:16:48
        arrive!(stage)
        wait(stage,(k÷16-1)&1)
        c = @inbounds mma_async(p,operand(p,OperandA(),a,(64,k)),operand(p,OperandB(),b,(k,8)),c)
        c = wait_mma(c,1)
    end
    dst = GlobalTile(pointer(out),@Layout((64, N), (1, 64)))
    @inbounds store!(dst,finish_mma(wait_mma(c)),tid)
    nothing
end
