# Kernel definitions shared by wgmma.jl and wgmma_fragments.jl; not a test.
# An independent descriptor test uses ordinary generic-proxy shared stores,
# then explicitly publishes them. Distinct coordinate values expose incorrect
# descriptor origins that a constant-filled matrix would hide.
@inline function wgmma_operand_probe!(out,p::WGMMA64{T,N,K},k0::Int32,::Val{Normalize}=Val(false)) where {T,N,K,Normalize}
    workspace = @inbounds CuDynamicSharedArray(UInt8,32768+1024)
    root = pointer(workspace)
    root += (UInt32(0)-PTX.smem_addr_u32(root)) & UInt32(1023)
    a = shared_tile(TMALoad(T,Val((128,64)),Val(2)),root)
    b = shared_tile(TMALoad(T,Val((64,128)),Val(1)),root+16384)
    tid = Int32(threadIdx().x)-Int32(1)
    for i in tid:Int32(128):Int32(8191)
        r,k = i÷Int32(64),i%Int32(64)
        unsafe_store!(pointer(a,(r,k)),T(Float32((r+Int32(3)*k)%Int32(19)-Int32(9))))
        unsafe_store!(pointer(b,(k,r)),T(Float32((Int32(2)*r+k)%Int32(17)-Int32(8))))
    end
    ptx"fence.proxy.async.shared::cta"()
    sync_threads()
    @inbounds ad = wgmma_operand(p,OperandA(),a,(Int32(64),k0))
    @inbounds bd = wgmma_operand(p,OperandB(),b,(k0,Int32(8)))
    c = finish_mma(wait_mma(mma_async(p,ad,bd,zero_accumulator(p))))
    if Normalize
        scores = c .* (1f0/128f0)
        weights = exp.(scores .- maximum(scores;dims=2))
        c = weights ./ sum(weights;dims=2)
    end
    dst = GlobalTile(pointer(out),@Layout((64, N), (1, 64)))
    store!(dst,c,tid)
    nothing
end
