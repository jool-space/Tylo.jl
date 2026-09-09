using Tylo, CUDACore, BFloat16s
using Tylo.Layouts: Layout, Swizzle, compose, static, cosize

value(::Val{N}) where N = N

function gemm_config(::Type{T}=BFloat16; block=(64,64,32),warps=(2,2),swizzled=true,stages=2) where T
    bm,bn,bk = block
    stages in (1,2) || throw(ArgumentError("one or two copy stages required"))
    bm % (16warps[1]) == bn % (8warps[2]) == 0 || throw(ArgumentError("warp tiles must divide block"))
    bk in (16,32,64) || throw(ArgumentError("supported K tiles: 16,32,64"))
    plan = TiledMMA(MMA16x8x16(T),Val(warps),Val((bm÷(16warps[1]),bn÷(8warps[2]))),Val(bk))
    a = Layout((static(bm),static(bk)),(static(bk),static(1)))
    b = Layout((static(bk),static(bn)),(static(1),static(bk)))
    bits = trailing_zeros(bk)-3
    sa = swizzled ? compose(Swizzle{bits,3,bits}(),a) : a
    sb = swizzled ? compose(Swizzle{bits,3,bits}(),b) : b
    ac = CopyPlan{(bm,bk),Tylo.threads(plan),2}()
    bc = CopyPlan{(bk,bn),Tylo.threads(plan),1}()
    validate_copy(ac,T,sa,a)
    validate_copy(bc,T,sb,b)
    (;plan,ac,bc,sa,sb,a_span=Val(Int(cosize(sa))),b_span=Val(Int(cosize(sb))),stages=Val(stages))
end
shared_bytes(config,::Type{T}) where T = value(config.stages)*(value(config.a_span)+value(config.b_span))*sizeof(T)

@inline function shared_stage(ptr,config,stage)
    a_span,b_span = value(config.a_span),value(config.b_span)
    base = ptr + stage*oftype(stage,(a_span+b_span)*sizeof(eltype(ptr)))
    (SharedTile(base,config.sa),SharedTile(base+a_span*sizeof(eltype(ptr)),config.sb))
end

@inline function prefetch_stage!(ptr,config,a,b,m,n,k,stage,tid)
    bm,bn,bk = size(config.plan)
    sa,sb = shared_stage(ptr,config,stage)
    @inbounds begin
        copy_async!(config.ac,sa,window(a,(m,k),Val((bm,bk))),tid)
        copy_async!(config.bc,sb,window(b,(k,n),Val((bk,bn))),tid)
    end
    commit_copies()
    nothing
end

# This example owns the pipeline. Tylo's operations do not impose CTA roles,
# allocation policy or buffer reuse; the two barriers have distinct purposes.
function tiled_gemm_kernel!(out,a_data,b_data,m::Int32,n::Int32,k::Int32,
        lda::Int32,ldb::Int32,ldc::Int32,config,alpha::Float32,::Val{Relu}) where Relu
    T = eltype(a_data)
    bm,bn,bk = size(config.plan)
    stages = value(config.stages)
    smem = @inbounds CuDynamicSharedArray(T,stages*(value(config.a_span)+value(config.b_span)))
    ptr = pointer(smem)
    a = GlobalTile(pointer(a_data),@inbounds Layout((m,k),(lda,static(1))))
    b = GlobalTile(pointer(b_data),@inbounds Layout((k,n),(static(1),ldb)))
    d = GlobalTile(pointer(out),@inbounds Layout((m,n),(static(1),ldc)))
    tid = Int32(threadIdx().x)-Int32(1)
    row = (Int32(blockIdx().x)-Int32(1))*Int32(bm)
    col = (Int32(blockIdx().y)-Int32(1))*Int32(bn)
    iterations = k ÷ Int32(bk)
    for stage in Int32(0):min(Int32(stages),iterations)-Int32(1)
        prefetch_stage!(ptr,config,a,b,row,col,stage*Int32(bk),stage,tid)
    end
    acc = zero_accumulator(config.plan)
    for i in Int32(0):iterations-Int32(1)
        stage = i % Int32(stages)
        if stages == 2 && i+Int32(1) < iterations
            wait_copies(Val(1))
        else
            wait_copies(Val(0))
        end
        sync_threads() # make each producer's completed copies visible to all warps
        sa,sb = shared_stage(ptr,config,stage)
        acc = @inbounds mma(config.plan,sa,sb,acc,tid)
        sync_threads() # every warp has finished reading this stage before overwrite
        if i+Int32(stages) < iterations
            prefetch_stage!(ptr,config,a,b,row,col,(i+Int32(stages))*Int32(bk),stage,tid)
        end
    end
    result = map(x -> Relu ? max(alpha*x,0f0) : alpha*x,acc)
    @inbounds store!(config.plan,window(d,(row,col),Val((bm,bn))),result,tid)
    nothing
end
