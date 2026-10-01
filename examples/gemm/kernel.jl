using Tylo, CUDACore, BFloat16s
using CUDACore: i32
using Tylo.Layouts: @Layout, Swizzle, compose, cosize

function gemm_config(::Type{T}=BFloat16; block=(64,64,32),warps=(2,2),swizzled=true,stages=2,bounds=false,relu=false) where T
    bm,bn,bk = block
    stages in (1,2) || throw(ArgumentError("one or two copy stages required"))
    bm % (16warps[1]) == bn % (8warps[2]) == 0 || throw(ArgumentError("warp tiles must divide block"))
    bk in (16,32,64) || throw(ArgumentError("supported K tiles: 16,32,64"))
    plan = TiledMMA(MMAAtom((16,8,16),T),warps,(bm÷(16warps[1]),bn÷(8warps[2])),bk)
    a = @Layout (bm, bk) (bk, 1)
    b = @Layout (bk, bn) (1, bk)
    bits = trailing_zeros(bk)-3
    sa = swizzled ? compose(Swizzle{bits,3,bits}(),a) : a
    sb = swizzled ? compose(Swizzle{bits,3,bits}(),b) : b
    # Copies deal 16-byte vectors to the plan's threads along the contiguous K.
    vector = 16 ÷ sizeof(T)
    ac = distribute((bm,bk),Tylo.threads(plan);vector,axis=2)
    bc = distribute((bk,bn),Tylo.threads(plan);vector,axis=1)
    Config(;plan,ac,bc,sa,sb,a_span=Int(cosize(sa)),b_span=Int(cosize(sb)),stages,bounds,relu)
end
# The copy stages of A and B, 16-byte aligned as dynamic shared memory is.
@inline gemm_shared(config,::Type{T}) where T = SharedLayout((;
    stage = buffers((config.a_span+config.b_span)*sizeof(T),config.stages;align=16)))

@inline function shared_stage(smem,config,::Type{T},stage) where T
    base = reinterpret(Core.LLVMPtr{T,3},smem.stage[stage])
    @inbounds (SharedTile(base,config.sa,16),SharedTile(base+config.a_span*Int32(sizeof(T)),config.sb,16))
end

# Interior stages copy unbounded tiles; a stage reaching past the matrix
# copies from bounded tiles, which zero-fill what lies outside. Both tiles
# declare the alignment the launch established.
@inline inside(t,origin,shape) = origin[1]+shape[1] <= size(t)[1] && origin[2]+shape[2] <= size(t)[2]

@inline function prefetch_stage!(smem,config,a,b,m,n,k,stage,tid)
    bm,bn,bk = size(config.plan)
    sa,sb = shared_stage(smem,config,eltype(a),stage)
    @inbounds begin
        if !config.bounds || inside(a,(m,k),(bm,bk))
            copy_async!(config.ac,sa,window(a,(m,k),(bm,bk)),tid)
        else
            copy_async!(config.ac,sa,tile(bound(a,size(a)),(bm,bk),(m ÷ Int32(bm),k ÷ Int32(bk))),tid)
        end
        if !config.bounds || inside(b,(k,n),(bk,bn))
            copy_async!(config.bc,sb,window(b,(k,n),(bk,bn)),tid)
        else
            copy_async!(config.bc,sb,tile(bound(b,size(b)),(bk,bn),(k ÷ Int32(bk),n ÷ Int32(bn))),tid)
        end
    end
    commit_copies()
    nothing
end

# 16-byte rows: the base and every leading dimension a multiple of 16 bytes.
@inline aligned16(p,ld) = reinterpret(UInt64,p) % UInt64(16) == 0 && (ld*Int32(sizeof(eltype(p)))) % 16i32 == 0

# This example owns the pipeline. Tylo's operations do not impose CTA roles,
# allocation policy or buffer reuse; the two barriers have distinct purposes.
# Full-tile launches promise 16-byte rows; a ragged launch checks, and
# copies with whatever alignment its operands have.
function tiled_gemm_kernel!(out,a_data,b_data,m::Int32,n::Int32,k::Int32,
        lda::Int32,ldb::Int32,ldc::Int32,config::Config,alpha::Float32)
    la = @inbounds @Layout(($m, $k), ($lda, 1))
    lb = @inbounds @Layout(($k, $n), (1, $ldb))
    d = GlobalTile(pointer(out),@inbounds @Layout(($m, $n), (1, $ldc)))
    pa, pb = pointer(a_data), pointer(b_data)
    if !config.bounds || (aligned16(pa,lda) && aligned16(pb,ldb))
        @inbounds gemm_tiles!(d,GlobalTile(pa,la,16),GlobalTile(pb,lb,16),config,alpha)
    else
        gemm_tiles!(d,GlobalTile(pa,la),GlobalTile(pb,lb),config,alpha)
    end
    nothing
end
@inline function gemm_tiles!(d,a,b,config,alpha::Float32)
    T = eltype(a)
    m, k = size(a)
    n = size(b)[2]
    bm,bn,bk = size(config.plan)
    stages = config.stages
    layout = gemm_shared(config,T)
    smem = bind(layout,pointer(@inbounds CuDynamicSharedArray(UInt8,shared_bytes(layout))))
    tid = Int32(threadIdx().x)-1i32
    row = (Int32(blockIdx().x)-1i32)*Int32(bm)
    col = (Int32(blockIdx().y)-1i32)*Int32(bn)
    iterations = config.bounds ? cld(k,Int32(bk)) : k ÷ Int32(bk)
    for stage in 0i32:min(stages,iterations)-1i32
        prefetch_stage!(smem,config,a,b,row,col,stage*Int32(bk),stage,tid)
    end
    acc = zero_accumulator(config.plan)
    for i in 0i32:iterations-1i32
        stage = i % stages
        if stages == 2 && i+1i32 < iterations
            wait_copies(1)
        else
            wait_copies(0)
        end
        sync_threads() # make each producer's completed copies visible to all warps
        sa,sb = shared_stage(smem,config,T,stage)
        acc = @inbounds mma(config.plan,sa,sb,acc,tid)
        sync_threads() # every warp has finished reading this stage before overwrite
        if i+stages < iterations
            prefetch_stage!(smem,config,a,b,row,col,(i+stages)*Int32(bk),stage,tid)
        end
    end
    result = map(x -> config.relu ? max(alpha*x,0f0) : alpha*x,acc)
    # An edge tile stores through the bounds, converted to the output type.
    @inbounds if !config.bounds || (eltype(d)===Float32 && row+bm<=m && col+bn<=n)
        store!(window(d,(row,col),(bm,bn)),result,tid)
    else
        store!(tile(bound(d,size(d)),(bm,bn),(row ÷ Int32(bm),col ÷ Int32(bn))),Fragment{eltype(d)}(result),tid)
    end
    nothing
end
