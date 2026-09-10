module SoftmaxExample
using Tylo, PTX, CUDACore, BFloat16s
using Tylo.Layouts: Layout, static, coordinate
include("../gemm/kernel.jl")

# Finite valid logits; invalid entries are represented by -Inf. The explicit
# all-masked case avoids -Inf - -Inf and division by zero. No hidden barriers.
@inline function softmax(f)
    shifted=row_map((x,m) -> m == -Inf32 ? -Inf32 : x-m,f,row_max(f))
    weights=map(exp,shifted)
    row_map((x,s) -> s == 0f0 ? 0f0 : x/s,weights,row_sum(weights))
end

# Physical input/output shape is (columns, rows), so each row is contiguous.
# The two versions intentionally share storage and masking semantics.
function lane_softmax_kernel!(output,input,mask,::Val{N}) where N
    row=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)
    width,rows=size(input)
    f=RowFragment(ntuple(Val(N)) do j
        row<=rows && j<=width && (@inbounds mask[j,row]) ? Float32(@inbounds input[j,row]) : -Inf32
    end)
    result=softmax(f)
    ntuple(Val(N)) do j
        if row<=rows && j<=width
            @inbounds output[j,row]=result.data[j]
        end
    end
    nothing
end
function warp_softmax_kernel!(output,input,mask,::Val{N}) where N
    tid=Int32(threadIdx().x)-Int32(1)
    row=(Int32(blockIdx().x)-Int32(1))*(Int32(blockDim().x)÷Int32(32))+tid÷Int32(32)+Int32(1)
    lane=tid%Int32(32)
    width,rows=size(input)
    f=WarpRowFragment(ntuple(Val(N)) do e
        j=lane+Int32(32(e-1))+Int32(1)
        row<=rows && j<=width && (@inbounds mask[j,row]) ? Float32(@inbounds input[j,row]) : -Inf32
    end)
    result=softmax(f)
    ntuple(Val(N)) do e
        j=lane+Int32(32(e-1))+Int32(1)
        if row<=rows && j<=width
            @inbounds output[j,row]=result.data[e]
        end
    end
    nothing
end

@generated function mask_accumulator(a::Tylo.MMAAccumulator{P,N},mask,tid) where {P,N}
    fragments = Expr[]
    for i in 1:N
        words = [quote
            r,c=coordinate(Tylo.Layouts.layout(a),tid,Val($(4(i-1)+j-1)))
            @inbounds mask[r+Int32(1),c+Int32(1)] ? a.data[$i].data[$j] : -Inf32
        end for j in 1:4]
        push!(fragments,:(Tylo.MMAFragment(Float32,Accumulator(),($(words...),))))
    end
    quote
        Base.@inline
        Tylo.MMAAccumulator(Tylo._plan($P),($(fragments...),))
    end
end

# One complete output tile. All its columns belong to a single N warp, so
# softmax covers the whole row. This is a worked MMA epilogue, not attention.
function mma_softmax_kernel!(output,a_data,b_data,mask,config)
    T=eltype(a_data)
    m,n,k=size(config.plan)
    tid=Int32(threadIdx().x)-Int32(1)
    smem=@inbounds CuDynamicSharedArray(T,value(config.a_span)+value(config.b_span))
    sa,sb=shared_stage(pointer(smem),config,Int32(0))
    a=GlobalTile(pointer(a_data),Layout((static(m),static(k)),(static(k),static(1))))
    b=GlobalTile(pointer(b_data),Layout((static(k),static(n)),(static(1),static(k))))
    @inbounds begin
        copy_async!(config.ac,sa,a,tid)
        copy_async!(config.bc,sb,b,tid)
    end
    commit_copies(); wait_copies(Val(0)); sync_threads()
    acc=@inbounds mma(config.plan,sa,sb,zero_accumulator(config.plan),tid)
    result=softmax(mask_accumulator(acc,mask,tid))
    dst=GlobalTile(pointer(output),Layout((static(m),static(n)),(static(1),static(m))))
    @inbounds store!(config.plan,dst,result,tid)
    nothing
end
end
