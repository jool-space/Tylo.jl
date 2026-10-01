module SoftmaxExample
using Tylo, PTX, CUDACore, BFloat16s
using Tylo.Layouts: @Layout, static
include("../gemm/kernel.jl")
include("ownerships.jl")

# Finite valid logits; invalid entries are represented by -Inf. The explicit
# all-masked case avoids -Inf - -Inf and division by zero. No hidden barriers.
@inline shift_logit(x,m) = m == -Inf32 ? -Inf32 : x-m
@inline normalize_weight(x,s) = s == 0f0 ? 0f0 : x/s
@inline function softmax(f;dims=2)
    m = maximum(f;dims)
    weights = exp.(shift_logit.(f,m))
    normalize_weight.(weights,sum(weights;dims))
end

# Physical input/output shape is (columns, rows), so each row is contiguous.
# The two versions intentionally share storage and masking semantics.
function lane_softmax_kernel!(output,input,mask,::Val{N}) where N
    row=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)
    width,rows=size(input)
    f=Fragment(ntuple(Val(N)) do j
        row<=rows && j<=width && (@inbounds mask[j,row]) ? Float32(@inbounds input[j,row]) : -Inf32
    end, row_per_lane(N))
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
    f=Fragment(ntuple(Val(N)) do e
        j=lane+Int32(32(e-1))+Int32(1)
        row<=rows && j<=width && (@inbounds mask[j,row]) ? Float32(@inbounds input[j,row]) : -Inf32
    end, row_per_warp(N))
    result=softmax(f)
    ntuple(Val(N)) do e
        j=lane+Int32(32(e-1))+Int32(1)
        if row<=rows && j<=width
            @inbounds output[j,row]=result.data[e]
        end
    end
    nothing
end

# Each value masked by the entry at its logical coordinate.
@inline mask_accumulator(a,mask,tid) =
    ifelse.(map(((r,c),) -> @inbounds(mask[r+Int32(1),c+Int32(1)]),@inbounds(coordinates(a,tid))),a,-Inf32)

# One complete output tile. All its columns belong to a single N warp, so
# softmax covers the whole row. This is a worked MMA epilogue, not attention.
function mma_softmax_kernel!(output,a_data,b_data,mask,config)
    T=eltype(a_data)
    m,n,k=size(config.plan)
    tid=Int32(threadIdx().x)-Int32(1)
    layout=gemm_shared(config,T)
    smem=bind(layout,pointer(@inbounds CuDynamicSharedArray(UInt8,shared_bytes(layout))))
    sa,sb=shared_stage(smem,config,T,Int32(0))
    # Device arrays start 256-byte aligned; these rows are 16-byte multiples.
    a=@inbounds GlobalTile(pointer(a_data),@Layout((m, k), (k, 1)),16)
    b=@inbounds GlobalTile(pointer(b_data),@Layout((k, n), (1, k)),16)
    @inbounds begin
        copy_async!(config.ac,sa,a,tid)
        copy_async!(config.bc,sb,b,tid)
    end
    commit_copies(); wait_copies(0); sync_threads()
    acc=@inbounds mma(config.plan,sa,sb,zero_accumulator(config.plan),tid)
    result=softmax(mask_accumulator(acc,mask,tid))
    dst=GlobalTile(pointer(output),@Layout((m, n), (1, m)))
    @inbounds store!(dst,result,tid)
    nothing
end
end
