module StreamingAttention
using Tylo, PTX, CUDACore, BFloat16s
using Tylo.Layouts: @Layout, Swizzle, compose, coordinate, cosize

# One CTA owns 64 queries. Four warps own disjoint groups of 16 rows,
# each with all 32 score columns and all 64 output columns.
function configuration()
    atom=MMA16x8x16(BFloat16)
    scores=TiledMMA(atom,Val((4,1)),Val((1,4)),Val(64))
    output=TiledMMA(atom,Val((4,1)),Val((1,8)),Val(32))
    q=compose(Swizzle{3,3,3}(),@Layout((64, 64), (64, 1)))
    k=compose(Swizzle{3,3,3}(),@Layout((64, 32), (1, 64)))
    v=compose(Swizzle{2,3,2}(),@Layout((32, 64), (1, 32)))
    (;scores,output,q,k,v)
end
shared_bytes(::Any)=16384 # Q: 64×64; K: 64×32; V: 32×64, all BF16

@inline function copy_tile!(plan,dst,src,origin,tid,::Val{Axis}) where Axis
    shape=map(Int,size(dst))
    full=origin[1]+shape[1]<=size(src)[1] && origin[2]+shape[2]<=size(src)[2] &&
         src.layout.strides[3-Axis]%8==0 && reinterpret(UInt64,pointer(src,origin))%UInt64(16)==0
    @inbounds if full
        copy_async!(plan,dst,window(src,origin,Val(shape)),tid)
    else
        copy_async!(plan,dst,src,origin,tid)
    end
    nothing
end

@generated function mask_scores(a::Tylo.MMAAccumulator{P,N},mask,tid,row,key,m,n,::Val{Causal}) where {P,N,Causal}
    fragments=Expr[]
    for i in 1:N
        words=[quote
            r,c=coordinate(Tylo.Layouts.layout(a),tid,Val($(4(i-1)+j-1)))
            qr,kc=Int(row)+Int(r),Int(key)+Int(c)
            valid=qr<m && kc<n && (!$Causal || kc<=qr) && (@inbounds mask[kc+1,qr+1])
            valid ? a.data[$i].data[$j]*0.125f0 : -Inf32
        end for j in 1:4]
        push!(fragments,:(Tylo.MMAFragment(Float32,Accumulator(),($(words...),))))
    end
    quote
        Base.@inline
        Tylo.MMAAccumulator(Tylo._plan($P),($(fragments...),))
    end
end

# A concrete bridge between two differently wide accumulators with identical
# row ownership. Each adjacent pair of score atoms becomes one A operand.
# Conversion to BF16 is visible here; the denominator above stays FP32.
@generated function weighted_values(plan::TiledMMA{A,W,R,K},weights::Tylo.MMAAccumulator,
                                    values::SharedTile,out::Tylo.MMAAccumulator,tid) where {A,W,R,K}
    W==(4,1) && R==(1,8) && K==32 || error("this worked kernel uses a fixed PV tile")
    cs=[Symbol(:c,i) for i in 1:8]
    statements=[:($(cs[j])=out.data[$j]) for j in 1:8]
    for k in 0:1
        push!(statements,:(a=pack_operand_a(plan.atom,weights,Val(0),Val($k))))
        for j in 1:8
            push!(statements,:(b=load_b(plan.atom,window(values,(Int32($(16k)),Int32($(8(j-1)))),Val((16,8))),lane)))
            push!(statements,:($(cs[j])=mma(plan.atom,a,b,$(cs[j]))))
        end
    end
    quote
        Base.@inline
        lane=tid%Int32(32)
        @inbounds begin
            $(statements...)
        end
        Tylo.MMAAccumulator(plan,($(cs...),))
    end
end

# Physical arrays: Q(64,M), K(64,N), V(ldv,64), output(64,M), mask(N,M).
# V's leading dimension may include padding; n is its logical key count.
# No scores or probabilities are stored to global memory. All sequence lengths
# are runtime values. One copy stage makes the ownership/reuse barriers explicit.
function attention_kernel!(output,q_data,k_data,v_data,mask,m::Int32,n::Int32,config,causal)
    tid=Int32(threadIdx().x)-Int32(1)
    row=(Int32(blockIdx().x)-Int32(1))*Int32(64)
    memory=@inbounds CuDynamicSharedArray(BFloat16,8192)
    sq=SharedTile(pointer(memory),config.q)
    sk=SharedTile(pointer(memory)+8192,config.k)
    sv=SharedTile(pointer(memory)+12288,config.v)
    q=GlobalTile(pointer(q_data),@inbounds @Layout(($m, 64), (64, 1)))
    k=GlobalTile(pointer(k_data),@inbounds @Layout((64, $n), (1, 64)))
    v=GlobalTile(pointer(v_data),@inbounds @Layout(($n, 64), (1, $(size(v_data,1)))))
    copy_tile!(CopyPlan{(64,64),128,2}(),sq,q,(row,Int32(0)),tid,Val(2))
    commit_copies();wait_copies(Val(0));sync_threads()
    state=SoftmaxState(zero_accumulator(config.scores))
    out=zero_accumulator(config.output)
    for key in Int32(0):Int32(32):n-Int32(1)
        copy_tile!(CopyPlan{(64,32),128,1}(),sk,k,(Int32(0),key),tid,Val(1))
        copy_tile!(CopyPlan{(32,64),128,1}(),sv,v,(key,Int32(0)),tid,Val(1))
        commit_copies();wait_copies(Val(0));sync_threads() # copies visible to every warp
        scores=@inbounds mma(config.scores,sq,sk,zero_accumulator(config.scores),tid)
        update=softmax_update(state,mask_scores(scores,mask,tid,row,key,m,n,causal))
        out=weighted_values(config.output,update.weights,sv,out .* update.rescale,tid)
        state=update.state
        sync_threads() # every warp has finished reading K/V before reuse
    end
    dst=GlobalTile(pointer(output),@inbounds @Layout(($m, 64), (64, 1)))
    @inbounds store!(config.output,dst,softmax_normalize(out,state),(row,Int32(0)),tid)
    nothing
end

function launch!(out,q,k,v,mask;causal=false,config=configuration())
    m,n=size(q,2),size(k,2)
    eltype(q)==eltype(k)==eltype(v)==BFloat16 || throw(ArgumentError("BF16 inputs required"))
    eltype(out)==Float32 && eltype(mask)==Bool || throw(ArgumentError("FP32 output and Boolean mask required"))
    size(q,1)==size(k,1)==size(v,2)==64 || throw(DimensionMismatch("head dimension 64 required"))
    size(v,1)>=n && size(mask)==(n,m) && size(out)==(64,m) || throw(DimensionMismatch("attention shapes differ"))
    all(a -> strides(a)==(1,size(a,1)),(out,q,k,v,mask)) || throw(ArgumentError("contiguous column-major arrays required"))
    0<=m<=typemax(Int32) && 0<=n<=typemax(Int32) || throw(ArgumentError("sequence length exceeds Int32"))
    m==0 && return out
    @cuda threads=128 blocks=cld(m,64) shmem=shared_bytes(config) attention_kernel!(out,q,k,v,mask,Int32(m),Int32(n),config,Val(causal))
    out
end
end
