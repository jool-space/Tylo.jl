module StreamingAttention
using Tylo, PTX, CUDACore, BFloat16s
using PTX: @ptx_str
using Tylo.Layouts: @Layout, Swizzle, compose, coordinate

# One CTA owns 128 queries of one head as two groups of four warps; each
# warp owns 16 rows with all 64 score columns of a key tile and all 64
# output columns. Q is held in registers as A operands. K and V tiles of 64
# keys arrive by TMA into four shared stages, one thread of group 1 issuing
# three tiles ahead while its group is off the tensor pipe. The two groups
# alternate on the tensor pipe through two named barriers: while one group
# issues its MMAs, the other runs its softmax.
function configuration()
    atom=MMAAtom((16,8,16),BFloat16)
    scores=TiledMMA(atom,Val((4,1)),Val((1,8)),Val(64))
    output=TiledMMA(atom,Val((4,1)),Val((1,8)),Val(64))
    q=TMATile(BFloat16,Val((64,64)),Val(2)) # (rows, dims), dims contiguous
    k=TMATile(BFloat16,Val((64,64)),Val(1)) # (dims, keys), dims contiguous
    v=TMATile(BFloat16,Val((64,64)),Val(1)) # (keys, dims), keys contiguous
    (;scores,output,q,k,v)
end
const TILE_BYTES=Int32(8192) # one 64×64 BF16 tile; pointer offsets are bytes
const STAGES=Int32(4)
# Four mbarriers, alignment slack for the 1024-byte swizzle period, two Q
# tiles, then K and V of each stage.
shared_bytes(::Any)=64+1024+(2+2STAGES)*Int(TILE_BYTES)

# Scale every score; mask by query bound, key bound, causality and an
# optional Boolean mask. Masked scores become -Inf. Mask reads are clamped
# into bounds and combined without short-circuits, so no element branches.
@inline _mask_bit(::Nothing,kc,qr,m,n)=true
@inline _mask_bit(mask,kc,qr,m,n)=@inbounds mask[min(kc,n-1)+1,min(qr,m-1)+1]
@generated function mask_scores(a::Fragment{Float32,N},mask,tid,row,key,m,n,::Val{Causal}) where {N,Causal}
    values=[quote
        r,c=coordinate(Tylo.Layouts.layout(a),tid,Val($(e-1)))
        qr,kc=Int(row)+Int(r),Int(key)+Int(c)
        valid=(qr<m) & (kc<n) & ($(!Causal) | (kc<=qr)) & _mask_bit(mask,kc,qr,m,n)
        ifelse(valid,a.data[$e]*0.125f0,-Inf32)
    end for e in 1:N]
    quote
        Base.@inline
        Fragment(($(values...),),Tylo.Layouts.layout(a))
    end
end

# Two horizontally adjacent B operands (16 K rows by 16 N columns) as one
# ownership: the ldmatrix plan derived from it is a single x4 load whose
# first two words are the left atom's operand and last two the right's.
const B_PAIR=Tylo._ownership((16,16),((4,2),(8,16)),((2,1),(2,8),(2,128)))
@inline function load_b_pair(atom,tile,lane)
    pair=@inbounds load_fragment(B_PAIR,tile,lane)
    layout=operand_layout(atom,OperandB())
    (PackedFragment(BFloat16,(pair.data[1],pair.data[2]),layout),PackedFragment(BFloat16,(pair.data[3],pair.data[4]),layout))
end

# Q·Kᵀ for one warp's 16 rows against a 64-key tile, with Q's four K-step
# operands already in registers.
@generated function scores(plan::TiledMMA{A,W,R,K},q::NTuple{4,PackedFragment},keys::SharedTile,lane) where {A,W,R,K}
    W==(4,1) && R==(1,8) && K==64 || error("this worked kernel uses a fixed QK tile")
    cs=[Symbol(:c,j) for j in 1:8]
    statements=[:($(cs[j])=zero_accumulator(plan.atom)) for j in 1:8]
    for k in 0:3, j in 1:2:8
        push!(statements,:((b0,b1)=load_b_pair(plan.atom,window(keys,(Int32($(16k)),Int32($(8(j-1)))),Val((16,16))),lane)))
        push!(statements,:($(cs[j])=mma(plan.atom,q[$(k+1)],b0,$(cs[j]))))
        push!(statements,:($(cs[j+1])=mma(plan.atom,q[$(k+1)],b1,$(cs[j+1]))))
    end
    quote
        Base.@inline
        @inbounds begin
            $(statements...)
        end
        Fragment(($([:($c.data[$i]) for c in cs for i in 1:4]...),),operand_layout(plan,Accumulator()))
    end
end

# The bridge between the two accumulators with identical row ownership:
# each adjacent pair of score atoms becomes one BF16 A operand over 16 keys,
# packed right after the softmax so only 16 registers carry the weights.
@inline packed_weights(atom,weights)=(pack_operand_a(atom,weights,Val(0),Val(0)),pack_operand_a(atom,weights,Val(0),Val(1)),
                                      pack_operand_a(atom,weights,Val(0),Val(2)),pack_operand_a(atom,weights,Val(0),Val(3)))
@generated function weighted_values(plan::TiledMMA{A,W,R,K},weights::NTuple{4,PackedFragment},
                                    values::SharedTile,out::Fragment,lane) where {A,W,R,K}
    W==(4,1) && R==(1,8) && K==64 || error("this worked kernel uses a fixed PV tile")
    cs=[Symbol(:c,i) for i in 1:8]
    statements=[:($(cs[j])=Tylo._atom_accumulator(out,Val($j))) for j in 1:8]
    for k in 0:3, j in 1:2:8
        push!(statements,:((b0,b1)=load_b_pair(plan.atom,window(values,(Int32($(16k)),Int32($(8(j-1)))),Val((16,16))),lane)))
        push!(statements,:($(cs[j])=mma(plan.atom,weights[$(k+1)],b0,$(cs[j]))))
        push!(statements,:($(cs[j+1])=mma(plan.atom,weights[$(k+1)],b1,$(cs[j+1]))))
    end
    quote
        Base.@inline
        @inbounds begin
            $(statements...)
        end
        Fragment(($([:($c.data[$k]) for c in cs for k in 1:4]...),),Tylo.Layouts.layout(out))
    end
end

# One warp's Q rows as the four K-step A operands, held for the whole stream.
Base.@propagate_inbounds function q_fragments(atom,sq,warp,lane)
    r=warp*Int32(16)
    (load_a(atom,window(sq,(r,Int32(0)),Val((16,16))),lane),load_a(atom,window(sq,(r,Int32(16)),Val((16,16))),lane),
     load_a(atom,window(sq,(r,Int32(32)),Val((16,16))),lane),load_a(atom,window(sq,(r,Int32(48)),Val((16,16))),lane))
end
@inline _head_pointer(a,head)=pointer(a)+Int(head)*(size(a,1)*size(a,2))*sizeof(eltype(a))

# Stage s holds K then V of tiles t ≡ s (mod STAGES), after the two Q tiles.
@inline k_pointer(base,t)=base+2TILE_BYTES+(t&Int32(3))*2TILE_BYTES
@inline v_pointer(base,t)=base+3TILE_BYTES+(t&Int32(3))*2TILE_BYTES
@inline function expect!(full,t,bytes)
    ptx"mbarrier.arrive.expect_tx.shared.b64"(full+(t&Int32(3))*Int32(8),UInt32(bytes))
end
@inline function load_tile!(kb,vb,base,full,t,head)
    bar=full+(t&Int32(3))*Int32(8)
    key=t*Int32(64)
    @inbounds tma_load!(shared_tile(kb,k_pointer(base,t)),kb,(Int32(0),key,head),bar)
    @inbounds tma_load!(shared_tile(vb,v_pointer(base,t)),vb,(key,Int32(0),head),bar)
    nothing
end
@inline function wait_tile(full,t)
    bar=full+(t&Int32(3))*Int32(8)
    parity=UInt32((t>>2)&Int32(1))
    while !ptx"mbarrier.try_wait.parity.shared.b64"(bar,parity) end
end

# Physical arrays: Q(64,M,H), K(64,N,H), V(ldv,64,H) behind TMA bindings,
# output(64,M,H), mask(N,M) or nothing. n is the logical key count. No
# scores or probabilities are stored to global memory.
function attention_kernel!(output,qb,kb,vb,mask,m::Int32,n::Int32,config,::Val{Causal}) where Causal
    tid=Int32(threadIdx().x)-Int32(1)
    g=tid>>7               # warp group
    wtid=tid&Int32(127)
    warp=wtid>>5
    lane=tid&Int32(31)
    row0=(Int32(blockIdx().x)-Int32(1))*Int32(128)
    row=row0+g*Int32(64)   # this group's first query row
    head=Int32(blockIdx().y)-Int32(1)
    memory=@inbounds CuDynamicSharedArray(UInt8,shared_bytes(config))
    raw=pointer(memory)
    full=reinterpret(Core.LLVMPtr{UInt64,3},raw)
    base=raw+64+((UInt32(0)-PTX.smem_addr_u32(raw+64))&UInt32(1023))
    limit=Causal ? min(n,row0+Int32(128)) : n
    tiles=(limit+Int32(63))÷Int32(64)
    if tid==Int32(0)
        for s in Int32(0):STAGES-Int32(1)
            ptx"mbarrier.init.shared.b64"(full+s*Int32(8),UInt32(1))
        end
        ptx"fence.proxy.async.shared::cta"()
    end
    sync_threads()
    if tid==Int32(0)
        # Both Q tiles and tile 0 complete the first stage's barrier.
        expect!(full,Int32(0),2TILE_BYTES+(tiles>Int32(0) ? 2TILE_BYTES : Int32(0)))
        @inbounds tma_load!(shared_tile(qb,base),qb,(row0,Int32(0),head),full)
        @inbounds tma_load!(shared_tile(qb,base+TILE_BYTES),qb,(row0+Int32(64),Int32(0),head),full)
        tiles>Int32(0) && load_tile!(kb,vb,base,full,Int32(0),head)
        for t in Int32(1):min(tiles,Int32(3))-Int32(1)
            expect!(full,t,2TILE_BYTES)
            load_tile!(kb,vb,base,full,t,head)
        end
    end
    wait_tile(full,Int32(0))
    atom=config.scores.atom
    sq=shared_tile(qb,base+g*TILE_BYTES)
    qf=@inbounds q_fragments(atom,sq,warp,lane)
    state=SoftmaxState(zero_accumulator(config.scores))
    out=zero_accumulator(config.output)
    w=packed_weights(atom,zero_accumulator(config.scores))
    # Block b of a group issues PV of tile b-1 and QK of tile b; the groups
    # take turns (barrier 1 admits group 0, barrier 2 group 1), so one
    # group's softmax overlaps the other's MMAs. Each barrier counts both
    # groups: one arrives without waiting, the other waits.
    for b in Int32(0):tiles-Int32(1)
        (b>Int32(0) || g==Int32(1)) && ptx"bar.sync"(Int32(1)+g,Int32(256))
        b>Int32(0) && (out=weighted_values(config.output,w,shared_tile(vb,v_pointer(base,b-Int32(1))),out,lane))
        wait_tile(full,b)
        s=scores(config.scores,qf,shared_tile(kb,k_pointer(base,b)),lane)
        ptx"bar.arrive"(Int32(2)-g,Int32(256))
        if tid==Int32(128) && b+Int32(3)<tiles
            # Stage (b+3)&3 held tile b-1, whose last readers were both
            # groups' blocks b, and group 1's has just finished.
            expect!(full,b+Int32(3),2TILE_BYTES)
            load_tile!(kb,vb,base,full,b+Int32(3),head)
        end
        key=b*Int32(64)
        plain=key+Int32(64)<=n && (!Causal || key<row) && mask === nothing
        s=plain ? s .* 0.125f0 : mask_scores(s,mask,wtid,row,key,m,n,Val(Causal))
        update=softmax_update(state,s)
        out=out .* update.rescale
        w=packed_weights(atom,update.weights)
        state=update.state
    end
    # The last block: PV of the last tile; group 0 admits group 1 once more.
    (g==Int32(1) || tiles>Int32(0)) && ptx"bar.sync"(Int32(1)+g,Int32(256))
    tiles>Int32(0) && (out=weighted_values(config.output,w,shared_tile(vb,v_pointer(base,tiles-Int32(1))),out,lane))
    g==Int32(0) && ptx"bar.arrive"(Int32(2),Int32(256))
    dst=@inbounds GlobalTile(_head_pointer(output,head),@Layout(($m, 64), (64, 1)),Val(16))
    result=softmax_normalize(out,state)
    @inbounds if row+Int32(64)<=m
        store!(config.output,window(dst,(row,Int32(0)),Val((64,64))),result,wtid)
    else
        store!(config.output,dst,result,(row,Int32(0)),wtid)
    end
    nothing
end

_batched(a::AbstractArray{T,3}) where T=a
_batched(a::AbstractArray{T,2}) where T=reshape(a,size(a,1),size(a,2),1)
"""
    pad_values(v)

V itself when its leading dimension is a multiple of eight, as TMA requires
of the key stride, otherwise a zero-padded copy with that property.
"""
function pad_values(v)
    ldv=size(v,1)
    ldv%8==0 && return v
    padded=CUDACore.zeros(eltype(v),cld(ldv,8)*8,size(v)[2:end]...)
    padded[1:ldv,:,:]=_batched(v)
    ndims(v)==2 ? reshape(padded,size(padded,1),size(padded,2)) : padded
end
"""
    prepare(q, k, v; config=configuration()) -> bindings

TMA bindings of Q(64,M,H), K(64,N,H) and V(ldv,64,H) for `launch!`; matrices
are one head, and V's leading dimension may include padding above the
logical N of K. Keep the result alive through launches and graph replays;
values may change, addresses and shapes may not.
"""
function prepare(q,k,v;config=configuration())
    eltype(q)==eltype(k)==eltype(v)==BFloat16 || throw(ArgumentError("BF16 inputs required"))
    m,n,heads=size(q,2),size(k,2),size(q,3)
    size(q,1)==size(k,1)==size(v,2)==64 || throw(DimensionMismatch("head dimension 64 required"))
    size(v,1)>=n && size(k,3)==size(v,3)==heads || throw(DimensionMismatch("attention shapes differ"))
    all(a -> strides(a)[1:2]==(1,size(a,1)) && (ndims(a)==2 || strides(a)[3]==size(a,1)*size(a,2)),(q,k,v)) ||
        throw(ArgumentError("contiguous column-major arrays required"))
    0<m<=typemax(Int32) && 0<=n<=typemax(Int32) || throw(ArgumentError("sequence length exceeds Int32 or is empty"))
    n==0 || size(v,1)%8==0 || throw(ArgumentError("V's leading dimension must be a multiple of eight; see pad_values"))
    # Without keys no tile is ever loaded; the K and V bindings then borrow Q.
    kb=prepare_tma(config.k,_batched(n==0 ? q : k))
    vb=n==0 ? prepare_tma(config.v,_batched(q)) : prepare_tma(config.v,_batched(v);bounds=(n,64,heads))
    (;q=prepare_tma(config.q,_batched(q)),k=kb,v=vb,m,n,heads)
end
function launch!(out,p::NamedTuple,mask=nothing;causal=false,config=configuration())
    eltype(out)==Float32 && (mask===nothing || eltype(mask)==Bool) || throw(ArgumentError("FP32 output and Boolean mask required"))
    size(out,1)==64 && size(out,2)==p.m && size(out,3)==p.heads || throw(DimensionMismatch("output is 64 × queries × heads"))
    mask===nothing || size(mask)==(p.n,p.m) || throw(DimensionMismatch("the mask is keys × queries"))
    all(a -> a===nothing || strides(a)[1:2]==(1,size(a,1)),(out,mask)) || throw(ArgumentError("contiguous column-major arrays required"))
    ndims(out)==2 || strides(out)[3]==64p.m || throw(ArgumentError("contiguous heads required"))
    p.heads==0 && return out
    kernel=@cuda launch=false attention_kernel!(out,p.q,p.k,p.v,mask,Int32(p.m),Int32(p.n),config,Val(causal))
    CUDACore.attributes(kernel.fun)[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES]=shared_bytes(config)
    kernel(out,p.q,p.k,p.v,mask,Int32(p.m),Int32(p.n),config,Val(causal);threads=256,blocks=(cld(p.m,128),p.heads),shmem=shared_bytes(config))
    out
end
launch!(out,q,k,v,mask=nothing;kw...)=launch!(out,prepare(q,k,pad_values(v)),mask;kw...)
end
