module StreamingAttention
using Tylo, PTX, CUDACore, BFloat16s
using Tylo.Layouts: @Layout

# One CTA owns 128 queries of one head as two groups of four warps; each
# warp owns 16 rows with all 64 score columns of a key tile and all 64
# output columns. Q is held in registers as A operands. K and V tiles of 64
# keys arrive by TMA into four shared stages, one thread of group 1 issuing
# three tiles ahead while its group is off the tensor pipe. The two groups
# alternate on the tensor pipe through two named barriers: while one group
# issues its MMAs, the other runs its softmax.
function configuration(;causal::Bool=false)
    atom=MMAAtom((16,8,16),BFloat16)
    scores=TiledMMA(atom,(4,1),(1,8),64)
    output=TiledMMA(atom,(4,1),(1,8),64)
    q=TMATile(BFloat16,(64,64),2) # (rows, dims), dims contiguous
    k=TMATile(BFloat16,(64,64),1) # (dims, keys), dims contiguous
    v=TMATile(BFloat16,(64,64),1) # (keys, dims), keys contiguous
    Config(;scores,output,q,k,v,causal)
end
const TILE_BYTES=Int32(8192) # one 64×64 BF16 tile; pointer offsets are bytes
const STAGES=Int32(4)
# Four barriers, then (from a 1024-byte boundary, the swizzle period) two Q
# tiles and the stages, K and V of each.
@inline layout()=SharedLayout((;
    full=barriers(STAGES;arrivals=1),   # stage s holds K and V of tiles t ≡ s (mod STAGES)
    q=buffers(TILE_BYTES,2;align=1024),
    kv=buffers(2TILE_BYTES,STAGES;align=1024)))

# Scale every score; mask by query bound, key bound, causality and an
# optional Boolean mask. Masked scores become -Inf. Mask reads are clamped
# into bounds and combined without short-circuits, so no element branches.
@inline _mask_bit(::Nothing,kc,qr,m,n)=true
@inline _mask_bit(mask,kc,qr,m,n)=@inbounds mask[min(kc,n-Int32(1))+Int32(1),min(qr,m-Int32(1))+Int32(1)]
@inline function mask_scores(a,mask,tid,row,key,m,n,causal::Bool)
    keep=map(coordinates(a,tid)) do (r,c)
        qr,kc=row+r,key+c
        (qr<m) & (kc<n) & (!causal | (kc<=qr)) & _mask_bit(mask,kc,qr,m,n)
    end
    ifelse.(keep,a .* 0.125f0,-Inf32)
end

# Two horizontally adjacent B operands (16 K rows by 16 N columns) as one
# ownership: the ldmatrix plan derived from it is a single x4 load whose
# first two words are the left atom's operand and last two the right's.
@inline function load_b_pair(atom,tile,lane)
    pair=@inbounds load_fragment(repeat(operand_layout(atom,OperandB()),1,2),tile,lane)
    (window(pair,(0,0),(16,8)),window(pair,(0,8),(16,8)))
end

# The atoms of one warp's 16 rows against a 64-wide tile: for each of the
# four K steps, the eight atoms in pairs that share one x4 load of B.
@inline function k_step(atom,a,tile,atoms,k,lane)
    # (Closures passed to ntuple must be inlined to stay in registers.)
    pairs=ntuple(@inline(p -> load_b_pair(atom,@inbounds(window(tile,(Int32(16k),Int32(16(p-1))),(16,16))),lane)),Val(4))
    ntuple(@inline(j -> mma(atom,a,pairs[(j+1)>>1][2-(j&1)],atoms[j])),Val(8))
end
@inline function multiply(atom,a::NTuple{4,Fragment},tile::SharedTile,atoms,lane)
    atoms=k_step(atom,a[1],tile,atoms,0,lane)
    atoms=k_step(atom,a[2],tile,atoms,1,lane)
    atoms=k_step(atom,a[3],tile,atoms,2,lane)
    k_step(atom,a[4],tile,atoms,3,lane)
end
# Q·Kᵀ for one warp's 16 rows against a 64-key tile, with Q's four K-step
# operands already in registers.
@inline scores(plan::TiledMMA,q,keys::SharedTile,lane)=
    Fragment(plan,multiply(plan.atom,q,keys,atom_accumulators(plan,zero_accumulator(plan)),lane))

# The bridge between the two accumulators with identical row ownership:
# each adjacent pair of score atoms becomes one BF16 A operand over 16 keys,
# packed right after the softmax so only 16 registers carry the weights.
@inline packed_weights(plan,weights)=(pack_operand_a(plan,weights,0,0),pack_operand_a(plan,weights,0,1),
                                      pack_operand_a(plan,weights,0,2),pack_operand_a(plan,weights,0,3))
# The output plus the weights times a 64-key tile of V.
@inline weighted_values(plan::TiledMMA,weights,values::SharedTile,out,lane)=
    Fragment(plan,multiply(plan.atom,weights,values,atom_accumulators(plan,out),lane))

# One warp's Q rows as the four K-step A operands, held for the whole stream.
Base.@propagate_inbounds function q_fragments(atom,sq,warp,lane)
    r=warp*Int32(16)
    (load_fragment(atom,OperandA(),window(sq,(r,Int32(0)),(16,16)),lane),load_fragment(atom,OperandA(),window(sq,(r,Int32(16)),(16,16)),lane),
     load_fragment(atom,OperandA(),window(sq,(r,Int32(32)),(16,16)),lane),load_fragment(atom,OperandA(),window(sq,(r,Int32(48)),(16,16)),lane))
end
@inline _head_pointer(a,head)=pointer(a)+Int(head)*(size(a,1)*size(a,2))*sizeof(eltype(a))

# Tile t's K and V land in its stage of the ring.
@inline function load_tile!(kb,vb,smem,s,t,head)
    at=smem.kv[s.slot]
    key=t*Int32(64)
    @inbounds tma_load!(shared_tile(kb,at),kb,(Int32(0),key,head),s.full)
    @inbounds tma_load!(shared_tile(vb,at+TILE_BYTES),vb,(key,Int32(0),head),s.full)
    nothing
end

# Physical arrays: Q(64,M,H), K(64,N,H), V(ldv,64,H) behind TMA bindings,
# output(64,M,H), mask(N,M) or nothing. n is the logical key count. No
# scores or probabilities are stored to global memory.
function attention_kernel!(output,qb,kb,vb,mask,m::Int32,n::Int32,config::Config)
    tid=Int32(threadIdx().x)-Int32(1)
    g=tid>>7               # warp group
    wtid=tid&Int32(127)
    warp=wtid>>5
    lane=tid&Int32(31)
    row0=(Int32(blockIdx().x)-Int32(1))*Int32(128)
    row=row0+g*Int32(64)   # this group's first query row
    head=Int32(blockIdx().y)-Int32(1)
    smem=bind(layout(),pointer(@inbounds CuDynamicSharedArray(UInt8,shared_bytes(layout()))))
    # Tile t's stage: the producer and both groups count tiles. A stage is
    # refilled only once both groups have passed it, which the turns order.
    tiles_ring=Ring(smem.full)
    # Barrier turn[0] admits group 0 to the tensor pipe, turn[1] group 1.
    turn=named_barriers((;turn=named(2;threads=256))).turn
    limit=config.causal ? min(n,row0+Int32(128)) : n
    tiles=(limit+Int32(63))÷Int32(64)
    if tid==Int32(0)
        initialize!(smem)
        fence_barrier_init()
    end
    sync_threads()
    if tid==Int32(0)
        # Both Q tiles and tile 0 complete the first stage's barrier.
        s=tiles_ring[0]
        arrive!(s.full,2TILE_BYTES+(tiles>Int32(0) ? 2TILE_BYTES : Int32(0)))
        @inbounds tma_load!(shared_tile(qb,smem.q[0]),qb,(row0,Int32(0),head),s.full)
        @inbounds tma_load!(shared_tile(qb,smem.q[1]),qb,(row0+Int32(64),Int32(0),head),s.full)
        tiles>Int32(0) && load_tile!(kb,vb,smem,s,Int32(0),head)
        for t in Int32(1):min(tiles,Int32(3))-Int32(1)
            s=tiles_ring[t]
            arrive!(s.full,2TILE_BYTES)
            load_tile!(kb,vb,smem,s,t,head)
        end
    end
    wait(tiles_ring[0])
    atom=config.scores.atom
    sq=shared_tile(qb,smem.q[g])
    qf=@inbounds q_fragments(atom,sq,warp,lane)
    state=SoftmaxState(zero_accumulator(config.scores))
    out=zero_accumulator(config.output)
    w=packed_weights(config.scores,zero_accumulator(config.scores))
    # Block b of a group issues PV of tile b-1 and QK of tile b; the groups
    # take turns, so one group's softmax overlaps the other's MMAs. Each
    # turn counts both groups: one arrives without waiting, the other waits.
    for b in Int32(0):tiles-Int32(1)
        (b>Int32(0) || g==Int32(1)) && wait(@inbounds turn[g])
        b>Int32(0) && (out=weighted_values(config.output,w,shared_tile(vb,smem.kv[tiles_ring[b-Int32(1)].slot]+TILE_BYTES),out,lane))
        wait(tiles_ring[b])
        s=scores(config.scores,qf,shared_tile(kb,smem.kv[tiles_ring[b].slot]),lane)
        arrive!(@inbounds turn[Int32(1)-g])
        if tid==Int32(128) && b+Int32(3)<tiles
            # Tile b+3's stage held tile b-1, whose last readers were both
            # groups' blocks b, and group 1's has just finished.
            next=tiles_ring[b+Int32(3)]
            arrive!(next.full,2TILE_BYTES)
            load_tile!(kb,vb,smem,next,b+Int32(3),head)
        end
        key=b*Int32(64)
        plain=key+Int32(64)<=n && (!config.causal || key<row) && mask === nothing
        s=plain ? s .* 0.125f0 : mask_scores(s,mask,wtid,row,key,m,n,config.causal)
        update=softmax_update(state,s)
        out=out .* update.rescale
        w=packed_weights(config.scores,update.weights)
        state=update.state
    end
    # The last block: PV of the last tile; group 0 admits group 1 once more.
    (g==Int32(1) || tiles>Int32(0)) && wait(@inbounds turn[g])
    tiles>Int32(0) && (out=weighted_values(config.output,w,shared_tile(vb,smem.kv[tiles_ring[tiles-Int32(1)].slot]+TILE_BYTES),out,lane))
    g==Int32(0) && arrive!(@inbounds turn[1])
    dst=@inbounds GlobalTile(_head_pointer(output,head),@Layout(($m, 64), (64, 1)),16)
    result=softmax_normalize(out,state)
    # The last tile of a sequence stores through the bounds.
    @inbounds if row+Int32(64)<=m
        store!(window(dst,(row,Int32(0)),(64,64)),result,wtid)
    else
        store!(window(bound(dst,size(dst)),(row,Int32(0)),(64,64)),result,wtid)
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
    config=merge(config,(;causal))
    kernel=@cuda launch=false attention_kernel!(out,p.q,p.k,p.v,mask,Int32(p.m),Int32(p.n),config)
    CUDACore.attributes(kernel.fun)[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES]=shared_bytes(layout())
    kernel(out,p.q,p.k,p.v,mask,Int32(p.m),Int32(p.n),config;threads=256,blocks=(cld(p.m,128),p.heads),shmem=shared_bytes(layout()))
    out
end
launch!(out,q,k,v,mask=nothing;kw...)=launch!(out,prepare(q,k,pad_values(v)),mask;kw...)
end
