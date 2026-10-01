using Tylo, PTX, CUDACore, BFloat16s
using CUDACore: i32
using Tylo.Layouts: Layout, static
using Tylo: tmem_allocate!, tmem_deallocate!, fence_after_thread_sync, fence_before_thread_sync
include(joinpath(@__DIR__,"..","launch.jl"))

# FlashAttention's forward pass on datacenter Blackwell, O = softmax(Q Kᵀ ·
# scale) V, in the style of FlashAttention-4. Q, K, V and O are
# d×seq×heads(×batch) BF16 arrays, d = 128, contiguous along d and otherwise
# of any strides (TMA descriptors of three or four axes; O's rows are
# addressed from its strides), K and V with fewer heads than Q when query
# heads share them (grouped-query attention).
#
# A persistent CTA of sixteen warps computes 256 query rows at a time, as two
# tiles of 128 whose work interleaves on the tensor cores:
#
#   warps 0-3, 4-7   softmax of tile 0, tile 1: thread t owns row t, reads its
#                    128 scores from TMEM, and writes the probabilities P back
#                    over them in BF16;
#   warps 8-11       correction: rescales O in TMEM when a row's maximum has
#                    grown, and finally normalizes and stores O;
#   warp 12          MMA: issues S = Q Kᵀ and O += P V (from one lane);
#   warp 13          loads: one thread streams Q, then K and V blocks of 128
#                    keys through a ring of shared stages.
#
# TMEM holds S of both tiles (columns 0-255, P over the second half of each)
# and O of both (256-511). Per tile, the MMA warp issues O += P V for the
# previous block, then S = Q Kᵀ for this one into the same columns (the
# tensor pipe runs in order), and commits `sfull`: once S is ready and O
# final up to the previous block. The softmax takes the new row maxima,
# hands the rescale factors to the correction warps through shared memory
# and a named barrier, writes P in four 32-key chunks, and arrives on
# `pready` after three of them, so that most of the next P V runs while it
# writes the last (`plast`). A row's maximum only moves when it grows by
# more than 2^8 (the probabilities stay below 256), so rescales are rare.
#
# The CTAs are persistent and take items (a head's 256 query rows) in
# rounds. When the last round is partial, most CTAs would idle through it:
# its items are split instead, along the keys into `parts` ranges of key
# blocks, one to a CTA. The part that finishes last, as an atomic count
# tells, merges them: it keeps its own in TMEM, and the others write their
# rows' normalized results and base-2 log-sum-exps to scratch slots. In two
# parts, with one CTA per item, the parts fall to the two CTAs of a cluster
# (`Clustered`): each merges one tile, the other's part of it copied into
# its shared memory in one bulk copy.
#
# With `CTAs = 2`, the two CTAs of a cluster compute 512 rows together, each
# its own two tiles, and every product is one CTA-pair instruction (M = 256)
# that the even CTA's MMA warp issues: each CTA holds half of a K block (64
# keys) and half of a V block (64 of d), so a stage is half as large and the
# ring twice as deep. Both CTAs' loads complete on the even CTA's barriers;
# commits arrive in both; the softmax and correction warps of both CTAs
# release P and O to the MMA warp through the even CTA's `paired` and
# `pairlast` barriers.
# A thread's row maximum in the row-per-lane ownership.
@inline _rowmax(x) = only(maximum(x;dims=2))
# A chunk of a row's scores, 32 keys from `first`, with keys past `row` masked.
# (Each key's offset is compared with one count: `first+e <= row` would let
# the compiler hoist `row-e` for all 128 keys out of the loop when `row` is
# invariant.)
@inline function _mask(x,first::Int32,row::Int32)
    n = row-first
    Fragment(ntuple(e -> ifelse(Int32(e-1) <= n,x.data[e],-Inf32),Val(32)),x.ownership)
end
# Chunk `q` of a row's four as probabilities, 2^(c x - m): the middle two
# chunks with the pairs `emulate` on the FMA pipe, as FlashAttention-4
# computes some of each row's exponentials to relieve MUFU.
@inline _chunk(s,q,c,m,cfg) = exp2_approx(muladd.(s[q],c,-m),q in (2,3) ? cfg.emulate : ())
# P's first `cfg.split` chunks, and the rest. (The configuration is passed
# whole: its constants are its type's, which closures keep.)
@inline _first(s,c,m,cfg) = ntuple(@inline(q -> _chunk(s,q,c,m,cfg)),Val(Int(cfg.split)))
@inline _rest(s,c,m,cfg) = ntuple(@inline(q -> _chunk(s,cfg.split+q,c,m,cfg)),Val(4-Int(cfg.split)))

# A tile's part of an item, as staged in shared memory (and kept in
# scratch) for bulk copies, tile i at `buffer`+64i KB: for each 4 columns,
# the tile's 128 rows, 4 values each, so that a warp's lanes (rows) write
# and read 512 contiguous bytes.
@inline _staged(buffer,i) = @inbounds SharedTile(reinterpret(Core.LLVMPtr{Float32,3},buffer)+65536i32*i,
                                                 Layout((static(128),(static(4),static(32))),(static(4),(static(1),static(512)))),16)
# The rows' base-2 log-sum-exps staged after a tile's part.
@inline _logsums(part) = Scalars(reinterpret(Core.LLVMPtr{Float32,3},part+65536i32),128)
# Counts and flags in global memory, shared between CTAs, in the orderings
# of PTX's memory model that the merge relies on.
@inline _count!(p) = ptx"atom.add.relaxed.gpu.global.s32"(p,1i32)   # the count before
@inline _publish!(p) = (ptx"fence.acq_rel.gpu"(); ptx"red.release.gpu.global.add.s32"(p,1i32))
@inline _observe(p) = ptx"ld.acquire.gpu.global.s32"(p)
@inline _reset!(p) = ptx"st.relaxed.gpu.global.s32"(p,0i32)
@inline _count_out!(p) = ptx"atom.add.acq_rel.gpu.global.s32"(p,1i32)   # the count before, ordered both ways
# A value another CTA wrote, from L2 (the SM's L1 is not coherent).
@inline _load_l2(p) = ptx"ld.global.cg.f32"(p)
# A merged row's largest log-sum-exp (online, over the parts), the inverse
# sum of weights, and the factor of this CTA's own part `own`, unnormalized
# with its sum and maximum at `stat`.
@inline function _weights(smem,stat,logsum,r,parts,own)
    l = @inbounds smem.sums[stat]
    mine = @inbounds(smem.maxima[stat])+log2_approx(l)
    top = mine; total = 1f0
    for p in 0i32:parts-1i32
        p == own && continue
        x = _load_l2(pointer(logsum(p))+4i32*r)
        if x > top
            total = muladd(total,exp2_approx(top-x),1f0); top = x
        else
            total += exp2_approx(x-top)
        end
    end
    (top,1f0/total,exp2_approx(mine-top)/(total*l))
end
# The end of a part of item `a`, by the correction warps (named barrier
# `merge`), once O of both tiles is final. The CTA (of each pair) whose part
# finishes last merges the item; the others write their part to its scratch
# slot, normalized, with their rows' base-2 log-sum-exps, and count it
# written, staging it in shared memory (the free K and V stages) for a bulk
# copy.
# Two counts per item and CTA, `finished` and `written`, tell; the merging
# CTA returns them to zero for the next launch. It keeps its own part in
# TMEM (O of tile i at column 256+128i of `words`, with sums and maxima of
# item parity `parity`), and copies each other part into the stages, a bulk
# copy per tile (`smem.copied[i]`): each thread merges its rows, the parts
# weighted by their sums of probabilities, 2^(log-sum-exp - the largest), in
# TMEM until the last, a tile while the next arrives.
@inline function finish_part!(smem,merge,scratch,counter,dst,words,parity,rows,a,parts,own,slots,q0,rank,band,lane,tid,CTAs)
    buffer = smem.kv[0]
    count = pointer(counter,3i32+2i32*(a*CTAs+rank))   # finished; written follows
    first = a*parts
    slot(p) = pointer(scratch,1i32+(first+p)*(32768i32*CTAs))+rank*131072i32
    half = 65536i32                                 # a tile's bytes
    logsum(p) = Scalars(pointer(scratch,1i32+slots*(32768i32*CTAs)+(first+p)*(256i32*CTAs)+rank*256i32),256)
    stat(i) = band+lane+128i32*(i+2i32*parity)
    part(i) = 128i32*i+band+lane                    # this thread's row of the CTA's part
    wait(merge)
    if tid == 256i32
        last = _count!(count) == parts-1i32
        smem.merging[] = Int32(last)
    end
    wait(merge)
    if smem.merging[] == 0i32
        for i in 0i32:1i32
            l = @inbounds smem.sums[stat(i)]
            for q in 0:3
                values = wait_load(@inbounds load_async(rows,window(words,(band,256i32+128i32*i+Int32(32q)),(32,32))))
                @inbounds store!(tile(_staged(buffer,i),(32,32),(band>>5,Int32(q))),values .* (1f0/l),lane)
            end
            @inbounds logsum(own)[part(i)] = @inbounds(smem.maxima[stat(i)])+log2_approx(l)
        end
        fence_async_shared()
        wait(merge)
        if tid == 256i32
            bulk_copy!(slot(own),buffer,131072)
            commit_tma_stores()
            wait_tma_stores(0)
            fence_async_global()
            _publish!(count+4i32)                   # written
        end
        return nothing
    end
    if tid == 256i32
        while _observe(count+4i32) != parts-1i32 end
        _reset!(count); _reset!(count+4i32)
        fence_async_global()
    end
    wait(merge)
    tops = (_weights(smem,stat(0i32),logsum,part(0i32),parts,own),
            _weights(smem,stat(1i32),logsum,part(1i32),parts,own))
    k = 0i32
    for p in 0i32:parts-1i32
        p == own && continue
        if tid == 256i32
            for i in 0i32:1i32
                arrive!(smem.copied[i],half)
                bulk_copy!(buffer+i*half,slot(p)+i*half,half,smem.copied[i])
            end
        end
        weights = (exp2_approx(_load_l2(pointer(logsum(p))+4i32*part(0i32))-tops[1][1])*tops[1][2],
                   exp2_approx(_load_l2(pointer(logsum(p))+4i32*part(1i32))-tops[2][1])*tops[2][2])
        final = k == parts-2i32
        for i in 0i32:1i32
            wait(Ring(smem.copied[i])[k])
            scale = k == 0i32 ? (i == 0i32 ? tops[1][3] : tops[2][3]) : 1f0
            weight = i == 0i32 ? weights[1] : weights[2]
            for q in 0:3
                o = @inbounds window(words,(band,256i32+128i32*i+Int32(32q)),(32,32))
                values = wait_load(@inbounds load_async(rows,o))
                theirs = @inbounds load_fragment(rows,tile(_staged(buffer,i),(32,32),(band>>5,Int32(q))),lane)
                merged = muladd.(weight,theirs,values .* scale)
                if final
                    @inbounds store!(window(dst,(q0+128i32*i+band,Int32(32q)),(32,32)),Fragment{BFloat16}(merged),lane)
                else
                    @inbounds store_async!(o,merged)
                end
            end
        end
        final && break
        wait_stores()
        wait(merge)                                     # the buffer is free for the next part
        k += 1i32
    end
    nothing
end

# The end of a part of an item split in two between the CTAs of a cluster
# (`clustered`): each CTA merges one tile of the item, rank r tile r, with
# the other CTA's part of it. That CTA stages the part in its own shared
# memory (`staged`, free K and V stages: normalized, then its rows' base-2
# log-sum-exps) and, once this CTA has arrived on its `ready` (this CTA's
# Q stages, `landing`, are then free), copies it there in one bulk copy
# that completes on this CTA's `received`: each merges from its own shared
# memory. (The kernel's closing cluster barrier keeps a CTA's staged part
# until the other has received it.)
@inline function finish_pair!(smem,merge,dst,words,parity,rows,q0,band,lane,tid)
    ready, received = smem.copied[0], smem.copied[1]
    staged, landing = smem.kv[1], smem.q[0]
    me = cluster_rank(); other = 1i32-me
    bytes = 65536+512                                   # a tile's part and its rows' log-sum-exps
    r = band+lane                                       # this thread's row of a tile
    stat(i) = r+128i32*(i+2i32*parity)
    tid == 256i32 && arrive!(received,bytes)
    l = @inbounds smem.sums[stat(other)]
    for q in 0:3
        values = wait_load(@inbounds load_async(rows,window(words,(band,256i32+128i32*other+Int32(32q)),(32,32))))
        @inbounds store!(tile(_staged(staged,0),(32,32),(band>>5,Int32(q))),values .* (1f0/l),lane)
    end
    @inbounds _logsums(staged)[r] = @inbounds(smem.maxima[stat(other)])+log2_approx(l)
    fence_async_shared()                                # the staged part, to the bulk copy
    wait(merge)
    if tid == 256i32
        arrive!(ready;cta=other,scope=:cluster)          # this CTA's landing is free
        wait(ready,0;scope=:cluster)                    # the other's is
        bulk_copy!(landing,staged,bytes,received;cta=other)
    end
    wait(received,0)
    theirs = @inbounds _logsums(landing)[r]
    l = @inbounds smem.sums[stat(me)]
    mine = @inbounds(smem.maxima[stat(me)])+log2_approx(l)
    top = max(mine,theirs)
    wm, wo = exp2_approx(mine-top), exp2_approx(theirs-top)
    inverse = 1f0/(wm+wo)
    scale, weight = wm*inverse/l, wo*inverse
    for q in 0:3
        values = wait_load(@inbounds load_async(rows,window(words,(band,256i32+128i32*me+Int32(32q)),(32,32))))
        theirs = @inbounds load_fragment(rows,tile(_staged(landing,0),(32,32),(band>>5,Int32(q))),lane)
        @inbounds store!(window(dst,(q0+128i32*me+band,Int32(32q)),(32,32)),Fragment{BFloat16}(muladd.(weight,theirs,values .* scale)),lane)
    end
    nothing
end


# A tile's origin in a binding of heads (three axes, one batch) or of heads
# and batches (four).
@inline _at(::Tylo.TMABinding{P,D,3},x,y,h,b) where {P,D} = (x,y,h)
@inline _at(::Tylo.TMABinding{P,D,4},x,y,h,b) where {P,D} = (x,y,h,b)

# The kernel's shared memory: its barriers, the TMEM address, the item
# slots, the row statistics, then two Q tiles and the ring of K and V
# stages.
@inline attention_shared(cfg,tq,tk) = SharedLayout((;
    qfull = barriers(2;arrivals=1),                         # tile i's queries have arrived
    qempty = barriers(2;arrivals=1),                        # tile i's queries may be replaced
    sfull = barriers(2;arrivals=1),                         # S of tile i is ready (and O to the previous block)
    pready = barriers(2;arrivals=cfg.ctas == 1 ? 128+4 : 128),  # P's first 96 keys are written and O rescaled
    plast = barriers(2;arrivals=128),                       # P's last 32 keys are written
    ofull = barriers(2;arrivals=1),                         # O of tile i is final
    summed = barriers((2,2);arrivals=128),                  # [i, b]: the row sums of tile i in buffer b are written
    paired = barriers(2;arrivals=8cfg.ctas),                # pready of both CTAs (the even CTA's)
    pairlast = barriers(2;arrivals=4cfg.ctas),              # plast of both CTAs (the even CTA's)
    itemready = barriers(2;arrivals=1),                     # item slot b holds the next item (`dynamic`)
    kvfull = barriers(cfg.stages;arrivals=1),               # stage s holds a K or V block
    kvempty = barriers(cfg.stages;arrivals=1),              # stage s may be refilled
    tmem = scalars(UInt32),
    ids = scalars(Int32,2),                                 # [slot]: an item, or -1
    merging = scalars(Int32),                               # this CTA merges an item's parts
    copied = barriers(2;arrivals=1),                        # tile i of another part has arrived to merge
    factors = scalars(Float32,256),                         # [row, tile]: rescale factor
    # [row, tile, item parity]: row sums and maxima, two items' worth, since
    # the softmax may finish an item of one block before the correction has
    # read the last.
    sums = scalars(Float32,512),
    maxima = scalars(Float32,512),
    q = buffers(transfer_bytes(tq),2;align=1024),
    kv = buffers(transfer_bytes(tk),cfg.stages;align=1024)))

function blackwell_attention_kernel!(o,tq,tk,tv,counter,scratch,seq_kv::Int32,counts::NTuple{3,Int32},scale::Float32,whole::Int32,parts::Int32,
                                     cfg::Config,timeline=nothing,table=nothing)
    CTAs, Split, Causal, Dynamic, Clustered = cfg.ctas, cfg.split, cfg.causal, cfg.dynamic, cfg.clustered
    qk = Tcgen05Atom((128CTAs,128,16),BFloat16)             # S = Q Kᵀ: 128 rows per CTA × 128 keys
    pv = Tcgen05Atom((128CTAs,128,16),BFloat16)             # O += P V: 128 rows per CTA × d
    qbytes, kvbytes = transfer_bytes(tq), transfer_bytes(tk)  # a Q tile; this CTA's share of a K or V block
    layout = attention_shared(cfg,tq,tk)
    smem = bind(layout,pointer(@inbounds CuDynamicSharedArray(UInt8,shared_bytes(layout))))
    kv = Ring(smem.kvfull,smem.kvempty)                     # K and V blocks, in turn
    # Named barriers: handoff[4i+b] pairs softmax warp (tile i, band b) with
    # correction warp b; merge synchronizes the correction warps.
    nbar = named_barriers((;handoff=named(8;threads=64),merge=named(;threads=128)))
    tid = Int32(threadIdx().x)-1i32
    warp = warp_uniform(tid ÷ 32i32)
    lane = tid%32i32
    if tid == 0i32
        initialize!(smem)
        fence_barrier_init()
    end
    warp == 12i32 && tmem_allocate!(qk,pointer(smem.tmem),512)
    fence_before_thread_sync()
    CTAs == 1 && !Clustered ? sync_threads() : cluster_sync()
    fence_after_thread_sync()
    tmem = smem.tmem[]
    rank = CTAs == 1 ? 0i32 : cluster_rank()
    # Commits arrive in both CTAs of a pair; loads complete on the even CTA's
    # barriers; P and O are released to the even CTA's MMA warp.
    @inline commit(b) = CTAs == 1 ? commit_mma(b) : commit_mma(qk,b,UInt16(0b11))
    @inline completion() = CTAs == 1 ? nothing : 0
    @inline function release_to_mma(local_barrier,pair_barrier)
        sync_warp()
        lane == 0i32 && (CTAs == 1 ? arrive!(local_barrier) : arrive!(pair_barrier;cta=0))
    end
    # The whole allocation, as FP32 columns and as BF16 halves of them; S, P
    # and O of each tile are windows of it, so that one register holds TMEM.
    words = TmemTile(Float32,tmem,Layout((static(128),static(512)),(static(1),static(128))))
    halves = TmemTile(BFloat16,tmem,Layout((static(128),static(1024)),(static(1),static(128))))
    @inline scores(i) = @inbounds window(words,(0,128i),(128,128))
    @inline probabilities(i) = @inbounds window(halves,(0,256i+128),(128,128))    # over S's second half
    @inline output(i) = @inbounds window(words,(0,256+128i),(128,128))
    seq = o.seq
    # Sequences: one length of queries and one of keys, unless a `table`
    # packs sequences of their own lengths along the rows (`Varlen`). Its
    # first entry counts a head's items; then each item's sequence and
    # query block, least work first; then each sequence's first query row,
    # queries, first key row and keys.
    Varlen = table !== nothing
    nq = Varlen ? (@inbounds table[1]) : (seq+Int32(256CTAs-1)) ÷ Int32(256CTAs)   # a head's items, the last of a sequence maybe partial
    # Heads of all batches, of one, and query heads per K and V head.
    heads, per_batch, group = counts
    tiles = nq*heads
    # (Read where each role starts, after its register reallocation: a value
    # live across it would be spilled.)
    @inline first_tile() = (Int32(blockIdx().x)-1i32) ÷ Int32(CTAs)
    stride = gridDim().x ÷ Int32(CTAs)
    @inline blocks_of(lk) = (lk+127i32) >> 7           # key blocks, the last maybe partial
    # Item `t`: its head, first query row and key blocks. Causal items walk
    # each head's query blocks longest first, and stop at their diagonal:
    # the last block is the second tile's alone.
    # Causal items go longest first across heads, not within each: heads in
    # sections whose K and V fit in L2 (about 50 MB), and within a section
    # every head's longest block before any shorter one. (Head by head, a
    # long item would still be handed out near the end.)
    fit = max(1i32,unchecked_div(Int32(50_000_000),512i32*seq_kv))*group
    section = min(1i32 << (31i32-leading_zeros(fit) % Int32),heads)
    # Its geometry: head, first query row (of its sequence), and the
    # sequence's first query row, queries, first key row and keys.
    @inline function item(t)
        if Causal
            per = section*nq
            s = unchecked_div(t,per)
            within = t-s*per
            width = min(section,heads-s*section)
            head = s*section+unchecked_rem(within,width)
            block = nq-1i32-unchecked_div(within,width)
        else
            head, block = unchecked_div(t,nq), unchecked_rem(t,nq)
        end
        if Varlen
            sequence = @inbounds table[2i32+2i32*block]
            entry = 2i32+2i32*nq+4i32*sequence
            q0 = (@inbounds table[3i32+2i32*block])*Int32(256CTAs)+rank*256i32
            @inbounds (; head, q0, qs=table[entry], lq=table[entry+1i32], ks=table[entry+2i32], lk=table[entry+3i32])
        else
            (; head, q0=block*Int32(256CTAs)+rank*256i32, qs=0i32, lq=seq, ks=0i32, lk=seq_kv)
        end
    end
    # Causal row r sees keys up to r+lk-lq: the diagonal meets the keys'
    # end (more keys than queries: a prompt after cached keys).
    @inline count(g) = Causal ? min(((g.q0+g.lk-g.lq+255i32) >> 7)+1i32,blocks_of(g.lk)) : blocks_of(g.lk)
    # Unit `u` of the work: item `u` whole or, from `whole` on, one of
    # `parts` parts of an item of the last round, a range of its key blocks
    # whose results go to scratch slot `u-whole`. Its item's geometry, and
    # its first key block, key blocks and slot (-1 for a whole item).
    units = whole+(tiles-whole)*parts
    @inline function unit(u)
        if u < whole
            g = item(u)
            (; g...,b0=0i32,nb=count(g),slot=-1i32)
        else
            w = u-whole
            a = unchecked_div(w,parts)
            g = item(whole+a)
            keys = blocks_of(g.lk)
            b0 = unchecked_div((w-a*parts)*keys,parts)
            (; g...,b0,nb=unchecked_div((w-a*parts+1i32)*keys,parts)-b0,slot=w)
        end
    end
    # Tile i's blocks of a unit: a causal item's first tile ends a block
    # before the second, at its own diagonal, unless the keys end first (the
    # second tile's rows then lie past the sequence).
    @inline early(g) = Causal && ((g.q0+g.lk-g.lq+255i32) >> 7)+1i32 <= blocks_of(g.lk)
    @inline tile_blocks(g,i) = g.nb-(i == 0 && early(g) ? 1i32 : 0i32)
    # The units: every `stride`th from `first_tile()`, or (`Dynamic`, whole
    # items only) handed out by the load thread from a global counter,
    # longest first, through two item slots; -1 ends the work.
    items = Ring(smem.itemready)                            # the item slots: the load thread fills, all read
    @inline next_item(t,n) = Dynamic ? (wait(items[n]); @inbounds(smem.ids[items[n].slot])) : t
    @inline more(t) = 0i32 <= t < units
    c2 = scale*1.442695f0                                   # scores to base-2 exponents (log2 e)
    # The TMEM lanes a warp may access, and a lane's row of them: derived in
    # each role, after its register reallocation (a value live across it
    # would be spilled).
    @inline lanes() = (warp%4i32)*32i32
    # The named barrier pairing softmax warp (tile i, band b) with correction warp b.
    @inline handoff(i) = @inbounds nbar.handoff[4i32*i+warp%4i32]
    # A tile's row sums of an item, in one of two buffers by item parity.
    @inline summed(i) = Ring(selectdim(smem.summed,1,i))
    rows = distribute((32,32),32;vector=32,axis=2)          # TMEM transfers: lane t holds row t, 32 columns
    # With a `Timeline`, each role's first lane records its steps (recorder =
    # warp): a softmax warp at slots 6k.. of its kth block, events 1 waiting
    # for S, 2 S ready, 3 its row maximum, 4 P published but the last chunk,
    # 5 all, 6 summed; a correction warp at 3s.. of its sth step, 10+i
    # waiting for tile i's O, 12+i the factor, 14+i O released; the MMA warp
    # at 8k+4i.. of tile i's kth P V, 20+i waiting for P, 22+i P ready,
    # 24+i P V issued, 26+i the next S issued; the load thread at 2x.. of
    # stage fill x, 30 waiting for the stage, 31 free.
    # (The recorder is derived where it records, and hoisted out of the
    # loops there: a value live across the register reallocation would be
    # spilled.)
    @inline mark(slot,event) = (lane == 0i32 && record!(recorder(timeline,warp),slot,event); nothing)
    if warp < 8i32
        acquire_registers(184)
        band = lanes(); row = band+lane
        i = warp ÷ 4i32
        k = 0i32; n = 0i32                          # blocks and items so far
        t = first_tile()
        while true
            t = next_item(t,n)
            more(t) || break
            m = -Inf32; l = 0f0
            g = unit(t)
            q0, b0, nb = g.q0, g.b0, tile_blocks(g,i)
            shift = g.lk-g.lq                               # causal row r sees keys up to r+shift
            ragged = g.lk & 127i32 != 0i32
            last = blocks_of(g.lk)-1i32
            for j in 0i32:nb-1i32
                mark(6k,1)
                wait(Ring(smem.sfull[i])[k])
                mark(6k+1i32,2)
                fence_after_thread_sync()
                # The row in four loads of 32 keys, then their waits.
                s = wait_load(@inbounds load_async(rows,chunks(window(scores(i),(band,0i32),(32,128)),(32,32))))
                # The maximum reduces each chunk as it lands; the diagonal
                # blocks, where keys past a row's diagonal do not count, mask
                # and reduce again. (Masking first would make every block wait
                # for all four loads at the branch.)
                top = max(_rowmax(s[1]),_rowmax(s[2]),_rowmax(s[3]),_rowmax(s[4]))*c2
                # (and the last block's keys past the sequence, without causality)
                if Causal ? 128i32*j+127i32 > q0+128i32*i+shift : ragged && b0+j == last
                    r = Causal ? q0+128i32*i+row+shift : g.lk-1i32
                    key = 128i32*(b0+j)
                    s = (_mask(s[1],key,r),_mask(s[2],key+32i32,r),_mask(s[3],key+64i32,r),_mask(s[4],key+96i32,r))
                    top = max(_rowmax(s[1]),_rowmax(s[2]),_rowmax(s[3]),_rowmax(s[4]))*c2
                end
                mark(6k+2i32,3)
                factor = 1f0
                if j == 0i32 || top-m > 8f0             # the maximum moves: rescale O and the sum
                    j > 0i32 && (factor = exp2_approx(m-top))
                    m = top
                end
                p = _first(s,c2,m,cfg)
                if j > 0i32
                    @inbounds smem.factors[row+128i32*i] = factor
                    arrive!(handoff(i))
                    l *= factor
                end
                # `Split` chunks of P, published; then the rest and the row
                # sum. Nothing orders pure arithmetic after a store or an
                # arrival, so the work that must wait takes an input through
                # the arrival's token: the compilers cannot hoist it above.
                @inbounds store_async!(chunks(window(probabilities(i),(band,0i32),(32,32Split)),(32,32)),p)
                wait_stores()
                fence_before_thread_sync()
                published = arrive!(smem.pready[i])
                CTAs == 1 || release_to_mma(smem.pready[i],smem.paired[i])
                mark(6k+3i32,4)
                basis = after(published,m)
                rest = _rest(s,c2,basis,cfg)
                @inbounds store_async!(chunks(window(probabilities(i),(band,32Split),(32,128-32Split)),(32,32)),rest)
                wait_stores()
                fence_before_thread_sync()
                published = arrive!(smem.plast[i])
                CTAs == 1 || release_to_mma(smem.plast[i],smem.pairlast[i])
                mark(6k+4i32,5)
                zero = after(published,0f0)
                probs = (p...,rest...)
                l += (chainsum(probs[1],zero)+chainsum(probs[2],zero))+(chainsum(probs[3],zero)+chainsum(probs[4],zero))
                mark(6k+5i32,6)
                k += 1i32
            end
            @inbounds smem.sums[row+128i32*(i+2i32*(n&1i32))] = l
            @inbounds smem.maxima[row+128i32*(i+2i32*(n&1i32))] = m
            arrive!(summed(i)[n].full)
            n += 1i32
            Dynamic || (t += stride)
        end
    elseif warp < 12i32
        release_registers(80)
        band = lanes(); row = band+lane
        k0 = 0i32; k1 = 0i32; n = 0i32          # each tile's blocks, and items so far
        t = first_tile()
        while true
            t = next_item(t,n)
            more(t) || break
            g = unit(t)
            head, q0, nb, w = g.head, g.q0, g.nb, g.slot
            for j in 0i32:nb-1i32
                for i in 0i32:1i32
                    j < tile_blocks(g,i) || continue
                    step = 3i32*(k0+k1)
                    if j > 0i32
                        # The softmax warp hands over the factor once S, and
                        # so O up to the previous block, is complete. (It
                        # cannot hand over twice in a row: its next S needs
                        # this P V, which waits for the correction.)
                        mark(step,10+i)
                        wait(handoff(i))
                        mark(step+1i32,12+i)
                        factor = @inbounds smem.factors[row+128i32*i]
                        if vote_any_sync(0xffffffff,factor < 1f0)
                            fence_after_thread_sync()
                            for q in 0:3
                                part = @inbounds window(output(i),(band,Int32(32q)),(32,32))
                                values = wait_load(@inbounds load_async(rows,part))
                                @inbounds store_async!(part,values .* factor)
                            end
                            wait_stores()
                        end
                        fence_before_thread_sync()
                    end
                    # At an item's first block O is free once the previous
                    # item's epilogue has read it, which released it then.
                    (j > 0i32 || n == 0i32) && release_to_mma(smem.pready[i],smem.paired[i])
                    mark(step+2i32,14+i)
                    i == 0i32 ? (k0 += 1i32) : (k1 += 1i32)
                end
            end
            batch = unchecked_div(head,per_batch)
            at = Int64(head-batch*per_batch)*o.head+Int64(batch)*o.batch+Int64(g.qs)*o.row   # elements to this sequence's first row
            # Rows past the sequence (of a last, partial item) are not stored.
            extent = (g.lq+Int32(256CTAs-1)) ÷ Int32(256CTAs)*Int32(256CTAs)
            dst = @inbounds bound(GlobalTile(o.ptr+2at,Layout((extent,static(128)),(o.row,static(1))),16),(g.lq,nothing))
            if w < 0i32
                for i in 0i32:1i32
                    wait(summed(i)[n])
                    inverse = 1f0/@inbounds(smem.sums[row+128i32*(i+2i32*(n&1i32))])
                    wait(Ring(smem.ofull[i])[n])
                    fence_after_thread_sync()
                    for q in 0:3
                        values = wait_load(@inbounds load_async(rows,window(output(i),(band,Int32(32q)),(32,32))))
                        @inbounds store!(window(dst,(q0+128i32*i+band,Int32(32q)),(32,32)),Fragment{BFloat16}(values .* inverse),lane)
                    end
                    fence_before_thread_sync()
                    release_to_mma(smem.pready[i],smem.paired[i])     # the next item's first P V of this tile may begin
                end
            else
                for i in 0i32:1i32
                    wait(summed(i)[n])
                    wait(Ring(smem.ofull[i])[n])
                end
                fence_after_thread_sync()
                if Clustered
                    finish_pair!(smem,nbar.merge,dst,words,n&1i32,rows,q0,band,lane,tid)
                else
                    a = unchecked_div(w,parts)
                    finish_part!(smem,nbar.merge,scratch,counter,dst,words,n&1i32,rows,a,parts,w-a*parts,units-whole,q0,rank,band,lane,tid,CTAs)
                end
                fence_before_thread_sync()
                for i in 0i32:1i32
                    release_to_mma(smem.pready[i],smem.paired[i])
                end
            end
            n += 1i32
            Dynamic || (t += stride)
        end
    else
        release_registers(64)
        if warp == 12i32 && rank == 0i32            # warp 12: the MMA warp (of the pair)
            # The whole warp runs the role, so that its counters and
            # descriptors are warp-uniform; one lane issues.
            c = 0i32; k0 = 0i32; k1 = 0i32; n = 0i32   # stages, each tile's blocks, items
            @inline ready(x) = (wait(kv[x]); fence_after_thread_sync())
            @inline stage_of(x) = smem.kv[kv[x].slot]
            @inline release(x) = commit(kv[x].empty)
            @inline function pv_step(i,vb,k,accumulate)
                mark(8i32*k+Int32(4i),20+i)
                wait(Ring(CTAs == 1 ? smem.pready[i] : smem.paired[i])[k])
                mark(8i32*k+Int32(4i+1),22+i)
                fence_after_thread_sync()
                @inbounds mma(pv,output(i),window(probabilities(i),(0i32,0i32),(128,32Split)),vb,32Split,accumulate)
                wait(Ring(CTAs == 1 ? smem.plast[i] : smem.pairlast[i])[k])
                fence_after_thread_sync()
                @inbounds mma(pv,output(i),window(probabilities(i),(0i32,Int32(32Split)),(128,128-32Split)),operand(vb,(Int32(32Split),0i32)),128-32Split,true)
                mark(8i32*k+Int32(4i+2),24+i)
            end
            t = first_tile()
            while true
                t = next_item(t,n)
                more(t) || break
                g = unit(t)
                blocks_t, shortened = g.nb, early(g)            # (the first tile's block fewer)
                wait(Ring(smem.qfull[0])[n]); wait(Ring(smem.qfull[1])[n])
                fence_after_thread_sync()
                qa = @inbounds (operand(qk,OperandA(),shared_tile(tq,smem.q[0])),operand(qk,OperandA(),shared_tile(tq,smem.q[1])))
                ready(c)
                kb = @inbounds operand(qk,OperandB(),shared_tile(tk,stage_of(c)))
                PTX.Utils.@unroll for i in 0:1
                    @inbounds mma(qk,scores(i),qa[i+1],kb,128,false)
                    commit(smem.sfull[i])
                end
                release(c)
                c += 1i32
                for j in 1i32:blocks_t-1i32
                    # Each block is awaited where it is first read: V before
                    # the first P V, K before the first Q Kᵀ, behind it. A
                    # causal item's first tile finishes a block early.
                    ready(c)
                    vb = @inbounds operand(pv,OperandB(),shared_tile(tv,stage_of(c)))
                    kb = @inbounds operand(qk,OperandB(),shared_tile(tk,stage_of(c+1i32)))
                    PTX.Utils.@unroll for i in 0:1
                        pv_step(i,vb,i == 0 ? k0 : k1,j > 1i32)
                        if i == 0 && shortened && j == blocks_t-1i32
                            commit(smem.ofull[0])
                            ready(c+1i32)               # the second tile's Q Kᵀ still reads K
                        else
                            i == 0 && ready(c+1i32)
                            @inbounds mma(qk,scores(i),qa[i+1],kb,128,false)
                            commit(smem.sfull[i])
                            mark(8i32*(i == 0 ? k0 : k1)+Int32(4i+3),26+i)
                        end
                        i == 0 ? (k0 += 1i32) : (k1 += 1i32)
                    end
                    release(c); release(c+1i32)
                    c += 2i32
                end
                commit(smem.qempty[0]); commit(smem.qempty[1])
                ready(c)
                vb = @inbounds operand(pv,OperandB(),shared_tile(tv,stage_of(c)))
                PTX.Utils.@unroll for i in 0:1
                    if !(i == 0 && shortened)               # done already
                        pv_step(i,vb,i == 0 ? k0 : k1,blocks_t > 1i32)
                        commit(smem.ofull[i])
                        i == 0 ? (k0 += 1i32) : (k1 += 1i32)
                    end
                end
                release(c)
                c += 1i32; n += 1i32
                Dynamic || (t += stride)
            end
        elseif tid == 416i32                            # warp 13: the load thread
            c = 0i32; n = 0i32
            # Stage `x` of the ring: returns the next.
            @inline function load(x,binding,origin)
                mark(2i32*x,30)
                acquire(kv[x])
                mark(2i32*x+1i32,31)
                rank == 0i32 && arrive!(kv[x].full,CTAs*kvbytes)
                @inbounds tma_load!(shared_tile(binding,smem.kv[kv[x].slot]),binding,origin,kv[x].full;cta=completion())
                x+1i32
            end
            t = first_tile()
            while true
                if Dynamic
                    @inbounds smem.ids[items[n].slot] = more(t) ? t : -1i32
                    arrive!(items[n].full)
                end
                if !more(t)
                    # The last CTA out returns the item counter to zero for
                    # the next launch: every CTA took its last item before
                    # it counted itself out (acquire-release orders them).
                    if Dynamic && _count_out!(pointer(counter,2)) == gridDim().x % Int32-1i32
                        _reset!(pointer(counter,1)); _reset!(pointer(counter,2))
                    end
                    break
                end
                g = unit(t)
                head, q0, b0, blocks_t = g.head, g.q0, g.b0, g.nb
                half = rank*Int32(128÷CTAs)                 # this CTA's keys of K, columns of V
                first = g.ks                                # the sequence's first key row
                batch = unchecked_div(head,per_batch)       # the batch, its query head and their K and V head
                h = head-batch*per_batch
                hk = unchecked_div(h,group)
                for i in 0i32:1i32
                    acquire(Ring(smem.qfull[i],smem.qempty[i])[n])
                    rank == 0i32 && arrive!(smem.qfull[i],CTAs*qbytes)
                    @inbounds tma_load!(shared_tile(tq,smem.q[i]),tq,_at(tq,g.qs+q0+128i32*i,0i32,h,batch),smem.qfull[i];cta=completion())
                    i == 0i32 && (c = load(c,tk,_at(tk,0i32,first+128i32*b0+half,hk,batch)))
                end
                for j in b0+1i32:b0+blocks_t-1i32
                    c = load(c,tv,_at(tv,first+128i32*(j-1i32),half,hk,batch))
                    c = load(c,tk,_at(tk,0i32,first+128i32*j+half,hk,batch))
                end
                c = load(c,tv,_at(tv,first+128i32*(b0+blocks_t-1i32),half,hk,batch))
                n += 1i32
                t = Dynamic ? gridDim().x % Int32+CUDACore.atomic_add!(pointer(counter),1i32) : t+stride
            end
        end
    end
    fence_before_thread_sync()
    CTAs == 1 && !Clustered ? sync_threads() : cluster_sync()   # no TMEM access (nor a merge's read) remains in either CTA
    fence_after_thread_sync()
    warp == 12i32 && tmem_deallocate!(qk,tmem,512)
    nothing
end

"""
    blackwell_attention_launch(q, k, v; stages, ctas=1, causal=false, tail=true, emulate, lengths, kv_lengths, o, sms) -> Launch

O = softmax(Q Kᵀ / √d) V for BF16 `q` (d×seq×heads, or d×seq×heads×batch),
`k` and `v` (d×seq_kv×heads_kv[×batch]), d = 128, sequences of any length:
a CTA computes 256 query rows at a time, of 128 keys at a time, and the
rows and keys past the sequences' ends of a last, partial item or block
are zero-filled, masked and not stored. `ctas = 2` makes CTA pairs compute
512 rows with 256-row products. With `causal`, row r sees keys up to
r + seq_kv - seq: with more keys than queries (a prompt after cached keys),
the diagonal meets the keys' end. With `lengths` (and `kv_lengths`, by default the same),
sequences of their own lengths pack along the rows of d×rows×heads arrays,
as FlashAttention's variable-length form packs them (PyTorch's (rows,
heads, d) tensor is `PermutedDimsArray(x, (1,3,2))`); each attends within
itself. With fewer K and V heads than query
heads (grouped-query attention), every `heads ÷ heads_kv` consecutive query
heads share one. The arrays are `CuArray`s or strided views of them,
contiguous along d, with every other stride a multiple of 16 bytes:
PyTorch's (batch, seq, heads, d) tensor, a d×heads×seq×batch array in
Julia, is `PermutedDimsArray(x, (1,3,2,4))`, and binds in place. O (by
default laid out as `q`) takes the same forms. `stages`
counts ring stages of a K or V block's share (4 by default for one CTA, 8
for pairs, whose stages are half as large). Persistent CTAs (or pairs) take
the items in rounds; with `tail` (not causal), the items of a last, partial
round, or of a first one when there are fewer items than SMs, are split
along the keys among the CTAs that would idle, and merged by the part that
finishes last (or, in two parts, by both, in clusters of two). `emulate` lists the pairs (0-15) of a 32-key chunk whose
exponentials the middle two of a row's four chunks compute on the FMA pipe
rather than MUFU, by default 6 of them (24 of 128 keys, about
FlashAttention-4's share): MUFU would otherwise be as busy as the tensor
pipe. With a `timeline`
(`Timeline(CUDACore.zeros(UInt64,capacity,14,blocks))`), each warp role
records its steps (see the kernel for the events). `sms` is the number of
multiprocessors the persistent CTAs fill, the device's by default. O is the
launch's result.
"""
function blackwell_attention_launch(q::AbstractArray{T,N},k::AbstractArray{T,N},v::AbstractArray{T,N};ctas::Integer=1,stages::Integer=4ctas,split::Integer=3,causal::Bool=false,dynamic::Bool=causal,emulate::Tuple=(3,4,8,9,13,14),tail::Bool=!causal,lengths=nothing,kv_lengths=lengths,o=nothing,timeline=nothing,sms::Integer=multiprocessors()) where {T,N}
    N in (3,4) || throw(ArgumentError("q, k and v are d×seq×heads or d×seq×heads×batch"))
    d, sq, heads = size(q); skv, heads_kv = size(k,2), size(k,3)
    batch = N == 4 ? size(q,4) : 1
    ctas in (1,2) || throw(ArgumentError("one CTA or a CTA pair"))
    split in (1,2,3) || throw(ArgumentError("P is published after one to three of its four chunks"))
    causal && ctas != 1 && throw(ArgumentError("causal attention runs in one CTA"))
    dynamic && ctas != 1 && throw(ArgumentError("dynamic scheduling hands items to single CTAs"))
    size(k) == size(v) && size(k,1) == d && (N == 3 || size(k,4) == batch) && heads % heads_kv == 0 ||
        throw(ArgumentError("k and v share a shape, with q's d and batch, and a number of heads dividing q's"))
    d == 128 || throw(ArgumentError("d must be 128"))
    0 < sq <= typemax(Int32) && 0 < skv <= typemax(Int32) || throw(ArgumentError("sequences are not empty and their lengths fit in Int32"))
    bh = heads*batch
    # Sequences packed along the rows: each's queries and keys.
    lq, lk = lengths === nothing ? ([sq],[skv]) : (collect(Int,lengths),collect(Int,kv_lengths))
    lengths === nothing || N == 3 || throw(ArgumentError("sequences of their own lengths pack along the rows of d×rows×heads arrays"))
    length(lq) == length(lk) && all(>(0),lq) && all(>(0),lk) && sum(lq) == sq && sum(lk) == skv ||
        throw(ArgumentError("lengths and kv_lengths are positive, one each per sequence, and sum to q's and k's rows"))
    causal && any(lq .> lk) && throw(ArgumentError("causal attention needs at least as many keys as queries"))
    tq = prepare_tma(TMATile(T,(128,128),2),q)                 # 128 query rows × d
    tk = prepare_tma(TMATile(T,(128,128÷ctas),1),k)            # d × this CTA's keys of a block
    tv = prepare_tma(TMATile(T,(128,128÷ctas),2),v)            # a block's keys × this CTA's columns of d
    shmem = shared_bytes(attention_shared(Config(;stages,ctas),tq,tk))
    shmem <= 227*1024 || throw(ArgumentError("$stages stages need $(cld(shmem,1024)) KiB of shared memory"))
    o = o === nothing ? _like(q) : o
    size(o) == size(q) || throw(DimensionMismatch("o has q's shape"))
    out = _rows(o)
    scale = Float32(1/sqrt(d))
    # A head's items, and for sequences of their own lengths the table of
    # them (see the kernel): least work first, the order causal items are
    # handed out from the end of.
    work(s,q0) = causal ? min(cld(q0+256+lk[s]-lq[s],128),cld(lk[s],128)) : cld(lk[s],128)
    items = sort!([(s,b) for s in eachindex(lq) for b in 0:cld(lq[s],256ctas)-1];by=((s,b),) -> work(s,256ctas*b))
    table = lengths === nothing ? nothing :
        CuArray(Int32[length(items); reduce(vcat,[[s-1,b] for (s,b) in items]); reduce(vcat,[[sum(lq[1:s-1]),lq[s],sum(lk[1:s-1]),lk[s]] for s in eachindex(lq)])])
    tiles = length(items)*bh
    most = sms÷ctas                                         # CTAs, or pairs, at most
    rounds, rest = divrem(tiles,most)
    parts = tail && !causal && rest > 0 ? min(most÷rest,cld(minimum(lk),128)÷2) : 1
    parts < 2 && (parts = 1)
    whole = parts > 1 ? rounds*most : tiles
    grid = min(whole+(tiles-whole)*parts,most)
    # An item's two parts fall to CTAs 2a and 2a+1: clusters of two merge
    # them in shared memory.
    clustered = ctas == 1 && parts == 2 && iseven(grid) && stages >= 4   # (the staged part fills three stages)
    # The next dynamic item, the CTAs done taking them, then each split
    # item's parts finished and written, per CTA of a pair: the kernel leaves all at zero.
    split_scratch = parts > 1 && !clustered
    counter = CuArray(zeros(Int32,2+(split_scratch ? 2*(tiles-whole)*ctas : 0)))
    scratch = CuArray{Float32}(undef,split_scratch ? (tiles-whole)*parts*256ctas*129 : 1)
    cfg = Config(;stages,ctas,split,causal,dynamic,emulate,clustered)
    Launch(blackwell_attention_kernel!,(out,tq,tk,tv,counter,scratch,Int32(maximum(lk)),map(Int32,(bh,heads,heads÷heads_kv)),scale,Int32(whole),Int32(parts),
                                        cfg,timeline,table);
           threads=512,blocks=ctas*grid,shmem,clustersize=clustered ? 2 : ctas,minthreads=512,maxregs=128,result=o)
end
"""
    blackwell_attention(q, k, v; kwargs...) -> (kernel, launch, o)

[`blackwell_attention_launch`](@ref), compiled: the kernel, a launcher, and O.
"""
function blackwell_attention(q::AbstractArray,k::AbstractArray,v::AbstractArray;kwargs...)
    l = blackwell_attention_launch(q,k,v;kwargs...)
    kernel = compile(l)
    kernel, () -> l(kernel), l.result
end
# O as the kernel writes it: the address of its first element, the query
# length, and the strides in elements of a row, a head and a batch.
_address(a::CUDACore.StridedCuArray) = pointer(a)
_address(a::PermutedDimsArray) = _address(parent(a))
_address(a) = throw(ArgumentError("o is a CuArray, or a strided view or permutation of one"))
function _rows(o::AbstractArray{T}) where T
    address, st = _address(o), strides(o)
    st[1] == 1 && UInt(address) % 16 == 0 && all(s -> (s*sizeof(T)) % 16 == 0,Base.tail(st)) ||
        throw(ArgumentError("o is contiguous along d, 16-byte aligned, and its other strides multiples of 16 bytes"))
    (; ptr=reinterpret(Core.LLVMPtr{T,1},UInt64(address)),seq=Int32(size(o,2)),row=Int32(st[2]),head=Int64(st[3]),batch=Int64(ndims(o) == 4 ? st[4] : 0))
end
# A new array of q's shape laid out as q: permuted as q is.
_like(q::PermutedDimsArray{T,N,perm}) where {T,N,perm} = PermutedDimsArray(similar(parent(q)),perm)
_like(q) = similar(q)
