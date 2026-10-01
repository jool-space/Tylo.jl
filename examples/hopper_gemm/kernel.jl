using Tylo, PTX, CUDACore, BFloat16s
using Tylo.Layouts: Layout, static
include(joinpath(@__DIR__,"..","launch.jl"))

# A warp-specialized Hopper GEMM, C = Aᵀ B: A is K×M and B is K×N, both
# contiguous along K, in BF16, FP16 or FP8; C is M×N, stored row-major (an
# N×M Julia matrix), in BF16 with FP32 accumulation. A stage holds one
# 128-byte swizzle row of K: 64 elements of BF16, 128 of FP8. Each CTA computes one 128×N tile with three
# warpgroups: the first loads A and B tiles with TMA into `Stages` shared
# buffers; the other two each multiply 64 rows with WGMMA, keeping one group
# in flight while they wait for the next stage.
# With `Cluster = 2`, the two CTAs of a cluster compute the tiles of rows 2m
# and 2m+1 in the same columns: each loads its own A and half of the shared B
# tile (`bh`), multicast into both CTAs' stages. A stage is refilled only
# when both CTAs' consumers have released it, since the peer's multicast
# writes it: one elected thread per consumer warpgroup arrives on the empty
# barriers of both.
function hopper_gemm_kernel!(c,a,b,bh,atom::WGMMAAtom{T,N,KS,1},k::Int32,::Val{Stages},::Val{Group},::Val{Cluster}=Val(1)) where {T,N,KS,Stages,Group,Cluster}
    bytes = transfer_bytes(a)+transfer_bytes(b)
    workspace = @inbounds CuDynamicSharedArray(UInt8,16Stages+1024+Stages*bytes)
    raw = pointer(workspace)
    @inline full(s) = Barrier(raw,s)                    # stage s holds A and B
    @inline empty(s) = Barrier(raw+8Stages,s)           # stage s may be refilled
    start = raw+16Stages
    buffers = start+((UInt32(0)-PTX.smem_addr_u32(start)) & UInt32(1023))
    tid = Int32(threadIdx().x)-Int32(1)
    warpgroup = warp_uniform(tid ÷ Int32(128))
    if tid == Int32(0)
        for s in Int32(0):Int32(Stages-1)
            initialize!(full(s),1)
            initialize!(empty(s),Cluster == 1 ? 256 : 2Cluster)   # consumer threads, or elected per warpgroup and CTA
        end
        fence_barrier_init()
    end
    Cluster == 1 ? sync_threads() : cluster_sync()
    rank = Cluster == 1 ? Int32(0) : cluster_rank()
    m, n = size(c,2) % Int32, size(c,1) % Int32
    # Shapes of any size: the last tiles and stage may be partial, their
    # loads zero-filled past the edges and their stores clipped there.
    tiles_m, tiles_n = cld(m,Int32(128*Cluster)), cld(n,Int32(N))  # in row tiles per cluster
    tiles = tiles_m*tiles_n
    count = cld(k,Int32(KS))
    first_tile = (Int32(blockIdx().x)-Int32(1)) ÷ Int32(Cluster)
    stride = gridDim().x ÷ Int32(Cluster)
    @inline stage(i) = buffers+(i%Int32(Stages))*Int32(bytes)
    # A CTA (or cluster) walks tiles blockIdx, blockIdx+gridDim, ... (one
    # when the grid covers them all). Loads and MMAs count stages across
    # tiles, so the producer fills the next tile's stages while the consumers
    # store.
    if warpgroup == Int32(0)
        release_registers(Val(40))
        if tid == Int32(0)
            i = Int32(0)
            t = first_tile
            while t < tiles
                pm, tn = tile_order(t,tiles_m,tiles_n,Val(Group))
                tm = Int32(Cluster)*pm+rank
                for kk in Int32(0):count-Int32(1)
                    slot = i%Int32(Stages)
                    i >= Int32(Stages) && wait(empty(slot),((i÷Int32(Stages))-Int32(1))&Int32(1))
                    arrive!(full(slot),bytes)
                    @inbounds tma_load!(shared_tile(a,stage(i)),a,(tm*Int32(128),Int32(KS)*kk),full(slot))
                    if Cluster == 1
                        @inbounds tma_load!(shared_tile(b,stage(i)+transfer_bytes(a)),b,(Int32(KS)*kk,tn*Int32(N)),full(slot))
                    else
                        half = stage(i)+transfer_bytes(a)+rank*Int32(transfer_bytes(bh))
                        @inbounds tma_load!(shared_tile(bh,half),bh,(Int32(KS)*kk,tn*Int32(N)+rank*Int32(N÷Cluster)),full(slot),UInt16(1 << Cluster - 1))
                    end
                    i += Int32(1)
                end
                t += stride
            end
        end
    else
        acquire_registers(Val(232))
        rows = (warpgroup-Int32(1))*Int32(64)
        thread = tid%Int32(128)
        @inline operands(i) = @inbounds (operand(atom,OperandA(),shared_tile(a,stage(i)),(rows,Int32(0))),
                                 operand(atom,OperandB(),shared_tile(b,stage(i)+transfer_bytes(a))))
        @inline function release(s)
            if Cluster == 1
                arrive!(empty(s))
            elseif thread == Int32(0)
                for r in Int32(0):Int32(Cluster-1)
                    arrive!(empty(s);cta=r)
                end
            end
        end
        # Tiles at the edges store through bounds; the others need none.
        out = @inbounds GlobalTile(pointer(c),Layout((tiles_m*Int32(128*Cluster),tiles_n*Int32(N)),(n,static(1))),4)
        edge = @inbounds bound(out,(m,n))
        i = Int32(0)
        t = first_tile
        while t < tiles
            pm, tn = tile_order(t,tiles_m,tiles_n,Val(Group))
            tm = Int32(Cluster)*pm+rank
            wait(full(i%Int32(Stages)),(i÷Int32(Stages))&Int32(1))
            acc = mma_async(atom,operands(i)...)                  # a fresh tile: no zeroing
            for kk in Int32(1):count-Int32(1)
                j = i+kk
                wait(full(j%Int32(Stages)),(j÷Int32(Stages))&Int32(1))
                acc = wait_mma(mma_async(atom,operands(j)...,acc),1)
                release((j-Int32(1))%Int32(Stages))              # stage j-1's group is complete
            end
            result = finish_mma(wait_mma(acc))
            release((i+count-Int32(1))%Int32(Stages))
            i += count
            values, origin = Fragment{BFloat16}(result), (tm*Int32(128)+rows,tn*Int32(N))
            if (tm+Int32(1))*Int32(128) <= m && (tn+Int32(1))*Int32(N) <= n
                @inbounds store!(window(out,origin,(64,N)),values,thread)
            else
                @inbounds store!(window(edge,origin,(64,N)),values,thread)
            end
            t += stride
        end
    end
    Cluster == 1 || cluster_sync()                          # no CTA leaves while its peer writes to it
    nothing
end

# Tile `t` in groups of `Group` row tiles, each walked down its rows first,
# so that the CTAs running at once share A and B tiles through L2.
@inline function tile_order(t::Int32,tiles_m::Int32,tiles_n::Int32,::Val{Group}) where Group
    width = Int32(Group)*tiles_n                            # tiles per group of rows
    first = unchecked_div(t,width)*Int32(Group)
    height = min(tiles_m-first,Int32(Group))
    within = unchecked_rem(t,width)
    first+unchecked_rem(within,height), unchecked_div(within,height)
end

"""
    hopper_gemm_launch(a, b; n=256, stages=4, group=8, persistent=true, cluster=1, c=nothing, sms) -> Launch

Prepare C = Aᵀ B for `a` (K×M) and `b` (K×N) on the device, BF16, FP16 or FP8: the tile
width `n`, the number of shared stages, the number of row tiles grouped
for L2 reuse, and whether one CTA per SM (`sms`, the device's by default)
walks the tiles (`persistent`) or one CTA per tile runs, and whether
clusters of two CTAs share B tiles by multicast (`cluster = 2`) are the
configuration. M, N and K may take any sizes TMA can address (K a multiple
of 16 bytes, N of 8): the last tiles overhang, zero-filled and clipped. C is
an N×M matrix (row-major M×N), the launch's result.
"""
function hopper_gemm_launch(a::CuMatrix{T},b::CuMatrix{T};n::Integer=256,stages::Integer=4,group::Integer=8,persistent::Bool=true,
                            cluster::Integer=1,c=nothing,sms::Integer=multiprocessors()) where T
    k, m = size(a); kb, nt = size(b)
    ks = 128 ÷ sizeof(T)                                  # one swizzle row of K per stage
    cluster in (1,2) || throw(ArgumentError("clusters of one or two CTAs"))
    # TMA's strides and C's rows are multiples of 16 bytes; the tiles may overhang.
    k == kb && (k*sizeof(T)) % 16 == 0 && nt % 8 == 0 && k > 0 && m > 0 && nt > 0 ||
        throw(ArgumentError("K spans a multiple of 16 bytes, and N is a multiple of 8"))
    atom = WGMMAAtom(T,n,ks)
    ta = prepare_tma(TMATile(T,(128,ks),2),a)
    tb = prepare_tma(TMATile(T,(ks,n),1),b)
    tbh = cluster == 1 ? tb : prepare_tma(TMATile(T,(ks,n÷cluster),1),b)   # each CTA's share of B
    shmem = 16stages+1024+stages*(transfer_bytes(ta)+transfer_bytes(tb))
    shmem <= 227*1024 || throw(ArgumentError("$stages stages need $(cld(shmem,1024)) KiB of shared memory"))
    c = c === nothing ? CuArray{BFloat16}(undef,nt,m) : c
    tiles = cld(m,128cluster)*cld(nt,n)                    # of a cluster
    blocks = cluster*(persistent ? min(tiles,sms÷cluster) : tiles)
    Launch(hopper_gemm_kernel!,(c,ta,tb,tbh,atom,Int32(k),Val(stages),Val(group),Val(cluster));
           threads=384,blocks,shmem,clustersize=cluster,minthreads=384,maxregs=168)
end
"""
    hopper_gemm(a, b; kwargs...) -> (kernel, launch, c)

[`hopper_gemm_launch`](@ref), compiled: the kernel, a launcher, and C.
"""
function hopper_gemm(a::CuMatrix,b::CuMatrix;kwargs...)
    l = hopper_gemm_launch(a,b;kwargs...)
    kernel = compile(l)
    kernel, () -> l(kernel), l.result
end
