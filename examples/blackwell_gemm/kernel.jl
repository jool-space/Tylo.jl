using Tylo, PTX, CUDACore, BFloat16s
using Tylo.Layouts: Layout, static
using Tylo: tmem_allocate!, tmem_deallocate!, fence_after_thread_sync, fence_before_thread_sync, ctas
include(joinpath(@__DIR__,"..","hopper_gemm","kernel.jl"))     # tile_order, Launch

# A warp-specialized, persistent Blackwell GEMM, C = Aᵀ B, with the operand
# conventions of the Hopper GEMM: A is K×M and B is K×N, both contiguous
# along K, in BF16, FP16 or FP8; C is M×N, stored row-major (an N×M Julia
# matrix), in BF16 with FP32 accumulation.
#
# Each CTA walks tiles with three roles. One thread of warp 0 loads A and B
# stages with TMA. Warp 1 issues the tile's tcgen05 MMAs, from one lane, into
# a TMEM accumulator and hands each stage back with `commit_mma`, which
# arrives once the MMAs reading it have completed. The four warps of warpgroup 1 read
# the finished accumulator out of TMEM, lane band by lane band, and store it.
# The accumulator is double-buffered in TMEM (2N of the 512 columns), so the
# epilogue of one tile overlaps the MMAs of the next.
#
# With a CTA-pair atom (M = 256), the two CTAs of a cluster compute a 256×N
# tile together: each loads its 128 rows of A and half of the tile's columns
# of B, and the even CTA's MMA warp multiplies both halves at once. Its
# `full` barriers count the bytes of both CTAs' loads (`tma_load!` with
# `cta = 0`); its commits arrive in both CTAs, releasing each one's stages and
# starting each one's epilogue; both epilogues release the accumulator to it.
# The kernel's shared memory: the stages' barriers, the accumulators'
# barriers, the TMEM address, and the stages of A and B.
@inline blackwell_gemm_shared(cfg,atom,a,b) = SharedLayout((;
    full = barriers(cfg.stages;arrivals=1),                 # stage s holds A and B
    empty = barriers(cfg.stages;arrivals=1),                # stage s may be refilled (tcgen05.commit)
    ready = barriers(2;arrivals=1),                         # accumulator t holds a finished tile (tcgen05.commit)
    drained = barriers(2;arrivals=4ctas(atom)),             # accumulator t may be overwritten: each epilogue warp
    tmem = scalars(UInt32),
    stage = buffers(transfer_bytes(a)+transfer_bytes(b),cfg.stages;align=1024)))

function blackwell_gemm_kernel!(c,a,b,atom::Tcgen05Atom{S},k::Int32,cfg::Config) where S
    M, N, K = S
    CTAs = ctas(atom)
    KS = 4K                                                 # K per stage: one 128-byte swizzle row
    bytes = transfer_bytes(a)+transfer_bytes(b)             # this CTA's share of a stage
    layout = blackwell_gemm_shared(cfg,atom,a,b)
    smem = bind(layout,pointer(@inbounds CuDynamicSharedArray(UInt8,shared_bytes(layout))))
    stages = Ring(smem.full,smem.empty)
    accumulators = Ring(smem.ready,smem.drained)            # double-buffered in TMEM
    tid = Int32(threadIdx().x)-Int32(1)
    warp = warp_uniform(tid ÷ Int32(32))
    if tid == Int32(0)
        initialize!(smem)
        fence_barrier_init()
    end
    warp == Int32(1) && tmem_allocate!(atom,pointer(smem.tmem),2N)
    fence_before_thread_sync()
    CTAs == 1 ? sync_threads() : cluster_sync()
    fence_after_thread_sync()
    tmem = smem.tmem[]
    rank = CTAs == 1 ? Int32(0) : cluster_rank()
    # A pair's commits arrive in both CTAs; multicast needs a cluster launch.
    @inline release(barrier) = CTAs == 1 ? commit_mma(atom,barrier) : commit_mma(atom,barrier,UInt16(0b11))
    # Shapes of any size: the last tiles and stage may be partial, their
    # loads zero-filled past the edges and their stores clipped there.
    m, n = size(c,2) % Int32, size(c,1) % Int32
    tiles_m, tiles_n = cld(m,Int32(M)), cld(n,Int32(N))
    tiles = tiles_m*tiles_n
    count = cld(k,Int32(KS))
    first_tile = (Int32(blockIdx().x)-Int32(1)) ÷ Int32(CTAs)
    stride = gridDim().x ÷ Int32(CTAs)
    if warp == Int32(0)
        if tid == Int32(0)
            i = Int32(0)                                    # stages filled
            t = first_tile
            while t < tiles
                tm, tn = tile_order(t,tiles_m,tiles_n,cfg.group)
                for kk in Int32(0):count-Int32(1)
                    s = stages[i]
                    acquire(s)
                    at = smem.stage[s.slot]
                    if CTAs == 1
                        arrive!(s.full,bytes)
                        @inbounds tma_load!(shared_tile(a,at),a,(tm*Int32(M),Int32(KS)*kk),s.full)
                        @inbounds tma_load!(shared_tile(b,at+transfer_bytes(a)),b,(Int32(KS)*kk,tn*Int32(N)),s.full)
                    else
                        rank == Int32(0) && arrive!(s.full,CTAs*bytes)
                        @inbounds tma_load!(shared_tile(a,at),a,(tm*Int32(M)+rank*Int32(128),Int32(KS)*kk),s.full;cta=0)
                        @inbounds tma_load!(shared_tile(b,at+transfer_bytes(a)),b,(Int32(KS)*kk,tn*Int32(N)+rank*Int32(N÷CTAs)),s.full;cta=0)
                    end
                    i += Int32(1)
                end
                t += stride
            end
        end
    elseif warp == Int32(1)
        if rank == Int32(0)
            # The whole warp runs the role, so that its counters and
            # descriptors are warp-uniform; one lane issues.
            i = Int32(0)                                    # stages consumed
            j = Int32(0)                                    # tiles issued by this CTA
            t = first_tile
            while t < tiles
                u = accumulators[j]
                acquire(u)
                fence_after_thread_sync()
                acc = accumulator(atom,tmem+UInt32(N)*(u.slot%UInt32))
                for kk in Int32(0):count-Int32(1)
                    s = stages[i]
                    wait(s)
                    fence_after_thread_sync()
                    at = smem.stage[s.slot]
                    @inbounds mma(atom,acc,operand(atom,OperandA(),shared_tile(a,at)),
                                  operand(atom,OperandB(),shared_tile(b,at+transfer_bytes(a))),KS,kk != Int32(0))
                    release(s.empty)                        # the stage is free once these MMAs complete
                    i += Int32(1)
                end
                release(u.full)
                j += Int32(1)
                t += stride
            end
        end
    elseif warp >= Int32(4)
        band = (warp-Int32(4))*Int32(32)                    # the TMEM lanes this warp may read
        lane = tid%Int32(32)
        # Tiles at the edges store through bounds; the others need none.
        out = @inbounds GlobalTile(pointer(c),Layout((tiles_m*Int32(M),tiles_n*Int32(N)),(n,static(1))),16)
        edge = @inbounds bound(out,(m,n))
        rows = distribute((32,32),32;vector=32,axis=2)      # lane t holds row t, 32 columns
        j = Int32(0)
        t = first_tile
        while t < tiles
            tm, tn = tile_order(t,tiles_m,tiles_n,cfg.group)
            u = accumulators[j]
            wait(u)
            fence_after_thread_sync()
            acc = accumulator(atom,tmem+UInt32(N)*(u.slot%UInt32))
            row = tm*Int32(M)+rank*Int32(128)+band
            inside = (tm+Int32(1))*Int32(M) <= m && (tn+Int32(1))*Int32(N) <= n
            for col in 0:32:N-32
                values = Fragment{BFloat16}(wait_load(@inbounds load_async(rows,window(acc,(band,Int32(col)),(32,32)))))
                origin = (row,tn*Int32(N)+Int32(col))
                inside ? (@inbounds store!(window(out,origin,(32,32)),values,lane)) : (@inbounds store!(window(edge,origin,(32,32)),values,lane))
            end
            fence_before_thread_sync()
            lane == Int32(0) && (CTAs == 1 ? arrive!(u.empty) : arrive!(u.empty;cta=0))
            j += Int32(1)
            t += stride
        end
    end
    fence_before_thread_sync()
    CTAs == 1 ? sync_threads() : cluster_sync()             # no TMEM access remains in either CTA
    fence_after_thread_sync()
    warp == Int32(1) && tmem_deallocate!(atom,tmem,2N)
    nothing
end

"""
    blackwell_gemm_launch(a, b; n=256, stages=4, group=8, ctas=1, c=nothing, sms) -> Launch

Prepare C = Aᵀ B for `a` (K×M) and `b` (K×N) on the device, BF16, FP16 or
FP8, with one persistent CTA per SM (`sms`, the device's by default): the
tile width `n` (the MMA's N, up to 256), the number of shared stages, the
number of row tiles grouped for L2 reuse, and whether CTA pairs compute
256-row tiles together (`ctas = 2`) are the configuration. M, N and K may
take any sizes TMA can address (K a multiple of 16 bytes, N of 8): the last
tiles overhang, zero-filled and clipped. C is an N×M matrix (row-major M×N),
the launch's result.
"""
function blackwell_gemm_launch(a::CuMatrix{T},b::CuMatrix{T};n::Integer=256,stages::Integer=4,group::Integer=8,ctas::Integer=1,c=nothing,sms::Integer=multiprocessors()) where T
    k, m = size(a); kb, nt = size(b)
    ks = 128 ÷ sizeof(T)                                    # one swizzle row of K per stage
    ctas in (1,2) || throw(ArgumentError("one CTA or a CTA pair"))
    # TMA's strides and C's rows are multiples of 16 bytes; the tiles may overhang.
    k == kb && (k*sizeof(T)) % 16 == 0 && nt % 8 == 0 && k > 0 && m > 0 && nt > 0 ||
        throw(ArgumentError("K spans a multiple of 16 bytes, and N is a multiple of 8"))
    atom = Tcgen05Atom((128ctas,n,32 ÷ sizeof(T)),T)
    ta = prepare_tma(TMATile(T,(128,ks),2),a)
    tb = prepare_tma(TMATile(T,(ks,n÷ctas),1),b)                # this CTA's columns of the tile
    cfg = Config(;stages,group)
    shmem = shared_bytes(blackwell_gemm_shared(cfg,atom,ta,tb))
    shmem <= 227*1024 || throw(ArgumentError("$stages stages need $(cld(shmem,1024)) KiB of shared memory"))
    c = c === nothing ? CuArray{BFloat16}(undef,nt,m) : c
    tiles = cld(m,128ctas)*cld(nt,n)
    Launch(blackwell_gemm_kernel!,(c,ta,tb,atom,Int32(k),cfg);
           threads=256,blocks=ctas*min(tiles,sms÷ctas),shmem,clustersize=ctas,minthreads=256)
end
"""
    blackwell_gemm(a, b; kwargs...) -> (kernel, launch, c)

[`blackwell_gemm_launch`](@ref), compiled: the kernel, a launcher, and C.
"""
function blackwell_gemm(a::CuMatrix,b::CuMatrix;kwargs...)
    l = blackwell_gemm_launch(a,b;kwargs...)
    kernel = compile(l)
    kernel, () -> l(kernel), l.result
end
