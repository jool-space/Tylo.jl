using Tylo, PTX, CUDACore, BFloat16s
using Tylo.Layouts: Layout, static

# A warp-specialized Hopper GEMM, C = Aᵀ B: A is K×M and B is K×N, both
# contiguous along K, in BF16, FP16 or FP8; C is M×N, stored row-major (an
# N×M Julia matrix), in BF16 with FP32 accumulation. A stage holds one
# 128-byte swizzle row of K: 64 elements of BF16, 128 of FP8. Each CTA computes one 128×N tile with three
# warpgroups: the first loads A and B tiles with TMA into `Stages` shared
# buffers; the other two each multiply 64 rows with WGMMA, keeping one group
# in flight while they wait for the next stage.
function hopper_gemm_kernel!(c,a,b,atom::WGMMAAtom{T,N,KS,1},k::Int32,::Val{Stages},::Val{Group}) where {T,N,KS,Stages,Group}
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
            initialize!(empty(s),256)           # every consumer thread
        end
        fence_barrier_init()
    end
    sync_threads()
    tm, tn = tile_order(Int32(blockIdx().x)-Int32(1),size(c,2) % Int32 ÷ Int32(128),Val(Group))
    row, col = tm*Int32(128), tn*Int32(N)
    count = k ÷ Int32(KS)
    @inline stage(i) = buffers+(i%Int32(Stages))*Int32(bytes)
    if warpgroup == Int32(0)
        release_registers(Val(40))
        if tid == Int32(0)
            for i in Int32(0):count-Int32(1)
                slot = i%Int32(Stages)
                i >= Int32(Stages) && wait(empty(slot),((i÷Int32(Stages))-Int32(1))&Int32(1))
                arrive!(full(slot),bytes)
                @inbounds tma_load!(shared_tile(a,stage(i)),a,(row,Int32(KS)*i),full(slot))
                @inbounds tma_load!(shared_tile(b,stage(i)+transfer_bytes(a)),b,(Int32(KS)*i,col),full(slot))
            end
        end
    else
        acquire_registers(Val(232))
        rows = (warpgroup-Int32(1))*Int32(64)
        @inline operands(i) = @inbounds (operand(atom,OperandA(),shared_tile(a,stage(i)),(rows,Int32(0))),
                                 operand(atom,OperandB(),shared_tile(b,stage(i)+transfer_bytes(a))))
        wait(full(Int32(0)),Int32(0))
        acc = mma_async(atom,operands(Int32(0))...,zero_accumulator(atom))
        for i in Int32(1):count-Int32(1)
            wait(full(i%Int32(Stages)),(i÷Int32(Stages))&Int32(1))
            acc = wait_mma(mma_async(atom,operands(i)...,acc),1)
            arrive!(empty((i-Int32(1))%Int32(Stages)))   # stage i-1's group is complete
        end
        result = finish_mma(wait_mma(acc))
        arrive!(empty((count-Int32(1))%Int32(Stages)))
        m, n = size(c,2) % Int32, size(c,1) % Int32
        out = @inbounds GlobalTile(pointer(c),Layout((m,n),(n,static(1))),4)
        @inbounds store!(window(out,(row+rows,col),(64,N)),Fragment{BFloat16}(result),tid%Int32(128))
    end
    nothing
end

# The tile of CTA `id`: groups of `Group` row tiles, each walked down its rows
# first, so that the CTAs running at once share A and B tiles through L2.
@inline function tile_order(id::Int32,tiles_m::Int32,::Val{Group}) where Group
    width = Int32(Group)*unchecked_div(gridDim().x,tiles_m) # CTAs per group of row tiles
    first = unchecked_div(id,width)*Int32(Group)
    height = min(tiles_m-first,Int32(Group))
    within = unchecked_rem(id,width)
    first+unchecked_rem(within,height), unchecked_div(within,height)
end

"""
    hopper_gemm(a, b; n=256, stages=4, group=16) -> (kernel, launch, c)

Prepare C = Aᵀ B for `a` (K×M) and `b` (K×N) on the device, BF16, FP16 or FP8: the tile
width `n`, the number of shared stages and the number of row tiles
grouped for L2 reuse are the configuration. Returns the
compiled kernel, a launcher, and C as an N×M matrix (row-major M×N).
"""
function hopper_gemm(a::CuMatrix{T},b::CuMatrix{T};n::Integer=256,stages::Integer=4,group::Integer=16,c=nothing) where T
    k, m = size(a); kb, nt = size(b)
    ks = 128 ÷ sizeof(T)                                  # one swizzle row of K per stage
    k == kb && k % ks == 0 && m % 128 == 0 && nt % n == 0 ||
        throw(ArgumentError("shapes must be multiples of the tile: K of $ks, M of 128, N of $n"))
    atom = WGMMAAtom(T,n,ks)
    ta = prepare_tma(TMATile(T,(128,ks),2),a)
    tb = prepare_tma(TMATile(T,(ks,n),1),b)
    shmem = 16stages+1024+stages*(transfer_bytes(ta)+transfer_bytes(tb))
    shmem <= 227*1024 || throw(ArgumentError("$stages stages need $(cld(shmem,1024)) KiB of shared memory"))
    c = c === nothing ? CuArray{BFloat16}(undef,nt,m) : c
    (m÷128) % group == 0 || throw(ArgumentError("row tiles must be a multiple of the group"))
    kernel = @cuda launch=false minthreads=384 hopper_gemm_kernel!(c,ta,tb,atom,Int32(k),Val(stages),Val(group))
    CUDACore.attributes(kernel.fun)[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem
    launch() = GC.@preserve ta tb kernel(c,ta,tb,atom,Int32(k),Val(stages),Val(group);threads=384,blocks=(m÷128)*(nt÷n),shmem)
    kernel, launch, c
end
