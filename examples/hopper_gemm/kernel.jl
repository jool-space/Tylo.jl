using Tylo, PTX, CUDACore, BFloat16s
using Tylo.Layouts: Layout, static

# A warp-specialized Hopper GEMM, C = Aᵀ B: A is K×M and B is K×N, both
# contiguous along K; C is M×N, stored row-major (an N×M Julia matrix), in
# BF16 with FP32 accumulation. Each CTA computes one 128×N tile with three
# warpgroups: the first loads A and B tiles with TMA into `Stages` shared
# buffers; the other two each multiply 64 rows with WGMMA, keeping one group
# in flight while they wait for the next stage.
function hopper_gemm_kernel!(c,a,b,atom::WGMMAAtom{T,N,64,1},k::Int32,::Val{Stages}) where {T,N,Stages}
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
    row = (Int32(blockIdx().x)-Int32(1))*Int32(128)
    col = (Int32(blockIdx().y)-Int32(1))*Int32(N)
    count = k ÷ Int32(64)
    @inline stage(i) = buffers+(i%Int32(Stages))*Int32(bytes)
    if warpgroup == Int32(0)
        release_registers(Val(40))
        if tid == Int32(0)
            for i in Int32(0):count-Int32(1)
                slot = i%Int32(Stages)
                i >= Int32(Stages) && wait(empty(slot),((i÷Int32(Stages))-Int32(1))&Int32(1))
                arrive!(full(slot),bytes)
                @inbounds tma_load!(shared_tile(a,stage(i)),a,(row,Int32(64)*i),full(slot))
                @inbounds tma_load!(shared_tile(b,stage(i)+transfer_bytes(a)),b,(Int32(64)*i,col),full(slot))
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

"""
    hopper_gemm(a, b; n=256, stages=4) -> (kernel, launch, c)

Prepare C = Aᵀ B for BF16 `a` (K×M) and `b` (K×N) on the device: the tile
width `n` and the number of shared stages are the configuration. Returns the
compiled kernel, a launcher, and C as an N×M matrix (row-major M×N).
"""
function hopper_gemm(a::CuMatrix{T},b::CuMatrix{T};n::Integer=256,stages::Integer=4,c=nothing) where T
    k, m = size(a); kb, nt = size(b)
    k == kb && k % 64 == 0 && m % 128 == 0 && nt % n == 0 ||
        throw(ArgumentError("shapes must be multiples of the tile: K of 64, M of 128, N of $n"))
    atom = WGMMAAtom(T,n,64)
    ta = prepare_tma(TMATile(T,(128,64),2),a)
    tb = prepare_tma(TMATile(T,(64,n),1),b)
    shmem = 16stages+1024+stages*(transfer_bytes(ta)+transfer_bytes(tb))
    shmem <= 227*1024 || throw(ArgumentError("$stages stages need $(cld(shmem,1024)) KiB of shared memory"))
    c = c === nothing ? CuArray{T}(undef,nt,m) : c
    kernel = @cuda launch=false minthreads=384 hopper_gemm_kernel!(c,ta,tb,atom,Int32(k),Val(stages))
    CUDACore.attributes(kernel.fun)[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem
    launch() = GC.@preserve ta tb kernel(c,ta,tb,atom,Int32(k),Val(stages);threads=384,blocks=(m÷128,nt÷n),shmem)
    kernel, launch, c
end
