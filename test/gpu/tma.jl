function tma_roundtrip!(out,binding,origin,iterations::Int32)
    bytes = transfer_bytes(binding)
    smem = @inbounds CuDynamicSharedArray(UInt8,bytes+1056)
    raw = pointer(smem)
    bar = reinterpret(Core.LLVMPtr{UInt64,3},raw)
    start = raw+32
    start += (UInt32(0)-PTX.smem_addr_u32(start)) & UInt32(1023)
    tile = shared_tile(binding,start)
    tid = Int32(threadIdx().x)-Int32(1)
    if tid == 0
        ptx"mbarrier.init.shared.b64"(bar,UInt32(1))
        ptx"fence.proxy.async.shared::cta"()
    end
    sync_threads()
    for phase in Int32(0):iterations-Int32(1)
        if tid == 0
            ptx"mbarrier.arrive.expect_tx.shared.b64"(bar,UInt32(bytes))
            @inbounds tma_load!(tile,binding,origin,bar)
        end
        while !ptx"mbarrier.try_wait.parity.shared.b64"(bar,UInt32(phase&Int32(1))) end
        m,n = size(tile)
        for i in tid:Int32(32):Int32(m*n-1)
            r,c = i%Int32(m),i÷Int32(m)
            @inbounds out[r+Int32(1),c+Int32(1)] = unsafe_load(pointer(tile,(r,c)))
        end
        sync_threads() # every consumer finished before phase reuse
    end
    tid == 0 && ptx"mbarrier.inval.shared.b64"(bar)
    nothing
end

if CUDACore.functional() && capability(device()) >= v"9.0"
@testset "TMA roundtrip, logical axes, OOB and phase reuse" begin
    for T in (BFloat16,Float16), axis in (1,2), rows in (8,16,64,128)
        shape = axis == 2 ? (rows,64) : (64,rows)
        plan = TMALoad(T,Val(shape),Val(axis))
        host = T.(reshape(mod.(0:128*77-1,113),128,77))
        binding = prepare_tma(plan,CuArray(host))
        @test isbitstype(typeof(CUDACore.cudaconvert(binding)))
        @test UInt(binding.descriptor)%64 == 0
        out = CuArray{T}(undef,shape)
        for (k,r) in ((0,0),(64,64),(-8,-8))
            origin = axis == 2 ? (Int32(r),Int32(k)) : (Int32(k),Int32(r))
            GC.gc(true)
            GC.@preserve binding begin
                @cuda threads=32 shmem=transfer_bytes(plan)+1056 tma_roundtrip!(out,binding,origin,Int32(5))
                synchronize()
            end
            expected = zeros(T,64,rows)
            for rr in 1:rows, kk in 1:64
                1 <= r+rr <= 77 && 1 <= k+kk <= 128 && (expected[kk,rr] = host[k+kk,r+rr])
            end
            @test Array(out) == (axis == 2 ? permutedims(expected) : expected)
        end
    end
    # Descriptor and tensor are hidden behind the isbits handle. Adaptation
    # must still register both allocations with CUDA's stream tracking.
    producer,consumer = CuStream(),CuStream()
    binding = CUDACore.stream!(producer) do
        prepare_tma(TMALoad(BFloat16,Val((64,64)),Val(2)),CuArray(ones(BFloat16,64,64)))
    end
    out = CuArray{BFloat16}(undef,64,64)
    for value in (2,3,5)
        CUDACore.stream!(producer) do
            fill!(binding.source,BFloat16(value))
        end
        GC.@preserve binding CUDACore.stream!(consumer) do
            @cuda threads=32 shmem=transfer_bytes(binding)+1056 tma_roundtrip!(out,binding,(Int32(0),Int32(0)),Int32(3))
            synchronize()
        end
        @test all(==(BFloat16(value)),Array(out))
    end
    @test_throws ArgumentError prepare_tma(TMALoad(BFloat16,Val((64,64)),Val(2)),CuArray(zeros(BFloat16,65,64)))
end
else
    @test_skip false # TMA runtime requires CC >= 9.0
end
