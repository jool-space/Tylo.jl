# TEST_TARGET: cc>=9.0
function tma_roundtrip!(out,binding,origin,iterations::Int32)
    bytes = transfer_bytes(binding)
    align = 8*swizzle_bytes(binding)
    smem = @inbounds CuDynamicSharedArray(UInt8,bytes+align+32)
    raw = pointer(smem)
    bar = Barrier(raw)
    start = raw+32
    start += (UInt32(0)-PTX.smem_addr_u32(start)) & UInt32(align-1)
    tile = shared_tile(binding,start)
    tid = Int32(threadIdx().x)-Int32(1)
    if tid == 0
        initialize!(bar,1)
        fence_barrier_init()
    end
    sync_threads()
    for phase in Int32(0):iterations-Int32(1)
        if tid == 0
            arrive!(bar,bytes)
            @inbounds tma_load!(tile,binding,origin,bar)
        end
        wait(bar,phase)
        m,n = size(tile)
        for i in tid:Int32(32):Int32(m*n-1)
            r,c = i%Int32(m),i÷Int32(m)
            @inbounds out[r+Int32(1),c+Int32(1)] = tile[(r,c)]
        end
        sync_threads() # every consumer finished before phase reuse
    end
    tid == 0 && invalidate!(bar)
    nothing
end
# Generic-proxy writes through the layout, published to the async proxy,
# then one bulk group of one store, waited for by the issuing thread.
function tma_store_kernel!(src,binding,origin)
    bytes = transfer_bytes(binding)
    align = 8*swizzle_bytes(binding)
    smem = @inbounds CuDynamicSharedArray(UInt8,bytes+align)
    start = pointer(smem)
    start += (UInt32(0)-PTX.smem_addr_u32(start)) & UInt32(align-1)
    tile = shared_tile(binding,start)
    tid = Int32(threadIdx().x)-Int32(1)
    m,n = size(tile)
    for i in tid:Int32(32):Int32(m*n-1)
        r,c = i%Int32(m),i÷Int32(m)
        @inbounds tile[(r,c)] = src[r+Int32(1),c+Int32(1)]
    end
    fence_async_shared()
    sync_threads()
    if tid == 0
        @inbounds tma_store!(binding,tile,origin)
        commit_tma_stores()
        wait_tma_reads(0)
        wait_tma_stores(0)
    end
    nothing
end

if runtime_supported(@__FILE__)
@testset "TMA loads: element widths, swizzle rows, logical axes, OOB and phase reuse" begin
    for T in (UInt8,BFloat16,Float16,Float32), W in (32,64,128), axis in (1,2), rows in (8,64,128), stripes in (1,2)
        (stripes == 2 && rows == 8) && continue
        row = stripes * W ÷ sizeof(T)      # the inner extent: one or two stripes
        shape = axis == 2 ? (rows,row) : (row,rows)
        plan = TMATile(T,shape,axis,W)
        host = T.(reshape(mod.(0:128*77-1,113),128,77))
        binding = prepare_tma(plan,CuArray(host))
        @test isbitstype(typeof(CUDACore.cudaconvert(binding)))
        @test UInt(binding.descriptor)%64 == 0
        out = CuArray{T}(undef,shape)
        for (k,r) in ((0,0),(64,64),(-16,-8))
            origin = axis == 2 ? (Int32(r),Int32(k)) : (Int32(k),Int32(r))
            GC.gc(true)
            GC.@preserve binding begin
                @cuda threads=32 shmem=transfer_bytes(plan)+8W+32 tma_roundtrip!(out,binding,origin,Int32(3))
                synchronize()
            end
            expected = zeros(T,row,rows)
            for rr in 1:rows, kk in 1:row
                1 <= r+rr <= 77 && 1 <= k+kk <= 128 && (expected[kk,rr] = host[k+kk,r+rr])
            end
            @test Array(out) == (axis == 2 ? permutedims(expected) : expected)
        end
    end
    # Descriptor and tensor are hidden behind the isbits handle. Adaptation
    # must still register both allocations with CUDA's stream tracking.
    producer,consumer = CuStream(),CuStream()
    binding = CUDACore.stream!(producer) do
        prepare_tma(TMATile(BFloat16,(64,64),2),CuArray(ones(BFloat16,64,64)))
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
    @test_throws ArgumentError prepare_tma(TMATile(BFloat16,(64,64),2),CuArray(zeros(BFloat16,65,64)))
    # A batch of matrices: the third origin coordinate selects one, each
    # bounded separately; `bounds` shrinks a padded array's logical extents.
    plan=TMATile(BFloat16,(64,64),1)
    host=BFloat16.(reshape(mod.(0:80*100*3-1,113),80,100,3))
    binding=prepare_tma(plan,CuArray(host);bounds=(70,100,3))
    out=CuArray{BFloat16}(undef,64,64)
    for (h,k,r) in ((0,0,0),(1,16,64),(2,32,48))
        GC.@preserve binding begin
            @cuda threads=32 shmem=transfer_bytes(plan)+1056 tma_roundtrip!(out,binding,(Int32(k),Int32(r),Int32(h)),Int32(2))
            synchronize()
        end
        expected=zeros(BFloat16,64,64)
        for rr in 1:64, kk in 1:64
            r+rr <= 100 && k+kk <= 70 && (expected[kk,rr]=host[k+kk,r+rr,h+1])
        end
        @test Array(out) == expected
    end
    @test_throws ArgumentError prepare_tma(plan,CuArray(host);bounds=(81,100,3))
    # Strided sources: a (d, heads, seq, batch) array permuted to (d, seq,
    # heads, batch), and a view of it, bind in place; the origin's third
    # and fourth coordinates select head and batch.
    host=BFloat16.(reshape(mod.(0:64*3*96*2-1,113),64,3,96,2))
    source=CuArray(host)
    out=CuArray{BFloat16}(undef,64,64)
    for (array,cropped) in ((PermutedDimsArray(source,(1,3,2,4)),host),(PermutedDimsArray(view(source,:,:,17:96,:),(1,3,2,4)),host[:,:,17:96,:]))
        binding=prepare_tma(plan,array)
        @test typeof(CUDACore.cudaconvert(binding)) <: Tylo.TMABinding{<:Any,<:Any,4}
        for (h,b,k,r) in ((0,0,0,0),(2,1,0,32),(1,1,16,48))
            GC.@preserve binding begin
                @cuda threads=32 shmem=transfer_bytes(plan)+1056 tma_roundtrip!(out,binding,(Int32(k),Int32(r),Int32(h),Int32(b)),Int32(2))
                synchronize()
            end
            expected=zeros(BFloat16,64,64)
            for rr in 1:64, kk in 1:64
                r+rr <= size(cropped,3) && k+kk <= 64 && (expected[kk,rr]=cropped[k+kk,h+1,r+rr,b+1])
            end
            @test Array(out) == expected
        end
    end
    @test_throws ArgumentError prepare_tma(plan,PermutedDimsArray(source,(2,1,3,4)))   # the inner axis is strided
    @test_throws ArgumentError prepare_tma(plan,view(source,5:64,:,:,:))               # 8 bytes past alignment
    @test_throws ArgumentError prepare_tma(plan,reshape(source,64,3,96,1,1,2))         # six axes
end

@testset "TMA stores: element widths, swizzle rows, logical axes and clipping" begin
    for T in (UInt8,BFloat16,Float32), W in (32,64,128), axis in (1,2), rows in (8,64,128)
        row = W ÷ sizeof(T)
        shape = axis == 2 ? (rows,row) : (row,rows)
        plan = TMATile(T,shape,axis,W)
        src = T.(reshape(mod.(0:prod(shape)-1,97),shape))
        for (k,r) in ((0,0),(64,64),(112,72))
            origin = axis == 2 ? (Int32(r),Int32(k)) : (Int32(k),Int32(r))
            dst = CuArray(fill(T(111),128,77))
            binding = prepare_tma(plan,dst)
            GC.@preserve binding begin
                @cuda threads=32 shmem=transfer_bytes(plan)+8W tma_store_kernel!(CuArray(src),binding,origin)
                synchronize()
            end
            expected = fill(T(111),128,77)
            for rr in 1:rows, kk in 1:row
                1 <= r+rr <= 77 && 1 <= k+kk <= 128 && (expected[k+kk,r+rr] = axis == 2 ? src[rr,kk] : src[kk,rr])
            end
            @test Array(dst) == expected
        end
    end
    # The store path: one tensor store, one commit, both waits, no calls.
    plan = TMATile(BFloat16,(64,32),2,64)
    binding = prepare_tma(plan,CuArray(zeros(BFloat16,64,64)))
    src = CuArray(zeros(BFloat16,64,32))
    code = compile_kernel(tma_store_kernel!,Tuple{typeof(CUDACore.cudaconvert(src)),typeof(CUDACore.cudaconvert(binding)),Tuple{Int32,Int32}};
                          arch=CUDACore.SMVersion(9,0,:arch),threads=32)
    body = entry_body(code.ptx)
    @test count("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group",body) == 1
    @test count("cp.async.bulk.commit_group",body) == 1
    @test count("cp.async.bulk.wait_group.read 0",body) == 1
    @test count(r"cp\.async\.bulk\.wait_group 0",body) == 1
    @test count("fence.proxy.async.shared::cta",body) == 1
    @test isempty(audit(code))
    # Stores into one matrix of a batch skip the padding beyond its bounds.
    plan=TMATile(Float32,(32,64),1)
    dst=CuArray(fill(111f0,40,64,2))
    binding=prepare_tma(plan,dst;bounds=(36,64,2))
    src=CuArray(Float32.(reshape(1:32*64,32,64)))
    GC.@preserve binding begin
        @cuda threads=32 shmem=transfer_bytes(plan)+8*128 tma_store_kernel!(src,binding,(Int32(16),Int32(8),Int32(1)))
        synchronize()
    end
    expected=fill(111f0,40,64,2)
    for c in 1:64, r in 1:32
        16+r <= 36 && 8+c <= 64 && (expected[16+r,8+c,2]=Float32(r+32(c-1)))
    end
    @test Array(dst) == expected
    # Five axes, permuted: the store lands in one matrix of the parent.
    parent_dst=CuArray(fill(111f0,32,2,64,3,2))
    binding=prepare_tma(plan,PermutedDimsArray(parent_dst,(1,3,2,4,5)))
    GC.@preserve binding begin
        @cuda threads=32 shmem=transfer_bytes(plan)+8*128 tma_store_kernel!(src,binding,(Int32(0),Int32(0),Int32(1),Int32(2),Int32(1)))
        synchronize()
    end
    expected=fill(111f0,32,2,64,3,2)
    expected[:,2,:,3,2]=Array(src)
    @test Array(parent_dst) == expected
end
else
    @test_skip false # TMA runtime requires CC >= 9.0
end
