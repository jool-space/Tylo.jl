using Tylo.Layouts: @Layout
# A real allocation and round trip on datacenter Blackwell. Raw PTX readback
# checks Tylo's packed TMEM store independently of its typed load path.
function tmem_roundtrip_kernel!(out,input,::Val{P}) where P
    tid = UInt32(threadIdx().x)-UInt32(1)
    warp = tid >> UInt32(5)
    slot = @inbounds CuStaticSharedArray(UInt32,1)
    if warp == UInt32(0)
        ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(
            PTX.smem_addr_u32(pointer(slot)),UInt32(128))
    end
    sync_threads()
    base = @inbounds slot[1]
    band = @inbounds window(TmemTile(Float32,base,@Layout((128,128),(1,128))),(UInt32(32)*warp,UInt32(0)),Val((32,128)))
    band = P ? permutedims(band) : band
    plan = P ? TmemTransfer{(64,32),1}() : TmemTransfer{(32,64),2}()
    shape = P ? (64,32) : (32,64)
    axis = P ? 1 : 2
    PTX.Utils.@unroll for half in 0:1
        values = Fragment(ntuple(i -> @inbounds(input[Int(tid)*128+64half+i]),Val(64)),plan)
        @inbounds store_async!(partition(plan,window(band,P ? (64half,0) : (0,64half),Val(shape))),values)
    end
    wait_stores()
    PTX.Utils.@unroll for half in 0:1
        values = @inbounds wait_load(load_async(partition(plan,window(band,P ? (64half,0) : (0,64half),Val(shape)))))
        packed = pack_bf16(values .* 0.25f0)
        bf = @inbounds partition(plan,window(reinterpret_tile(BFloat16,band;dims=axis),P ? (64half,0) : (0,64half),Val(shape)))
        store_async!(bf,packed)
        wait_stores()
        words = ptx"tcgen05.ld.sync.aligned.32x32b.x32.b32"(bf.address)
        ptx"tcgen05.wait::ld.sync.aligned"()
        store!(pointer(out)+Int(tid)*256+128half,PackedBF16(words))
    end
    sync_threads()
    if warp == UInt32(0)
        ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(base,UInt32(128))
        ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    end
    nothing
end

@testset "TMEM round-trip assembly" begin
    for p in (false,true)
        code = compile_kernel(tmem_roundtrip_kernel!,
            Tuple{CuDeviceVector{UInt16,1},CuDeviceVector{Float32,1},Val{p}})
        save_code(p ? "tmem-roundtrip-permuted" : "tmem-roundtrip",code)
        @test !isempty(code.image)
        @test !occursin(".local .",entry_body(code.ptx))
        @test !occursin(r"\bcall",entry_body(code.ptx))
    end
end

if CUDACore.functional() && CUDACore.capability(CUDACore.device()) in (v"10.0",v"10.3")
    @testset "TMEM round-trip execution" begin
        values = randn(MersenneTwister(93),Float32,128*128)
        input = CuArray(values)
        output = CuArray{UInt16}(undef,length(values))
        for p in (false,true)
            @cuda threads=128 tmem_roundtrip_kernel!(output,input,Val(p))
            @test Array(output) == reinterpret.(UInt16,BFloat16.(values .* 0.25f0))
        end
    end
else
    @testset "TMEM execution requires B200/B300" begin
        @test_skip false
    end
end
