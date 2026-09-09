# A real allocation and round trip on datacenter Blackwell. Raw PTX readback
# checks Tylo's packed TMEM store independently of its typed load path.
function tmem_roundtrip_kernel!(out,input)
    tid = UInt32(threadIdx().x)-UInt32(1)
    warp = tid >> UInt32(5)
    slot = @inbounds CuStaticSharedArray(UInt32,1)
    if warp == UInt32(0)
        ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(
            PTX.smem_addr_u32(pointer(slot)),UInt32(128))
    end
    sync_threads()
    base = @inbounds slot[1]
    rows = warp_rows(TmemTile{Float32,128}(base),warp)
    PTX.Utils.@unroll for half in 0:1
        values = RowFragment(ntuple(i -> @inbounds(input[Int(tid)*128+64half+i]),Val(64)))
        store_async!(columns(rows,Val(64half),Val(64)),values)
    end
    wait_stores()
    PTX.Utils.@unroll for half in 0:1
        values = wait_load(load_async(columns(rows,Val(64half),Val(64))))
        packed = pack_bf16(scale(values,0.25f0))
        bf = columns(reinterpret_tile(BFloat16,rows),Val(64half),Val(64))
        store_async!(bf,packed)
        wait_stores()
        words = ptx"tcgen05.ld.sync.aligned.32x32b.x32.b32"(bf.address)
        ptx"tcgen05.wait::ld.sync.aligned"()
        store_row!(pointer(out)+Int(tid)*256+128half,PackedBF16(words))
    end
    sync_threads()
    if warp == UInt32(0)
        ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(base,UInt32(128))
        ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    end
    nothing
end

@testset "TMEM round-trip assembly" begin
    code = compile_kernel(tmem_roundtrip_kernel!,
        Tuple{CuDeviceVector{UInt16,1},CuDeviceVector{Float32,1}})
    save_code("tmem-roundtrip",code)
    @test !isempty(code.image)
    @test !occursin(".local .",entry_body(code.ptx))
    @test !occursin(r"\bcall",entry_body(code.ptx))
end

if CUDACore.functional() && CUDACore.capability(CUDACore.device()) in (v"10.0",v"10.3")
    @testset "TMEM round-trip execution" begin
        values = randn(MersenneTwister(93),Float32,128*128)
        input = CuArray(values)
        output = CuArray{UInt16}(undef,length(values))
        @cuda threads=128 tmem_roundtrip_kernel!(output,input)
        @test Array(output) == reinterpret.(UInt16,BFloat16.(values .* 0.25f0))
    end
else
    @testset "TMEM execution requires B200/B300" begin
        @test_skip false
    end
end
