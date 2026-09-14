# TEST_TARGET: cc>=8.0
using Tylo.Layouts: @Layout
# This path runs on GB10 too: no TMEM instructions, but the SAME fragment
# scale/slice/BF16/vector-store implementation used by the attention epilogue.
function fragment_kernel!(out, input, alpha)
    tid = Int32(threadIdx().x)-Int32(1)
    values = local_fragment(ntuple(i -> @inbounds(input[64tid+i]),Val(64)))
    for_half = window(values,Val((0,0)),Val((32,32)))
    for_tail = window(values,Val((0,32)),Val((32,32)))
    store!(pointer(out)+128tid,pack(BFloat16,for_half .* alpha))
    store!(pointer(out)+128tid+64,pack(BFloat16,for_tail .* alpha))
    nothing
end

function tmem_correction_kernel!(base::UInt32, alpha::Float32)
    warp = (UInt32(threadIdx().x)-UInt32(1)) >> UInt32(5)
    output = @inbounds window(TmemTile(Float32,base,@Layout((128,128),(1,128))),(UInt32(32)*warp,UInt32(0)),Val((32,128)))
    fence_after_thread_sync()
    PTX.Utils.@unroll for half in 0:1
        chunk = @inbounds partition(TmemTransfer{(32,64),2}(),window(output,(0,64half),Val((32,64))))
        values = wait_load(load_async(chunk))
        store_async!(chunk,values .* alpha)
    end
    wait_stores()
    fence_before_thread_sync()
    nothing
end

function tmem_epilogue_kernel!(out, base::UInt32, alpha::Float32)
    tid = UInt32(threadIdx().x)-UInt32(1)
    output = @inbounds window(TmemTile(Float32,base,@Layout((128,64),(1,128))),(UInt32(32)*(tid >> UInt32(5)),UInt32(0)),Val((32,64)))
    values = @inbounds wait_load(load_async(partition(TmemTransfer{(32,64),2}(),output)))
    PTX.Utils.@unroll for half in 0:1
        part = window(values,Val((0,32half)),Val((32,32)))
        store!(pointer(out)+Int(tid)*128+64half,pack(BFloat16,part .* alpha))
    end
    nothing
end

@testset "Fragment and TMEM code generation" begin
    for (name,f,tt) in (
            ("correction",tmem_correction_kernel!,Tuple{UInt32,Float32}),
            ("epilogue",tmem_epilogue_kernel!,Tuple{CuDeviceVector{UInt16,1},UInt32,Float32}))
        code = compile_kernel(f,tt)
        save_code(name,code)
        @test !isempty(code.image)
        @test !occursin(".local .",entry_body(code.ptx))
        @test !occursin(r"\bcall",entry_body(code.ptx))
        @test occursin("tcgen05.wait::ld.sync.aligned",code.ptx)
        @test occursin("tcgen05.ld.sync.aligned.32x32b.x64",code.ptx)
        body = entry_body(code.ptx)
        @test first(findfirst("tcgen05.ld.sync",body)) <
              first(findfirst("tcgen05.wait::ld",body)) <
              first(findfirst("mul.f32",body))
    end
    code = compile_kernel(fragment_kernel!,
        Tuple{CuDeviceVector{UInt16,1},CuDeviceVector{Float32,1},Float32};
        arch=CUDACore.SMVersion(12,1,:arch))
    save_code("fragments",code)
    @test !occursin(".local .",entry_body(code.ptx))
    @test !occursin(r"\bcall",entry_body(code.ptx))
    @test occursin("cvt.rn.bf16x2.f32",code.ptx)
    @test occursin("st.global.v4.b32",code.ptx)
end

if runtime_supported(@__FILE__)
    @testset "GPU fragment arithmetic and packed stores" begin
        rng = MersenneTwister(47)
        values = randn(rng,Float32,32*64)
        # Include halfway cases, signed zero, infinities, and subnormal inputs.
        special = Float32[0f0,-0f0,Inf32,-Inf32,1.00390625f0,1.01171875f0,
                          -1.00390625f0, reinterpret(Float32,UInt32(0x00018000))]
        values[1:length(special)] .= special
        input = CuArray(values)
        output = CuArray{UInt16}(undef,length(values))
        for alpha in (1f0,0.25f0,-2f0)
            @cuda threads=32 fragment_kernel!(output,input,alpha)
            actual = Array(output)
            expected = reinterpret.(UInt16,BFloat16.(values .* alpha))
            @test actual == expected
        end
    end
else
    @test_skip false # GPU register execution unavailable
end
