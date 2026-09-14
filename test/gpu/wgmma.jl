# TEST_TARGET: cc==9.0
include("../../examples/hopper/kernel.jl")
function hopper_signature(T,n,partials,stages)
    ap = TMALoad(T,Val((64,64)),Val(2))
    bp = TMALoad(T,Val((64,n)),Val(1))
    p = WGMMA64(T,Val(n),Val(64),Val(partials))
    Tuple{CuDeviceMatrix{Float32,1},Tylo.DeviceTMA{typeof(ap),PTX.TMADescriptorPtr},
        Tylo.DeviceTMA{typeof(bp),PTX.TMADescriptorPtr},typeof(p),Int32,Val{stages}}
end
begin # assembly checks
@testset "TMA to Hopper WGMMA complete pipeline assembly" begin
    for T in (BFloat16,Float16), (n,partials) in ((8,1),(8,4),(16,1),(24,1),(64,1),(128,1),(256,1)), stages in (1,2)
        code = compile_kernel(hopper_gemm_kernel!,hopper_signature(T,n,partials,stages);
            arch=CUDACore.SMVersion(9,0,:arch),threads=160)
        save_code("hopper-$(T)-n$n-p$partials-s$stages",code)
        body = entry_body(code.ptx)
        @test !isempty(code.image)
        @test !occursin(".local .",body)
        @test count(r"\bcall",body) == 0
        for instruction in ("cp.async.bulk.tensor.2d", "mbarrier.arrive.expect_tx", "wgmma.fence.sync.aligned",
                            "wgmma.mma_async.sync.aligned.m64n$(n)k16", "wgmma.commit_group.sync.aligned",
                            "wgmma.wait_group.sync.aligned 0")
            @test occursin(instruction,body)
        end
    end
end
end

if capability_is(v"9.0")
@testset "Hopper WGMMA runtime and producer/consumer reuse" begin
    rng = MersenneTwister(1049)
    for T in (BFloat16,Float16), (n,partials) in ((8,1),(8,4),(16,1),(24,1),(64,1),(128,1),(256,1)), stages in (1,2), k in (64,128,320,328)
        ha,hb = T.(0.1f0 .* randn(rng,Float32,k,64)),T.(0.1f0 .* randn(rng,Float32,k,n))
        a = prepare_tma(TMALoad(T,Val((64,64)),Val(2)),CuArray(ha))
        b = prepare_tma(TMALoad(T,Val((64,n)),Val(1)),CuArray(hb))
        plan = WGMMA64(T,Val(n),Val(64),Val(partials))
        out = CuArray{Float32}(undef,64,n)
        GC.@preserve a b begin
            kernel = @cuda launch=false arch=CUDACore.SMVersion(9,0,:arch) minthreads=160 hopper_gemm_kernel!(out,a,b,plan,Int32(k),Val(stages))
            launch() = kernel(out,a,b,plan,Int32(k),Val(stages);threads=160,shmem=1056+stages*(transfer_bytes(a)+transfer_bytes(b)))
            launch(); synchronize()
            expected = transpose(Float64.(ha))*Float64.(hb)
            @test all(abs.(Array(out) .- expected) .<= 2e-4 .+ 2e-4 .* abs.(expected))
            graph = CUDACore.instantiate(CUDACore.capture(launch))
            for repeat in 1:2
                ha .*= T(0.5); copyto!(a.source,ha); GC.gc(true)
                CUDACore.launch(graph); synchronize()
                expected = transpose(Float64.(ha))*Float64.(hb)
                @test all(abs.(Array(out) .- expected) .<= 2e-4 .+ 2e-4 .* abs.(expected))
            end
        end
    end
end
else
    @testset "WGMMA execution requires H100/H200" begin
        @test_skip false
    end
end

include("wgmma_defs.jl")
begin # assembly checks
@testset "WGMMA K extents and nonzero descriptor origins assemble" begin
    for T in (BFloat16,Float16), k in (16,32,64), n in (8,24), partials in unique((1,k÷16))
        p = WGMMA64(T,Val(n),Val(k),Val(partials))
        code = compile_kernel(wgmma_operand_probe!,Tuple{CuDeviceMatrix{Float32,1},typeof(p),Int32};
            arch=CUDACore.SMVersion(9,0,:arch))
        save_code("hopper-origin-$(T)-n$n-k$k-p$partials",code)
        @test !isempty(code.image)
        @test !occursin(".local .",entry_body(code.ptx))
    end
end
end
if capability_is(v"9.0")
@testset "WGMMA shared stores and descriptor origins" begin
    for T in (BFloat16,Float16), k in (16,32,64), n in (8,24), partials in unique((1,k÷16)), k0 in unique((0,64-k))
        p = WGMMA64(T,Val(n),Val(k),Val(partials))
        out = CuArray{Float32}(undef,64,n)
        @cuda threads=128 shmem=33792 arch=CUDACore.SMVersion(9,0,:arch) wgmma_operand_probe!(out,p,Int32(k0))
        a = Float64[(r+3kk)%19-9 for r in 64:127, kk in k0:k0+k-1]
        b = Float64[(2c+kk)%17-8 for kk in k0:k0+k-1,c in 8:8+n-1]
        @test Array(out) == a*b # small exact integer products in FP32
    end
end
end
