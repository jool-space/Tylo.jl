# TEST_TARGET: cc==9.0
include("../../examples/hopper/kernel.jl")
function hopper_signature(T,n,partials,stages)
    ap = TMATile(T,(64,64),2)
    bp = TMATile(T,(64,n),1)
    p = WGMMAAtom(T,n,64,partials)
    Tuple{CuDeviceMatrix{Float32,1},Tylo.TMABinding{typeof(ap),PTX.TMADescriptorPtr,2},
        Tylo.TMABinding{typeof(bp),PTX.TMADescriptorPtr,2},typeof(p),Int32,Val{stages}}
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
        arrives, fences = wgmma_arrives(code)
        @test arrives == fences
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
        a = prepare_tma(TMATile(T,(64,64),2),CuArray(ha))
        b = prepare_tma(TMATile(T,(64,n),1),CuArray(hb))
        plan = WGMMAAtom(T,n,64,partials)
        out = CuArray{Float32}(undef,64,n)
        GC.@preserve a b begin
            kernel = @cuda launch=false arch=CUDACore.SMVersion(9,0,:arch) minthreads=160 hopper_gemm_kernel!(out,a,b,plan,Int32(k),Val(stages))
            shmem = 1056+stages*(transfer_bytes(a)+transfer_bytes(b))
            # Beyond 48 KiB a kernel must opt in (n=128 with two stages).
            CUDACore.attributes(kernel.fun)[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem
            launch() = kernel(out,a,b,plan,Int32(k),Val(stages);threads=160,shmem)
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
        p = WGMMAAtom(T,n,k,partials)
        code = compile_kernel(wgmma_operand_probe!,Tuple{CuDeviceMatrix{Float32,1},typeof(p),Int32};
            arch=CUDACore.SMVersion(9,0,:arch))
        save_code("hopper-origin-$(T)-n$n-k$k-p$partials",code)
        @test !isempty(code.image)
        arrives, fences = wgmma_arrives(code)
        @test arrives == fences
        @test !occursin(".local .",entry_body(code.ptx))
    end
end
end
@testset "Pipelined WGMMA issue assembles" begin
    for T in (BFloat16,Float16), n in (8,64)
        p = WGMMAAtom(T,n,16)
        code = compile_kernel(wgmma_pipelined_probe!,Tuple{CuDeviceMatrix{Float32,1},typeof(p)};arch=CUDACore.SMVersion(9,0,:arch))
        save_code("hopper-pipelined-$(T)-n$n",code)
        body = entry_body(code.ptx)
        @test count("wgmma.mma_async",body) == 4 && count("wgmma.commit_group",body) == 4
        @test count("wgmma.fence",body) == 4
        arrives, fences = wgmma_arrives(code)
        @test arrives == fences
        @test count("wgmma.wait_group.sync.aligned 1",body) == 3
        @test count("wgmma.wait_group.sync.aligned 0",body) == 1
        commits = [first(r) for r in findall("wgmma.commit_group",body)]
        waits = [first(r) for r in findall("wgmma.wait_group",body)]
        @test issorted(vcat(commits[1:2],waits[1],commits[3],waits[2],commits[4],waits[3:4]))
        @test !occursin(".local .",body) && !occursin(r"\bcall",body)
    end
end
if capability_is(v"9.0")
@testset "Pipelined WGMMA equals one K=64 issue" begin
    for T in (BFloat16,Float16), n in (8,64)
        chained, whole = CuArray{Float32}(undef,64,n), CuArray{Float32}(undef,64,n)
        @cuda threads=128 shmem=33800 arch=CUDACore.SMVersion(9,0,:arch) wgmma_pipelined_probe!(chained,WGMMAAtom(T,n,16))
        @cuda threads=128 shmem=33792 arch=CUDACore.SMVersion(9,0,:arch) wgmma_operand_probe!(whole,WGMMAAtom(T,n,64),Int32(0))
        @test Array(chained) == Array(whole)
    end
end
@testset "WGMMA shared stores and descriptor origins" begin
    for T in (BFloat16,Float16), k in (16,32,64), n in (8,24), partials in unique((1,k÷16)), k0 in unique((0,64-k))
        p = WGMMAAtom(T,n,k,partials)
        out = CuArray{Float32}(undef,64,n)
        @cuda threads=128 shmem=33792 arch=CUDACore.SMVersion(9,0,:arch) wgmma_operand_probe!(out,p,Int32(k0))
        a = Float64[(r+3kk)%19-9 for r in 64:127, kk in k0:k0+k-1]
        b = Float64[(2c+kk)%17-8 for kk in k0:k0+k-1,c in 8:8+n-1]
        @test Array(out) == a*b # small exact integer products in FP32
    end
end
end
