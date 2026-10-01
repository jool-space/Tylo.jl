# TEST_TARGET: cc==9.0
# The warp-specialized Hopper GEMM example: register reallocation, a TMA
# pipeline and pipelined WGMMA, compiled clean for sm_90a and correct on Hopper.
include("../../examples/hopper_gemm/kernel.jl")

@testset "Warp-specialized Hopper GEMM assembly" begin
    T = BFloat16
    for (n,stages) in ((128,6),(256,4))
        l = hopper_gemm_launch(CUDACore.zeros(T,256,256),CUDACore.zeros(T,256,n);n,stages,sms=132)
        code = machine_code(l;arch=CUDACore.SMVersion(9,0,:arch))
        save_code("hopper-gemm-n$n-s$stages",code)
        @test isempty(audit(code))
        body = entry_body(code.ptx)
        @test occursin(".reqntid 384",code.ptx)
        @test count("setmaxnreg.dec.sync.aligned.u32",body) == 1 && count("setmaxnreg.inc.sync.aligned.u32",body) == 1
        @test count("USETMAXREG",Tylo.kernel_sass(code)) == 2
        @test count("wgmma.wait_group.sync.aligned 1",body) == 1      # one group in flight
        @test count("cp.async.bulk.tensor.2d",body) == 2
    end
    # Clusters of two: B shared by multicast, stages released in both CTAs.
    let n = 256, stages = 4
        l = hopper_gemm_launch(CUDACore.zeros(T,256,256),CUDACore.zeros(T,256,n);n,stages,cluster=2,sms=132)
        code = machine_code(l;arch=CUDACore.SMVersion(9,0,:arch))
        save_code("hopper-gemm-n$n-s$stages-cluster2",code)
        @test isempty(audit(code))
        body = entry_body(code.ptx)
        @test count("multicast::cluster",body) == 1
        @test occursin("mapa.shared::cluster",body) && occursin("mbarrier.arrive.shared::cluster",body)
        @test count("barrier.cluster.wait",body) == 2
    end
end

if capability_is(v"9.0")
@testset "Warp-specialized Hopper GEMM on Hopper" begin
    rng = MersenneTwister(5)
    # 2048×2304 has 144 tiles of 128×256: more than an H100's SMs, so
    # persistent CTAs walk two tiles, carrying the pipeline across them.
    for T in (BFloat16,Float8_E4M3FN), (m,n,k) in ((256,256,256),(384,512,640),(2048,2304,256)),
            (w,stages) in ((128,3),(256,4)), persistent in (false,true), cluster in (1,2)
        n % w == 0 && m % (128cluster) == 0 || continue
        ha = T.(randn(rng,Float32,k,m) ./ sqrt(Float32(k)))
        hb = T.(randn(rng,Float32,k,n))
        k % (128 ÷ sizeof(T)) == 0 || continue
        _, launch, c = hopper_gemm(CuArray(ha),CuArray(hb);n=w,stages,group=1,persistent,cluster)
        launch(); synchronize()
        expected = transpose(Float64.(Float32.(ha)))*Float64.(Float32.(hb))   # M×N
        actual = permutedims(Float64.(Array(c)))
        @test all(abs.(actual .- expected) .<= 0.02 .+ 0.01 .* abs.(expected))
    end
    # Shapes of any size TMA addresses: partial tiles in M, N and K, zero-filled
    # and clipped; a partial group of row tiles; odd row tiles of clusters.
    for T in (BFloat16,Float8_E4M3FN), (m,n,k) in ((200,136,72),(1000,264,1040),(1300,2056,96)), persistent in (false,true), cluster in (1,2)
        (k*sizeof(T)) % 16 == 0 || continue
        ha = T.(randn(rng,Float32,k,m) ./ sqrt(Float32(k)))
        hb = T.(randn(rng,Float32,k,n))
        _, launch, c = hopper_gemm(CuArray(ha),CuArray(hb);persistent,cluster)
        fill!(c,BFloat16(NaN)); launch(); synchronize()
        expected = transpose(Float64.(Float32.(ha)))*Float64.(Float32.(hb))
        @test all(abs.(permutedims(Float64.(Array(c))) .- expected) .<= 0.02 .+ 0.01 .* abs.(expected))
    end
    @test_throws ArgumentError hopper_gemm(CUDACore.zeros(BFloat16,64,256),CUDACore.zeros(BFloat16,64,100))
end
end
