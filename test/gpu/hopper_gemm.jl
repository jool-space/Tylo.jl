# TEST_TARGET: cc==9.0
# The warp-specialized Hopper GEMM example: register reallocation, a TMA
# pipeline and pipelined WGMMA, compiled clean for sm_90a and correct on Hopper.
include("../../examples/hopper_gemm/kernel.jl")

@testset "Warp-specialized Hopper GEMM assembly" begin
    T = BFloat16
    for (n,stages) in ((128,6),(256,4))
        ta = Tylo.TMABinding{typeof(TMATile(T,(128,64),2)),PTX.TMADescriptorPtr,2}
        tb = Tylo.TMABinding{typeof(TMATile(T,(64,n),1)),PTX.TMADescriptorPtr,2}
        tt = Tuple{CuDeviceMatrix{T,1},ta,tb,typeof(WGMMAAtom(T,n,64)),Int32,Val{stages},Val{8}}
        code = compile_kernel(hopper_gemm_kernel!,tt;arch=CUDACore.SMVersion(9,0,:arch),threads=384)
        save_code("hopper-gemm-n$n-s$stages",code)
        @test isempty(audit(code))
        body = entry_body(code.ptx)
        @test occursin(".reqntid 384",code.ptx)
        @test count("setmaxnreg.dec.sync.aligned.u32",body) == 1 && count("setmaxnreg.inc.sync.aligned.u32",body) == 1
        @test count("USETMAXREG",Tylo.kernel_sass(code)) == 2
        @test count("wgmma.wait_group.sync.aligned 1",body) == 1      # one group in flight
        @test count("cp.async.bulk.tensor.2d",body) == 2
    end
end

if capability_is(v"9.0")
@testset "Warp-specialized Hopper GEMM on Hopper" begin
    rng = MersenneTwister(5)
    for (m,n,k) in ((256,256,256),(384,512,640)), (w,stages) in ((128,3),(256,4))
        n % w == 0 || continue
        ha = BFloat16.(randn(rng,Float32,k,m) ./ sqrt(Float32(k)))
        hb = BFloat16.(randn(rng,Float32,k,n))
        _, launch, c = hopper_gemm(CuArray(ha),CuArray(hb);n=w,stages)
        launch(); synchronize()
        expected = transpose(Float64.(ha))*Float64.(hb)                # M×N
        actual = permutedims(Float64.(Array(c)))
        @test all(abs.(actual .- expected) .<= 0.02 .+ 0.01 .* abs.(expected))
    end
end
end
