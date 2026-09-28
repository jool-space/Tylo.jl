# TEST_TARGET: cc==10
# The warp-specialized Blackwell GEMM example: a TMA pipeline, tcgen05 MMAs
# into a double-buffered TMEM accumulator, and CTA pairs that share one
# 256-row MMA; compiled clean for sm_100a and correct on B200/B300.
include("../../examples/blackwell_gemm/kernel.jl")

@testset "Warp-specialized Blackwell GEMM assembly" begin
    T = BFloat16
    for (ctas,n,stages) in ((1,256,4),(2,256,6),(2,128,6))
        ta = Tylo.TMABinding{typeof(TMATile(T,(128,64),2)),PTX.TMADescriptorPtr,2}
        tb = Tylo.TMABinding{typeof(TMATile(T,(64,n÷ctas),1)),PTX.TMADescriptorPtr,2}
        atom = Tcgen05Atom((128ctas,n,16),T)
        tt = Tuple{CuDeviceMatrix{T,1},ta,tb,typeof(atom),Int32,Val{stages},Val{8}}
        code = compile_kernel(blackwell_gemm_kernel!,tt;arch=CUDACore.SMVersion(10,0,:arch),threads=256)
        save_code("blackwell-gemm-ctas$ctas-n$n-s$stages",code)
        @test isempty(audit(code))
        # The MMA warp issues in line: no MMA sits in a loop over the lanes
        # that might issue it.
        sass = split(Tylo.kernel_sass(code),'\n')
        @test !any(i -> occursin("UTCHMMA",sass[i]) && any(l -> occursin("BRA.U.ANY",l),sass[i+1:min(i+4,end)]),eachindex(sass))
        body = entry_body(code.ptx)
        # One stage is four MMAs, every tcgen05 instruction in one CTA group.
        @test count("tcgen05.mma.cta_group::$ctas.kind::f16",body) == 4
        @test count("tcgen05.mma",body) == 4
        @test count("tcgen05.alloc.cta_group::$ctas",body) == 1 && count("tcgen05.dealloc.cta_group::$ctas",body) == 1
        @test count("tcgen05.commit.cta_group::$ctas",body) == 2 && count("tcgen05.commit",body) == 2
        @test occursin("tcgen05.ld.sync.aligned.32x32b.x32",body)
        if ctas == 2
            # Commits arrive in both CTAs; both CTAs' loads complete on the
            # even CTA's barriers; the epilogues release it remotely.
            @test count("multicast::cluster",body) == 2
            @test count("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster",body) == 2
            @test occursin("mbarrier.arrive.shared::cluster",body)
            @test count("barrier.cluster.wait",body) == 2
        else
            @test !occursin("barrier.cluster",body) && !occursin("multicast",body) && !occursin("mapa",body)
        end
    end
end

if runtime_supported(@__FILE__)
@testset "Warp-specialized Blackwell GEMM on Blackwell" begin
    rng = MersenneTwister(5)
    # 4096×2560 has 320 tiles of 128×256 (160 of 256×256): more than two per
    # SM (per pair), so persistent CTAs cycle both TMEM accumulators.
    for T in (BFloat16,Float8_E4M3FN), (m,n,k) in ((256,256,256),(512,512,640),(4096,2560,256)),
            (w,stages) in ((128,4),(256,4)), ctas in (1,2)
        n % w == 0 && m % (128ctas) == 0 && k % (128 ÷ sizeof(T)) == 0 || continue
        ha = T.(randn(rng,Float32,k,m) ./ sqrt(Float32(k)))
        hb = T.(randn(rng,Float32,k,n))
        _, launch, c = blackwell_gemm(CuArray(ha),CuArray(hb);n=w,stages,group=1,ctas)
        fill!(c,BFloat16(NaN)); launch(); synchronize()
        expected = transpose(Float64.(Float32.(ha)))*Float64.(Float32.(hb))   # M×N
        actual = permutedims(Float64.(Array(c)))
        @test all(abs.(actual .- expected) .<= 0.02 .+ 0.01 .* abs.(expected))
    end
end
end
