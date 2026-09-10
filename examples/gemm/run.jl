using Tylo, PTX, CUDACore, BFloat16s, Random
include("kernel.jl")
CUDACore.functional() || error("CUDA device required")

function demo(;m=512,n=512,k=512)
    rng = MersenneTwister(42)
    a = BFloat16.(0.2f0 .* randn(rng,Float32,m,k))
    b = BFloat16.(0.2f0 .* randn(rng,Float32,k,n))
    da,db = CuArray(permutedims(a)),CuArray(b)
    expected = Float32.(a)*Float32.(b)
    println("Device: ",CUDACore.name(device()),"; A * B = ",(m,n,k))
    cases = NamedTuple[]
    # Graph nodes retain device addresses. The cases retain output allocation
    # owners, and all inputs/outputs stay live through capture and replay.
    GC.@preserve da db cases begin
        for swizzled in (false,true),stages in (1,2)
            bounds = m%64 != 0 || n%64 != 0 || k%32 != 0
            cfg = gemm_config(BFloat16;swizzled,stages,bounds)
            bm,bn,bk = size(cfg.plan)
            out = CuArray(fill(NaN32,m,n))
            function launch()
                @cuda threads=Tylo.threads(cfg.plan) blocks=(cld(m,bm),cld(n,bn)) shmem=shared_bytes(cfg,BFloat16) tiled_gemm_kernel!(
                    out,da,db,Int32(m),Int32(n),Int32(k),Int32(k),Int32(k),Int32(m),cfg,1f0,Val(false))
            end
            launch()
            actual = Array(out)
            @assert all(abs.(actual .- expected) .<= 2f-4 .+ 2f-4 .* abs.(expected))
            graph = GC.@preserve out CUDACore.instantiate(CUDACore.capture() do
                for _ in 1:32
                    launch()
                end
            end)
            push!(cases,(;swizzled,stages,out,graph,times=Float64[]))
        end
        # Compile every configuration before measurement; alternate order
        # each round so startup and clock drift do not always favor one case.
        for round in 1:27
            order = isodd(round) ? (1,2,3,4) : (4,3,2,1)
            for i in order
                elapsed = CUDACore.@elapsed CUDACore.launch(cases[i].graph)
                round > 6 && push!(cases[i].times,elapsed*1e6/32)
            end
        end
        for case in cases
            microseconds = sort(case.times)[11]
            println((;case.swizzled,case.stages,microseconds,
                      tflops=2.0*m*n*k/(microseconds*1e6)))
        end
    end

end

if isempty(ARGS)
    demo()
else
    length(ARGS)==3 || error("usage: run.jl [M N K]")
    m,n,k=parse.(Int,ARGS)
    min(m,n,k)>0 || error("positive dimensions required")
    demo(;m,n,k)
end
