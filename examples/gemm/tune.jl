# Tune the tiled GEMM's configuration for this device and shape:
#   julia --project=test examples/gemm/tune.jl [m n k]
# Candidates the configuration refuses, or whose machine code fails the audit
# (spills, local memory), are rejected before anything runs; the rest are
# timed, and the winner is checked against a reference product.
using Tylo, PTX, CUDACore, BFloat16s, Random
include("kernel.jl")
CUDACore.functional() || error("CUDA device required")

function tune_gemm(m,n,k)
    rng = MersenneTwister(7)
    a = BFloat16.(0.2f0 .* randn(rng,Float32,m,k))
    b = BFloat16.(0.2f0 .* randn(rng,Float32,k,n))
    da, db = CuArray(permutedims(a)), CuArray(b)
    out = CuArray{Float32}(undef,m,n)
    bounds = m%64 != 0 || n%64 != 0 || k%32 != 0
    candidates = [(; block, warps, stages, swizzled)
                  for block in ((64,64,32),(64,64,64),(128,64,32),(64,128,32),(128,128,32),(128,128,64)),
                      warps in ((2,2),(4,2),(2,4),(4,4)), stages in (1,2), swizzled in (true,)]
    launcher(c) = begin
        config = gemm_config(BFloat16;c...,bounds)
        bm, bn, _ = size(config.plan)
        (bounds || (m%bm == 0 && n%bn == 0)) || throw(ArgumentError("block does not divide the matrix"))
        shmem = shared_bytes(gemm_shared(config,BFloat16))
        shmem <= 99*1024 || throw(ArgumentError("needs $(shmem ÷ 1024) KiB of shared memory"))
        args = (out,da,db,Int32(m),Int32(n),Int32(k),Int32(k),Int32(k),Int32(m),config,1f0)
        kernel = @cuda launch=false tiled_gemm_kernel!(args...)
        shmem > 48*1024 && (CUDACore.attributes(kernel.fun)[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem)
        kernel, () -> kernel(args...;threads=Tylo.threads(config.plan),blocks=(cld(m,bm),cld(n,bn)),shmem)
    end
    tuning = tune(launcher,candidates)
    # Check the winner's product.
    _, launch = launcher(Tylo.best(tuning))
    fill!(out,NaN32); launch()
    expected = Float32.(a)*Float32.(b)
    @assert all(abs.(Array(out) .- expected) .<= 2f-4 .+ 2f-4 .* abs.(expected))
    tuning
end

m, n, k = isempty(ARGS) ? (2048,2048,2048) : parse.(Int,ARGS)
println(CUDACore.name(device()),": BF16 GEMM ",(m,n,k))
tuning = tune_gemm(m,n,k)
show(stdout,MIME("text/plain"),tuning); println()
fastest = first(tuning.trials)
println("best: ",fastest.candidate,", ",round(2e-12*m*n*k/fastest.median;digits=1)," TFLOP/s")
