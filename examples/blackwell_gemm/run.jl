# julia --project=test examples/blackwell_gemm/run.jl [bf16|fp8] [m n k]
# Checks the warp-specialized Blackwell GEMM against cuBLASLt, then tunes
# its tile width, stage count, grouping and CTA pairing, timing cuBLASLt
# as one more candidate.
using Random
include("kernel.jl")
include(joinpath(@__DIR__,"..","hopper_gemm","cublaslt.jl"))
CUDACore.capability(CUDACore.device()).major == 10 || error("requires a B200 or B300")

function main(::Type{T},m,n,k) where T
    rng = MersenneTwister(1)
    a = CuArray(T.(randn(rng,Float32,k,m) ./ sqrt(Float32(k))))
    b = CuArray(T.(randn(rng,Float32,k,n)))
    reference = CuArray{BFloat16}(undef,n,m)
    library = cublaslt_gemm(a,b,reference)
    library(); CUDACore.synchronize()
    expected = Float32.(Array(reference))
    println(CUDACore.name(device()),": C = AᵀB, ",T," inputs, BF16 C, ",(m,n,k))
    candidates = [:cublaslt; [(; n=w, stages=s, group=g, ctas=p) for w in (128,256) for s in 3:7 for g in (4,8,16) for p in (1,2)]]
    tuning = tune(candidates) do cand
        cand === :cublaslt && return library
        kernel, launch, c = blackwell_gemm(a,b;cand...)
        fill!(c,BFloat16(NaN)); launch()
        err = maximum(abs.(Float32.(Array(c)) .- expected))
        err <= 0.03f0*maximum(abs,expected) || error("$cand differs from cuBLASLt by $err")
        kernel, launch
    end
    show(stdout,MIME("text/plain"),tuning); println()
    flops = 2.0*m*n*k
    ours = first(filter(t -> t.status === :timed && t.candidate !== :cublaslt,tuning.trials))
    theirs = only(filter(t -> t.candidate === :cublaslt,tuning.trials))
    println("cuBLASLt ",round(flops/theirs.median/1e12;digits=1)," TFLOP/s; Tylo ",ours.candidate," ",
            round(flops/ours.median/1e12;digits=1)," TFLOP/s: ",round(100*theirs.median/ours.median;digits=1),"% of cuBLASLt")
end

T = get(ARGS,1,"bf16") == "fp8" ? Float8_E4M3FN : BFloat16
m, n, k = length(ARGS) >= 4 ? parse.(Int,ARGS[2:4]) : (4096,4096,4096)
main(T,m,n,k)
