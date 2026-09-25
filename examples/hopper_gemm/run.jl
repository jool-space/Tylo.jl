# julia --project=test examples/hopper_gemm/run.jl [m n k]
# Checks the warp-specialized GEMM against cuBLAS, then tunes its tile width
# and stage count, timing cuBLAS as one more candidate.
using cuBLAS, Random
include("kernel.jl")
CUDACore.capability(CUDACore.device()) == v"9.0" || error("requires an H100 or H200")

# C (N×M, row-major M×N) = Bᵀ A in cuBLAS's terms, the same product.
function cublas_gemm(a,b,c)
    k, m = size(a); n = size(b,2)
    alpha, beta = CuArray(Float32[1]), CuArray(Float32[0])
    handle = cuBLAS.handle()
    cuBLAS.cublasSetPointerMode_v2(handle,cuBLAS.CUBLAS_POINTER_MODE_DEVICE)
    () -> cuBLAS.cublasGemmEx(handle,'T','N',n,m,k,alpha,b,eltype(b),k,a,eltype(a),k,beta,c,eltype(c),n,
                              cuBLAS.CUBLAS_COMPUTE_32F,cuBLAS.CUBLAS_GEMM_DEFAULT)
end

function main(m,n,k)
    rng = MersenneTwister(1)
    a = CuArray(BFloat16.(randn(rng,Float32,k,m) ./ sqrt(Float32(k))))
    b = CuArray(BFloat16.(randn(rng,Float32,k,n)))
    reference = CuArray{BFloat16}(undef,n,m)
    cublas = cublas_gemm(a,b,reference)
    cublas(); CUDACore.synchronize()
    expected = Float32.(Array(reference))
    println(CUDACore.name(device()),": C = AᵀB, BF16, ",(m,n,k))
    candidates = [:cublas; [(; n=w, stages=s) for w in (128,256) for s in 3:8]]
    tuning = tune(candidates) do cand
        cand === :cublas && return cublas
        kernel, launch, c = hopper_gemm(a,b;cand...)
        fill!(c,BFloat16(NaN)); launch()
        err = maximum(abs.(Float32.(Array(c)) .- expected))
        err <= 0.03f0*maximum(abs,expected) || error("$cand differs from cuBLAS by $err")
        kernel, launch
    end
    show(stdout,MIME("text/plain"),tuning); println()
    flops = 2.0*m*n*k
    for t in tuning.trials
        t.status === :timed || continue
        println(rpad(string(t.candidate),26),round(flops/t.median/1e12;digits=1)," TFLOP/s")
    end
    ours = first(filter(t -> t.status === :timed && t.candidate !== :cublas,tuning.trials))
    theirs = only(filter(t -> t.candidate === :cublas,tuning.trials))
    println("best Tylo: ",ours.candidate,", ",round(100*theirs.median/ours.median;digits=1),"% of cuBLAS")
end

m, n, k = isempty(ARGS) ? (4096,4096,4096) : parse.(Int,ARGS)
main(m,n,k)
