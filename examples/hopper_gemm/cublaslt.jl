# cuBLASLt's GEMM as the reference for FP8, which cublasGemmEx does not
# offer: C (N×M, row-major M×N) = Bᵀ A in cuBLAS's column-major terms, the
# same product the Tylo kernel computes, with FP32 accumulation and BF16 C.
using cuBLAS

_lt_type(::Type{Float8_E4M3FN}) = CUDACore.R_8F_E4M3
_lt_type(::Type{Float8_E5M2}) = CUDACore.R_8F_E5M2
_lt_type(::Type{BFloat16}) = CUDACore.R_16BF
_lt_type(::Type{Float16}) = CUDACore.R_16F

function cublaslt_gemm(a::CuMatrix{T},b::CuMatrix{T},c::CuMatrix{BFloat16}) where T
    k, m = size(a); n = size(b,2)
    handle = Ref{cuBLAS.cublasLtHandle_t}()
    cuBLAS.cublasLtCreate(handle)
    desc = Ref{cuBLAS.cublasLtMatmulDesc_t}()
    cuBLAS.cublasLtMatmulDescCreate(desc,cuBLAS.CUBLAS_COMPUTE_32F,CUDACore.R_32F)
    transa, transb = Ref(cuBLAS.CUBLAS_OP_T), Ref(cuBLAS.CUBLAS_OP_N)
    cuBLAS.cublasLtMatmulDescSetAttribute(desc[],cuBLAS.CUBLASLT_MATMUL_DESC_TRANSA,transa,sizeof(transa[]))
    cuBLAS.cublasLtMatmulDescSetAttribute(desc[],cuBLAS.CUBLASLT_MATMUL_DESC_TRANSB,transb,sizeof(transb[]))
    layout(type,rows,cols,ld) = (l = Ref{cuBLAS.cublasLtMatrixLayout_t}(); cuBLAS.cublasLtMatrixLayoutCreate(l,type,rows,cols,ld); l[])
    la = layout(_lt_type(T),k,n,k)            # cuBLAS's A: our B, K×N, transposed
    lb = layout(_lt_type(T),k,m,k)            # cuBLAS's B: our A, K×M
    lc = layout(CUDACore.R_16BF,n,m,n)
    workspace = CuArray{UInt8}(undef,32*1024*1024)
    preference = Ref{cuBLAS.cublasLtMatmulPreference_t}()
    cuBLAS.cublasLtMatmulPreferenceCreate(preference)
    bytes = Ref{UInt64}(sizeof(workspace))
    cuBLAS.cublasLtMatmulPreferenceSetAttribute(preference[],cuBLAS.CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,bytes,sizeof(bytes[]))
    results = Vector{cuBLAS.cublasLtMatmulHeuristicResult_t}(undef,1)
    found = Ref{Cint}(0)
    cuBLAS.cublasLtMatmulAlgoGetHeuristic(handle[],desc[],la,lb,lc,lc,preference[],1,results,found)
    found[] > 0 || error("cuBLASLt has no algorithm for this GEMM")
    algo = Ref(results[1].algo)
    alpha, beta = Ref(1f0), Ref(0f0)
    () -> GC.@preserve workspace cuBLAS.cublasLtMatmul(handle[],desc[],alpha,b,la,a,lb,beta,c,lc,c,lc,
                                                       algo,workspace,sizeof(workspace),CUDACore.stream())
end
