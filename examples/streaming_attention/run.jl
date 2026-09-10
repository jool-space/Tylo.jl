# Uses the benchmark environment with cuBLAS available; see README.md.
using Tylo,PTX,CUDACore,BFloat16s,cuBLAS,Random,TOML,Dates,LinearAlgebra
include("kernel.jl")
include("reference.jl")
include("../softmax/baseline.jl")

function measure(m,n;causal=false,padded=false,samples=41)
    rng=MersenneTwister(m+n)
    q=BFloat16.(randn(rng,Float32,64,m));k=BFloat16.(randn(rng,Float32,64,n))
    v=BFloat16.(randn(rng,Float32,n,64));mask=rand(rng,Float32,n,m).>0.1f0
    mask[:,2].=false
    if causal
        for i in 1:m,j in 1:n;mask[j,i] &= j<=i;end
    end
    started=time_ns()
    storage=v
    if padded
        storage=zeros(BFloat16,cld(n,8)*8,64);storage[1:n,:]=v
    end
    dq,dk,dv,dm=CuArray(q),CuArray(k),CuArray(storage),CuArray(mask)
    output=CuArray{Float32}(undef,64,m);baseline=similar(output)
    scores=CuArray{Float32}(undef,n,m);prob=CuArray{BFloat16}(undef,n,m)
    CUDACore.synchronize();preparation_ms=(time_ns()-started)/1e6
    config=StreamingAttention.configuration()
    args=(output,dq,dk,dv,dm,Int32(m),Int32(n),config,Val(causal))
    started=time_ns()
    kernel=@cuda launch=false StreamingAttention.attention_kernel!(args...)
    fused=()->kernel(args...;threads=128,blocks=cld(m,64),shmem=StreamingAttention.shared_bytes(config))
    # The high-level wrapper constructs device scalar references per call.
    # Allocate them before capture and retain them across all graph replays.
    scalars=(scale=CUDACore.CuRef(0.125f0),one=CUDACore.CuRef(1f0),zero=CUDACore.CuRef(0f0))
    materialized=()->begin
        cuBLAS.cublasGemmEx(cuBLAS.handle(),'T','N',n,m,64,scalars.scale,dk,BFloat16,64,
            dq,BFloat16,64,scalars.zero,scores,Float32,n,cuBLAS.CUBLAS_COMPUTE_32F,cuBLAS.CUBLAS_GEMM_DEFAULT)
        @cuda threads=128 blocks=cld(m,4) baseline_softmax_kernel!(prob,scores,dm)
        cuBLAS.cublasGemmEx(cuBLAS.handle(),'T','N',64,m,n,scalars.one,dv,BFloat16,size(dv,1),
            prob,BFloat16,n,scalars.zero,baseline,Float32,64,cuBLAS.CUBLAS_COMPUTE_32F,cuBLAS.CUBLAS_GEMM_DEFAULT)
    end
    fused();materialized();CUDACore.synchronize();compile_and_first_ms=(time_ns()-started)/1e6
    reference=attention_reference(q,k,v,mask)
    yf,yb=Array(output),Array(baseline)
    bound=0.004*maximum(abs,Float64.(v))+3e-5
    @assert maximum(abs,yf.-reference)<=bound
    @assert maximum(abs,yb.-reference)<=bound
    graphs=[CUDACore.instantiate(CUDACore.capture() do
        for _ in 1:8;launch();end
    end) for launch in (fused,materialized)]
    times=[Float64[],Float64[]]
    GC.@preserve args dq dk dv dm output baseline scores prob scalars fused materialized begin
    for (i,graph) in enumerate(graphs),j in 1:10
        CUDACore.launch(graph)
    end
    CUDACore.synchronize()
    for _ in 1:samples,i in randperm(rng,2)
        push!(times[i],1e6*(CUDACore.@elapsed CUDACore.launch(graphs[i]))/8)
    end
    end
    records=[Dict("implementation"=>name,"median_us"=>sort(times[i])[cld(samples,2)],"samples_us"=>times[i],
        "max_abs_error"=>maximum(abs,(i==1 ? yf : yb).-reference),"explicit_workspace_bytes"=>i==1 ? 0 : 6m*n)
        for (i,name) in enumerate(("streaming","materialized_cublas"))]
    println((;m,n,causal,padded,registers=CUDACore.registers(kernel),local_bytes=CUDACore.memory(kernel).local,
        times=[(r["implementation"],r["median_us"],r["max_abs_error"]) for r in records]));flush(stdout)
    Dict("m"=>m,"n"=>n,"causal"=>causal,"v_leading_dimension"=>size(dv,1),"preparation_ms"=>preparation_ms,
        "compilation_and_first_execution_ms"=>compile_and_first_ms,"registers"=>CUDACore.registers(kernel),
        "local_bytes"=>CUDACore.memory(kernel).local,
        "max_active_blocks_per_sm"=>CUDACore.active_blocks(kernel.fun,128;shmem=StreamingAttention.shared_bytes(config)),
        "theoretical_occupancy"=>CUDACore.occupancy(kernel.fun,128;shmem=StreamingAttention.shared_bytes(config)),
        "query_ctas"=>cld(m,64),"shared_bytes_per_cta"=>StreamingAttention.shared_bytes(config),
        "records"=>records)
end
function main()
    out=abspath(only(ARGS));ispath(out)&&error("output exists");mkpath(out)
    CUDACore.math_mode!(CUDACore.DEFAULT_MATH)
    cases=[measure(m,n;causal) for (m,n,causal) in ((64,64,false),(129,257,false),(256,256,false),(1024,1024,false),(2048,2048,false),(1024,1024,true))]
    push!(cases,measure(129,257;padded=true))
    report=Dict("created_at"=>string(now(UTC)),"device"=>CUDACore.name(device()),"julia"=>string(VERSION),
        "compiler"=>string(CUDACore.compiler_version()),"cases"=>cases,
        "sm_count"=>CUDACore.attribute(device(),CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        "measurement"=>"41 interleaved samples, 8 repetitions per graph, warmed; same BF16 Q/K/V and effective Boolean mask, scale 1/8; timings exclude preparation/compilation/transfers",
        "numerics"=>"streaming FP32 statistics with BF16 unnormalized weights per 32-key tile; baseline BF16 GEMMs with FP32 compute/output, softmax writes BF16 normalized probabilities; DEFAULT_MATH; different rounding boundaries",
        "workspace"=>"explicit score/probability buffers only; excludes common inputs/output/mask and cuBLAS internal workspace; streaming uses 16 KiB CTA shared memory")
    open(io->TOML.print(io,report),joinpath(out,"results.toml"),"w")
end
main()
