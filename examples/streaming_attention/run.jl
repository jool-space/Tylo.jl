# Uses the benchmark environment with cuBLAS available; see README.md.
using Tylo,PTX,CUDACore,BFloat16s,cuBLAS,Random,TOML,Dates,LinearAlgebra
include("kernel.jl")
include("reference.jl")
include("../softmax/baseline.jl")

# One head or many; a random Boolean mask or none; causal or full.
function measure(m,n;heads=1,causal=false,masked=true,padded=false,samples=41)
    rng=MersenneTwister(m+n+heads)
    q=BFloat16.(randn(rng,Float32,64,m,heads));k=BFloat16.(randn(rng,Float32,64,n,heads))
    v=BFloat16.(randn(rng,Float32,n,64,heads))
    mask=masked ? rand(rng,Float32,n,m).>0.1f0 : nothing
    masked && m>1 && (mask[:,2].=false)
    effective=masked ? copy(mask) : trues(n,m)
    causal && for i in 1:m,j in 1:n;effective[j,i] &= j<=i;end
    started=time_ns()
    storage=v
    if padded
        storage=zeros(BFloat16,cld(n,8)*8,64,heads);storage[1:n,:,:]=v
    end
    dq,dk,dv=CuArray(q),CuArray(k),CuArray(storage)
    dm=masked ? CuArray(mask) : nothing
    # The materialized baseline reads one effective mask per head unless
    # every score is valid; its softmax runs over all heads' rows at once.
    baseline_mask=all(effective) ? nothing : CuArray(repeat(effective,1,heads))
    output=CuArray{Float32}(undef,64,m,heads);baseline=similar(output)
    scores=CuArray{Float32}(undef,n,m,heads);prob=CuArray{BFloat16}(undef,n,m,heads)
    CUDACore.synchronize();preparation_ms=(time_ns()-started)/1e6
    config=StreamingAttention.configuration()
    prepared=StreamingAttention.prepare(dq,dk,StreamingAttention.pad_values(dv);config)
    args=(output,prepared.q,prepared.k,prepared.v,dm,Int32(m),Int32(n),config,Val(causal))
    started=time_ns()
    kernel=@cuda launch=false StreamingAttention.attention_kernel!(args...)
    shmem=StreamingAttention.shared_bytes(config)
    CUDACore.attributes(kernel.fun)[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES]=shmem
    fused=()->kernel(args...;threads=256,blocks=(cld(m,128),heads),shmem)
    # The high-level wrapper constructs device scalar references per call.
    # Allocate them before capture and retain them across all graph replays.
    scalars=(scale=CUDACore.CuRef(0.125f0),one=CUDACore.CuRef(1f0),zero=CUDACore.CuRef(0f0))
    ldv=size(dv,1)
    materialized=()->begin
        cuBLAS.cublasGemmStridedBatchedEx(cuBLAS.handle(),'T','N',n,m,64,scalars.scale,dk,BFloat16,64,64n,
            dq,BFloat16,64,64m,scalars.zero,scores,Float32,n,n*m,heads,cuBLAS.CUBLAS_COMPUTE_32F,cuBLAS.CUBLAS_GEMM_DEFAULT)
        @cuda threads=128 blocks=cld(m*heads,4) baseline_softmax_kernel!(reshape(prob,n,m*heads),reshape(scores,n,m*heads),baseline_mask)
        cuBLAS.cublasGemmStridedBatchedEx(cuBLAS.handle(),'T','N',64,m,n,scalars.one,dv,BFloat16,ldv,64ldv,
            prob,BFloat16,n,n*m,scalars.zero,baseline,Float32,64,64m,heads,cuBLAS.CUBLAS_COMPUTE_32F,cuBLAS.CUBLAS_GEMM_DEFAULT)
    end
    fused();materialized();CUDACore.synchronize();compile_and_first_ms=(time_ns()-started)/1e6
    reference=cat((attention_reference(q[:,:,h],k[:,:,h],v[:,:,h],effective) for h in 1:heads)...;dims=3)
    yf,yb=Array(output),Array(baseline)
    bound=0.004*maximum(abs,Float64.(v))+3e-5
    @assert maximum(abs,yf.-reference)<=bound
    @assert maximum(abs,yb.-reference)<=bound
    graphs=[CUDACore.instantiate(CUDACore.capture() do
        for _ in 1:8;launch();end
    end) for launch in (fused,materialized)]
    times=[Float64[],Float64[]]
    GC.@preserve prepared args dq dk dv dm output baseline scores prob scalars baseline_mask fused materialized begin
    for (i,graph) in enumerate(graphs),j in 1:10
        CUDACore.launch(graph)
    end
    CUDACore.synchronize()
    for _ in 1:samples,i in randperm(rng,2)
        push!(times[i],1e6*(CUDACore.@elapsed CUDACore.launch(graphs[i]))/8)
    end
    end
    # Useful work: four flops per valid query/key pair and head dimension.
    pairs=causal ? sum(min(i,n) for i in 1:m) : m*n
    flops=4*64*pairs*heads
    records=[Dict("implementation"=>name,"median_us"=>sort(times[i])[cld(samples,2)],"samples_us"=>times[i],
        "tflops"=>flops/sort(times[i])[cld(samples,2)]/1e6,
        "max_abs_error"=>maximum(abs,(i==1 ? yf : yb).-reference),"explicit_workspace_bytes"=>i==1 ? 0 : 6m*n*heads)
        for (i,name) in enumerate(("streaming","materialized_cublas"))]
    println((;m,n,heads,causal,masked,padded,registers=CUDACore.registers(kernel),local_bytes=CUDACore.memory(kernel).local,
        times=[(r["implementation"],round(r["median_us"],digits=2),round(r["tflops"],digits=2),r["max_abs_error"]) for r in records]));flush(stdout)
    Dict("m"=>m,"n"=>n,"heads"=>heads,"causal"=>causal,"masked"=>masked,"v_leading_dimension"=>size(dv,1),"preparation_ms"=>preparation_ms,
        "compilation_and_first_execution_ms"=>compile_and_first_ms,"registers"=>CUDACore.registers(kernel),
        "local_bytes"=>CUDACore.memory(kernel).local,
        "max_active_blocks_per_sm"=>CUDACore.active_blocks(kernel.fun,256;shmem),
        "theoretical_occupancy"=>CUDACore.occupancy(kernel.fun,256;shmem),
        "query_ctas"=>cld(m,128)*heads,"shared_bytes_per_cta"=>shmem,"flops"=>flops,
        "records"=>records)
end
function main()
    out=abspath(ARGS[1]);ispath(out)&&error("output exists");mkpath(out)
    CUDACore.math_mode!(CUDACore.DEFAULT_MATH)
    quick=length(ARGS)>1 && ARGS[2]=="quick" # a subset while iterating on the schedule
    cases=[measure(m,n;causal) for (m,n,causal) in (quick ? ((129,257,false),(2048,2048,false)) :
        ((64,64,false),(129,257,false),(256,256,false),(1024,1024,false),(2048,2048,false),(1024,1024,true)))]
    quick || push!(cases,measure(129,257;padded=true))
    for (m,n,heads,causal) in (quick ? ((1024,1024,16,false),(2048,2048,8,true)) :
            ((512,512,32,false),(1024,1024,16,false),(1024,1024,16,true),(2048,2048,8,false),(2048,2048,8,true),(4096,4096,4,false)))
        push!(cases,measure(m,n;heads,causal,masked=false))
    end
    report=Dict("created_at"=>string(now(UTC)),"device"=>CUDACore.name(device()),"julia"=>string(VERSION),
        "compiler"=>string(CUDACore.compiler_version()),"cases"=>cases,
        "sm_count"=>CUDACore.attribute(device(),CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        "measurement"=>"41 interleaved samples, 8 repetitions per graph, warmed; same BF16 Q/K/V and effective Boolean mask, scale 1/8; timings exclude preparation/compilation/transfers; tflops count four flops per valid query/key pair and head dimension",
        "numerics"=>"streaming FP32 statistics with BF16 unnormalized weights per 64-key tile; baseline BF16 GEMMs with FP32 compute/output, softmax writes BF16 normalized probabilities; DEFAULT_MATH; different rounding boundaries",
        "workspace"=>"explicit score/probability buffers only; excludes common inputs/output/mask and cuBLAS internal workspace; streaming uses 81 KiB CTA shared memory")
    open(io->TOML.print(io,report),joinpath(out,"results.toml"),"w")
end
main()
