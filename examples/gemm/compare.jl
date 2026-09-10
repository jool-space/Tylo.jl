# julia --project=test/gpu examples/gemm/compare.jl OUTPUT_DIR
using Tylo,PTX,CUDACore,BFloat16s,Random,TOML,SHA,Dates
include("kernel.jl")
include("../../test/gpu/codegen.jl")
const ROOT=normpath(joinpath(@__DIR__,"../.."))
const BASELINE="0ad5de27d350ebc693ba53c78e50daf91cf85a9c"
module GemmReference end
function measure(output,m,n,k;samples=41)
    rng=MersenneTwister(m+n+k)
    a=BFloat16.(0.2f0 .* randn(rng,Float32,m,k));b=BFloat16.(0.2f0 .* randn(rng,Float32,k,n))
    expected=Float64.(a)*Float64.(b)
    full=m%64==n%64==k%32==0
    names=full ? ("before","aligned","bounded") : ("bounded_compact","bounded_padded")
    variants=[]
    for name in names
        ld=name=="bounded_padded" ? 8cld(k,8) : k
        ha,hb=zeros(BFloat16,ld,m),zeros(BFloat16,ld,n)
        ha[1:k,:] .= permutedims(a);hb[1:k,:] .= b
        da,db=CuArray(ha),CuArray(hb);out=CuArray{Float32}(undef,m,n)
        cfg=name=="before" ? GemmReference.gemm_config(BFloat16) : gemm_config(BFloat16;bounds=!(name=="aligned"))
        f=name=="before" ? GemmReference.tiled_gemm_kernel! : tiled_gemm_kernel!
        args=(out,da,db,Int32(m),Int32(n),Int32(k),Int32(ld),Int32(ld),Int32(m),cfg,1f0,Val(false))
        kernel=@cuda launch=false f(args...)
        launch=()->kernel(args...;threads=128,blocks=(cld(m,64),cld(n,64)),shmem=shared_bytes(cfg,BFloat16))
        launch();actual=Array(out)
        @assert all(abs.(actual .- expected) .<= 2e-4 .+ 2e-4 .* abs.(expected))
        graph=CUDACore.instantiate(CUDACore.capture() do
            for _ in 1:16
                launch()
            end
        end)
        tt=Tuple{map(x->typeof(CUDACore.cudaconvert(x)),args)...}
        code=compile_kernel(f,tt;arch=CUDACore.SMVersion(12,1,:arch),threads=128)
        stem="gemm-$m-$n-$k-$name"
        write(joinpath(output,stem*".ptx"),code.ptx);write(joinpath(output,stem*".cubin"),code.image)
        push!(variants,(;name,ld,args,kernel,graph,code,times=Float64[]))
    end
    for v in variants,_ in 1:5
        CUDACore.launch(v.graph)
    end
    CUDACore.synchronize()
    for _ in 1:samples,i in randperm(rng,length(variants))
        v=variants[i];push!(v.times,1e6*(CUDACore.@elapsed CUDACore.launch(v.graph))/16)
    end
    records=[Dict("implementation"=>v.name,"leading_dimension"=>v.ld,"median_us"=>sort(v.times)[cld(samples,2)],
        "min_us"=>minimum(v.times),"samples_us"=>v.times,"registers"=>CUDACore.registers(v.kernel),
        "local_bytes"=>CUDACore.memory(v.kernel).local) for v in variants]
    result=Dict("m"=>m,"n"=>n,"k"=>k,"records"=>records)
    if full
        result["aligned_machine_code_identical"]=kernel_text(variants[1].code.image,".text._Z18tiled_gemm_kernel_")==
                                                 kernel_text(variants[2].code.image,".text._Z18tiled_gemm_kernel_")
    end
    println((;m,n,k,timings=[(r["implementation"],round(r["median_us"];digits=3),r["registers"],r["local_bytes"]) for r in records]));flush(stdout)
    result
end
function main()
    length(ARGS)==1 || error("provide a new output directory")
    output=abspath(only(ARGS));ispath(output)&&error("output exists");mkpath(output)
    source=read(`git -C $ROOT show $BASELINE:examples/gemm/kernel.jl`,String)
    include_string(GemmReference,source,"reference-gemm.jl")
    write(joinpath(output,"reference-gemm.jl"),source)
    cases=Base.invokelatest(()->[measure(output,m,n,k) for (m,n,k) in ((512,512,512),(65,97,73),(513,511,509))])
    report=Dict("created_at"=>string(now(UTC)),"baseline"=>BASELINE,"reference_sha256"=>bytes2hex(sha256(source)),
        "device"=>CUDACore.name(device()),"julia"=>string(VERSION),"compiler"=>string(CUDACore.compiler_version()),
        "measurement"=>"BF16 GEMM; graph batch of 16 warm kernels; excludes compilation, packing, padding and transfers", "cases"=>cases)
    open(io->TOML.print(io,report),joinpath(output,"results.toml"),"w")
end
main()
