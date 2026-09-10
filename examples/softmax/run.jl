# julia --project=test/gpu examples/softmax/run.jl OUTPUT_DIR
using Tylo,PTX,CUDACore,BFloat16s,Random,TOML,Dates
include("kernel.jl")
using .SoftmaxExample: lane_softmax_kernel!,warp_softmax_kernel!

# A straightforward three-pass warp kernel: explicit scalar loops and
# shuffles, without Tylo fragments. Same arrays, mask, dtype and row ownership.
@inline function baseline_reduce(op::F,x) where F
    PTX.Utils.@unroll for offset in (16,8,4,2,1)
        x=op(x,shfl_xor_sync(typemax(UInt32),x,offset))
    end
    x
end
function baseline_softmax_kernel!(out,input,mask)
    tid=Int32(threadIdx().x)-Int32(1);lane=tid%Int32(32)
    row=(Int32(blockIdx().x)-Int32(1))*Int32(4)+tid÷Int32(32)+Int32(1)
    width=size(input,1)%Int32;rows=size(input,2)%Int32;m=-Inf32
    for j in lane+Int32(1):Int32(32):width
        if row<=rows && (@inbounds mask[j,row])
            m=max(m,Float32(@inbounds input[j,row]))
        end
    end
    m=baseline_reduce(max,m);s=0f0
    for j in lane+Int32(1):Int32(32):width
        if row<=rows && (@inbounds mask[j,row])
            s+=exp(Float32(@inbounds input[j,row])-m)
        end
    end
    s=baseline_reduce(+,s)
    for j in lane+Int32(1):Int32(32):width
        if row<=rows
            @inbounds out[j,row]=mask[j,row] && s>0f0 ? exp(Float32(input[j,row])-m)/s : 0f0
        end
    end
    nothing
end
function measure(width,rows;samples=41)
    rng=MersenneTwister(width+rows)
    input=CuArray(randn(rng,Float32,width,rows)); mask=CuArray(rand(rng,Float32,width,rows) .> 0.2f0)
    variants=[];reference=nothing
    for name in ("scalar_warp","tylo_warp","tylo_lane")
        output=similar(input)
        if name=="scalar_warp"
            kernel=@cuda launch=false baseline_softmax_kernel!(output,input,mask)
            args=(output,input,mask);blocks=cld(rows,4)
        elseif name=="tylo_warp"
            args=(output,input,mask,Val(cld(width,32)));blocks=cld(rows,4)
            kernel=@cuda launch=false warp_softmax_kernel!(args...)
        else
            args=(output,input,mask,Val(width));blocks=cld(rows,128)
            kernel=@cuda launch=false lane_softmax_kernel!(args...)
        end
        launch=()->kernel(args...;threads=128,blocks)
        launch();actual=Array(output)
        if reference===nothing
            reference=actual
        else
            @assert isapprox(actual,reference;rtol=3f-5,atol=3f-7)
        end
        graph=CUDACore.instantiate(CUDACore.capture() do
            for _ in 1:32
                launch()
            end
        end)
        push!(variants,(;name,kernel,args,output,graph,times=Float64[]))
    end
    for v in variants,_ in 1:5
        CUDACore.launch(v.graph)
    end
    CUDACore.synchronize()
    for _ in 1:samples,i in randperm(rng,length(variants))
        v=variants[i]
        push!(v.times,1e6*(CUDACore.@elapsed CUDACore.launch(v.graph))/32)
    end
    records=[Dict("implementation"=>v.name,"median_us"=>sort(v.times)[cld(samples,2)],
        "min_us"=>minimum(v.times),"samples_us"=>v.times,
        "registers"=>CUDACore.registers(v.kernel),"local_bytes"=>CUDACore.memory(v.kernel).local) for v in variants]
    println((;width,rows,timings=[(r["implementation"],r["median_us"],r["registers"],r["local_bytes"]) for r in records]));flush(stdout)
    Dict("width"=>width,"rows"=>rows,"records"=>records)
end
function main()
    length(ARGS)==1 || error("provide a new output directory")
    out=abspath(only(ARGS));ispath(out)&&error("output already exists");mkpath(out)
    cases=[measure(width,1024) for width in (31,97,257)]
    report=Dict("created_at"=>string(now(UTC)),"device"=>CUDACore.name(device()),"julia"=>string(VERSION),
        "compiler"=>string(CUDACore.compiler_version()),"cases"=>cases,
        "measurement"=>"FP32, same column-contiguous row storage and masks; graph batch of 32 warm kernels; excludes compilation and transfers")
    open(io->TOML.print(io,report),joinpath(out,"results.toml"),"w")
end
main()
