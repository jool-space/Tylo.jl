# julia --project=test/gpu examples/softmax/compare_streaming.jl OUTPUT_DIR
using Tylo,PTX,CUDACore,BFloat16s,Random,TOML,Dates
include("kernel.jl")
include("streaming.jl")
include("baseline.jl")
using .SoftmaxExample: warp_softmax_kernel!
function measure(width,rows;samples=41)
    rng=MersenneTwister(width+rows)
    input=CuArray(randn(rng,Float32,width,rows));mask=CuArray(rand(rng,Float32,width,rows).>0.2f0)
    variants=[];reference=nothing
    for (name,f,tail) in (("scalar_three_pass",baseline_softmax_kernel!,()),
                         ("full_register",warp_softmax_kernel!,(Val(cld(width,32)),)),
                         ("streaming_two_pass",streaming_softmax_kernel!,(Val(4),)))
        output=similar(input);args=(output,input,mask,tail...)
        kernel=@cuda launch=false f(args...)
        launch=()->kernel(args...;threads=128,blocks=cld(rows,4))
        launch();y=Array(output)
        if reference===nothing;reference=y;else;@assert isapprox(y,reference;rtol=3e-5,atol=3e-7);end
        graph=CUDACore.instantiate(CUDACore.capture() do
            for _ in 1:16;launch();end
        end)
        push!(variants,(;name,kernel,args,output,graph,times=Float64[]))
    end
    for v in variants,_ in 1:5;CUDACore.launch(v.graph);end
    CUDACore.synchronize()
    for _ in 1:samples,i in randperm(rng,length(variants))
        v=variants[i];push!(v.times,1e6*(CUDACore.@elapsed CUDACore.launch(v.graph))/16)
    end
    records=[Dict("implementation"=>v.name,"median_us"=>sort(v.times)[cld(samples,2)],
        "samples_us"=>v.times,"registers"=>CUDACore.registers(v.kernel),"local_bytes"=>CUDACore.memory(v.kernel).local) for v in variants]
    println((;width,rows,timings=[(r["implementation"],r["median_us"],r["registers"],r["local_bytes"]) for r in records]));flush(stdout)
    Dict("width"=>width,"rows"=>rows,"records"=>records)
end
function main()
    out=abspath(only(ARGS));ispath(out)&&error("output exists");mkpath(out)
    cases=[measure(n,1024) for n in (31,97,257,1024,4099)]
    report=Dict("created_at"=>string(now(UTC)),"device"=>CUDACore.name(device()),"julia"=>string(VERSION),
        "compiler"=>string(CUDACore.compiler_version()),"cases"=>cases,
        "measurement"=>"same FP32 inputs/masks, interleaved 41 samples of 16-kernel graphs; excludes compilation/preparation/transfers")
    open(io->TOML.print(io,report),joinpath(out,"results.toml"),"w")
end
main()
