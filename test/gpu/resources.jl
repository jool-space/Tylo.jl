# Offline resource evidence for saved PTX/cubins. Does not load a kernel.
using CUDACore, TOML
length(ARGS) == 1 || error("usage: resources.jl EVIDENCE_DIRECTORY")
root = abspath(only(ARGS))
records = Dict{String,Any}[]
for file in sort(readdir(root;join=true))
    endswith(file,".ptx") || continue
    name = splitext(basename(file))[1]
    target = match(r"\.target\s+(\w+)",read(file,String)).captures[1]
    cubin = joinpath(root,name*".cubin")
    rebuilt = joinpath(root,name*".reassembled.cubin")
    io = IOBuffer()
    run(pipeline(`$(CUDACore.CUDA_Compiler.ptxas()) -v -arch=$target $file -o $rebuilt`;
                 stdout=io,stderr=io))
    log = String(take!(io))
    write(joinpath(root,name*".ptxas.log"),log)
    sass = read(`$(CUDACore.CUDA_Compiler.nvdisasm()) -c $cubin`,String)
    write(joinpath(root,name*".sass"),sass)
    # The entry precedes exception reporting helpers in these generated cubins.
    entry = split(split(sass,"//--------------------- .text.";limit=2)[2],
                  "//---------------------";limit=2)[1]
    instructions = filter(l -> occursin(r"^\s*/\*[0-9a-f]+\*/",l),split(entry,'\n'))
    resources = [m.match for m in eachmatch(
        r"\d+ bytes stack frame, \d+ bytes spill stores, \d+ bytes spill loads|Used \d+ registers[^\n]*",log)]
    record = Dict("name"=>name,"target"=>target,"resources"=>resources,
                  "instructions"=>length(instructions),
                  "local_loads"=>count(l->occursin(r"\bLDL\b",l),instructions),
                  "local_stores"=>count(l->occursin(r"\bSTL\b",l),instructions))
    push!(records,record)
    println(name,": ",join(resources,"; "))
end
open(joinpath(root,"resources.toml"),"w") do io
    TOML.print(io,Dict("kernels"=>records))
end
