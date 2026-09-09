using Tylo, PTX, CUDACore, BFloat16s, Random, Test
CUDACore.functional() && CUDACore.capability(device()) >= v"8.0" ||
    error("GEMM sanitizer tests require a CUDA device with CC >= 8.0")
push!(ARGS,"--runtime-only")
include("gemm.jl")

include("tma.jl")
include("wgmma.jl")
println("Tylo tile sanitizer workload completed")
