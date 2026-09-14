using Tylo, PTX, CUDACore, BFloat16s, Random, Test
CUDACore.functional() && CUDACore.capability(device()) >= v"8.0" ||
    error("GEMM sanitizer tests require a CUDA device with CC >= 8.0")
push!(ARGS,"--runtime-only")
include("../fixtures.jl")
include("gemm.jl")

include("tma.jl")
include("wgmma.jl")
include("packing.jl")
include("wgmma_fragments.jl")

include("rows.jl")
include("arrayops.jl")

include("softmax.jl")

include("boundaries.jl")

include("online.jl")

include("operand_a.jl")

include("streaming_attention.jl")

println("Tylo tile sanitizer workload completed")
