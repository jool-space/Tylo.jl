using Tylo, PTX, CUDACore, BFloat16s, Random, Test
include("codegen.jl")
include("fragments.jl")
include("tmem.jl")
include("gemm.jl")
include("tma.jl")
include("wgmma.jl")
if "--attention" in ARGS
    include("../../examples/flash_attention/comparison.jl")
end

include("rows.jl")

include("softmax.jl")

include("boundaries.jl")

include("online.jl")

include("operand_a.jl")

include("streaming_attention.jl")
