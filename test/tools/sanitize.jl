# Sanitizer workload: every runtime GPU test in one process, for
# compute-sanitizer. Run from the repository root:
#   compute-sanitizer --tool racecheck julia --project=test test/tools/sanitize.jl
using Tylo, PTX, CUDACore, BFloat16s, Random, Test, SHA, TOML
using Tylo.Layouts: @Layout, Layout, Swizzle, compose, coordinate, cosize, shape, static
include(joinpath(@__DIR__, "..", "setup.jl"))
capability_at_least(v"8.0") || error("the sanitizer workload requires a CUDA device with CC >= 8.0")
for name in ("gemm","tma","wgmma","packing","wgmma_fragments","rows","arrayops",
             "softmax","boundaries","online","operand_a","streaming_attention")
    include(joinpath(@__DIR__, "..", "gpu", name * ".jl"))
end
println("Tylo tile sanitizer workload completed")
