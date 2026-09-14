using Tylo, PTX, CUDACore, BFloat16s, Random, Test
include("../fixtures.jl")


Base.JLOptions().code_coverage == 0 || error(
    "GPU structural tests require --code-coverage=none; collect host coverage separately.")

CUDACore.CUDA_Compiler.is_available() || error(
    "CUDA compiler artifacts are unavailable. In this test environment, run " *
    "CUDACore.set_runtime_version!(v\"13.3\"; local_toolkit=false), then restart Julia.")
if get(ENV, "TYLO_REQUIRE_GPU_RUNTIME", "false") == "true"
    CUDACore.functional() || error(
        "TYLO_REQUIRE_GPU_RUNTIME is set, but CUDACore has no functional GPU.")
    @info "GPU runtime required" device=CUDACore.name(CUDACore.device()) capability=CUDACore.capability(CUDACore.device())
end
println("CUDA compiler: ", CUDACore.compiler_version())

include("codegen.jl")
include("tuples.jl")
include("layouts.jl")
include("fragments.jl")
include("tmem.jl")
include("tmem_views.jl")
include("gemm.jl")
include("atoms.jl")
include("tma.jl")
include("wgmma.jl")
include("packing.jl")
include("wgmma_fragments.jl")
if "--attention" in ARGS
    include("../../examples/flash_attention/comparison.jl")
end

include("rows.jl")
include("arrayops.jl")

include("softmax.jl")

include("boundaries.jl")

include("online.jl")

include("operand_a.jl")

include("streaming_attention.jl")

snapshot_report()
