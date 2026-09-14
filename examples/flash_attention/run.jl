# Standalone attention comparison, one process, no test runner:
#
#     julia --project=test examples/flash_attention/run.jl            # code generation checks
#     julia --project=test examples/flash_attention/run.jl --bench    # plus paired timings (B200/B300)
#
# TYLO_EVIDENCE=<dir> saves the PTX and cubins of every compiled variant.
using Tylo, PTX, CUDACore, BFloat16s, Random, Test
using Tylo.Layouts: @Layout
include(joinpath(@__DIR__,"..","..","test","targets.jl"))
using .TestTargets
include(joinpath(@__DIR__,"..","..","test","setup.jl"))
if CUDACore.functional()
    @info "GPU runtime available" device=CUDACore.name(CUDACore.device()) capability=CUDACore.capability(CUDACore.device())
else
    @warn "No functional GPU; code generation checks only"
end
include(joinpath(@__DIR__,"comparison.jl"))
