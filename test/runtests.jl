# Test runner: `julia --project=test test/runtests.jl [--jobs=N] [--list] [names...]`.
#
#   host/   pure host; runs everywhere.
#   gpu/    needs the CUDA compiler; assembly checks always run, runtime
#           sections run when the live device satisfies the file's
#           `# TEST_TARGET:` banner (see targets.jl).
#   tools/  standalone scripts (sanitizers, evidence), not tests.
#   *_defs.jl files hold definitions shared by several test files.
#
# TYLO_REQUIRE_GPU_RUNTIME=true fails the run when no functional GPU is
# present. TYLO_EVIDENCE=<dir> saves PTX and cubins.
using Tylo, PTX, CUDACore, Test
using ParallelTestRunner
include(joinpath(@__DIR__, "targets.jl"))
using .TestTargets

const SUPPORT = ("setup", "targets")
testsuite = find_tests(@__DIR__)
filter!(p -> !(p.first in SUPPORT) && !startswith(p.first, "tools/") && !endswith(p.first, "_defs"), testsuite)
args = parse_args(ARGS)
default_routing = filter_tests!(testsuite, args)

if args.list === nothing
    gpu_tests = sort!([t for t in keys(testsuite) if startswith(t, "gpu/")])
    toolchain = CUDACore.CUDA_Compiler.is_available()
    functional = CUDACore.functional()
    cap = functional ? CUDACore.capability(CUDACore.device()) : v"0.0"
    if !isempty(gpu_tests)
        if !toolchain
            @warn "CUDA compiler artifacts unavailable; skipping gpu/ tests"
            foreach(t -> delete!(testsuite, t), gpu_tests)
        elseif functional
            @info "GPU runtime available" device=CUDACore.name(CUDACore.device()) capability=cap
        else
            @warn "No functional GPU; gpu/ files run their assembly checks only"
        end
    end
    if toolchain
        for t in gpu_tests
            path = joinpath(@__DIR__, t * ".jl")
            predicates = runtime_predicates(path)
            ok = TestTargets.runtime_supported(path, cap)
            println(rpad(t, 30), " runtime ", rpad(describe_predicates(predicates), 14),
                    ok ? "executes" : "skipped")
        end
    end
    if get(ENV, "TYLO_REQUIRE_GPU_RUNTIME", "false") == "true" && default_routing
        functional || error("TYLO_REQUIRE_GPU_RUNTIME is set, but CUDACore has no functional GPU")
    end
end

# Each test file runs in its own module; names every file may use are
# imported here rather than in each file.
init_code = quote
    using Tylo, PTX, CUDACore, BFloat16s, Random, Test, SHA, TOML
    using Tylo.Layouts: @Layout, Layout, Swizzle, compose, coordinate, cosize, shape, static
    include($(joinpath(@__DIR__, "setup.jl")))
end
runtests(Tylo, args; testsuite, init_code)
