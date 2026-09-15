# Loaded into every test worker when Julia runs with `--code-coverage`.
#
# Coverage is applied during type inference: a frame whose file is tracked is
# instrumented and loses its effect-free bit, which stops constant folding and
# concrete evaluation. Device code compiled that way is not the code users get
# (validation checks and layout arithmetic turn into real device code with
# exception calls), so the assembly gates would fail on an artifact nobody
# runs. GPUCompiler already generates device code with coverage disabled; this
# makes its inference consistent with that. Frames built by the GPU
# interpreter are marked uninstrumented and effect-free again; host inference
# is untouched. Device coverage keeps GPUCompiler's meaning ("this line was
# compiled") by recording the lines of every compiled statement, since there
# are no coverage statements left to read.
#
# This belongs in GPUCompiler; carry it here until it lands upstream.
if Base.JLOptions().code_coverage != 0
    @eval CUDACore.GPUCompiler begin
        const _CacheMode = VERSION >= v"1.11" ? UInt8 : Symbol
        function CC.InferenceState(result::CC.InferenceResult, cache_mode::_CacheMode,
                                   interp::GPUInterpreter)
            frame = invoke(CC.InferenceState,
                           Tuple{CC.InferenceResult, _CacheMode, CC.AbstractInterpreter},
                           result, cache_mode, interp)
            if frame !== nothing && frame.insert_coverage
                frame.insert_coverage = false
                frame.ipo_effects = CC.Effects(frame.ipo_effects; effect_free = CC.ALWAYS_TRUE)
            end
            return frame
        end

        function _statement_location(src::CodeInfo, pc::Int)
            @static if VERSION >= v"1.12"
                scopes = Base.IRShow.buildLineInfoNode(src.debuginfo, nothing, pc)
                isempty(scopes) && return nothing
                loc = scopes[end]   # innermost scope, i.e. the inlined callee
                return (loc.file, loc.line)
            else
                idx = src.codelocs[pc]
                idx == 0 && return nothing
                loc = src.linetable[idx]::Core.LineInfoNode
                return (loc.file, loc.line)
            end
        end

        const _coverage_recorded = Set{MethodInstance}()
        function record_coverage(mi::MethodInstance, src::CodeInfo)
            mi in _coverage_recorded && return
            push!(_coverage_recorded, mi)
            tracked = false
            for pc in 1:length(src.code)
                loc = _statement_location(src, pc)
                loc === nothing && continue
                file, line = loc
                (file isa Symbol && Base.is_file_tracked(file)) || continue
                tracked = true
                coverage_visit_line(file, line)
            end
            if tracked
                def = mi.def
                def isa Method && coverage_visit_line(def.file, def.line)
            end
            return
        end
    end
    # CUDACore drives the compiler in the world captured at its initialization;
    # the methods above must be visible there.
    CUDACore._initialization_world[] = Base.get_world_counter()
end
