# Tracing calls through `code_llvm`'s inlining comments, on a synthetic
# kernel: the frames open with ┌, close with └, and a line of │ updates the
# frame at its depth.
@testset "Call tracing through inlining comments" begin
    ir = """
    ;  @ /work/kernel.jl:3 within `outer`
    define ptx_kernel void @_Z5outer() {
    top:
    ;  @ /work/kernel.jl:4 within `outer`
    ; ┌ @ range.jl:22 within `Colon`
    ; │┌ @ range.jl:367 within `steprange_last`
       %1 = call fastcc i64 @julia_overflow_case_123(i64 %0)
    ; └└
    ;  @ /work/kernel.jl:5 within `outer`
    ; ┌ @ /work/lib.jl:10 within `helper`
    ; │┌ @ math.jl:645 within `sqrt` @ math.jl:640
       call fastcc void @julia_throw_complex_domainerror_7()
       %2 = call float @llvm.sqrt.f32(float %1)
    ; └└
      ret void
    }
    define internal void @julia_overflow_case_123(i64 %0) {
      call void @gpu_report_exception()
      ret void
    }
    """
    traced = Tylo._trace_calls(ir)
    @test traced["julia_overflow_case"] == ["kernel.jl:4 (outer) → range.jl:367 (steprange_last)"]
    @test traced["julia_throw_complex_domainerror"] == ["kernel.jl:5 (outer) → lib.jl:10 (helper) → math.jl:640 (sqrt)"]
    @test !haskey(traced,"gpu_report_exception")          # only the kernel's own calls
    @test !any(startswith("llvm."),keys(traced))
    @test occursin("StepRange",Tylo._call_hint("julia_overflow_case",traced["julia_overflow_case"]))
    @test occursin("sqrt_llvm",Tylo._call_hint("julia_throw_complex_domainerror",String[]))
    @test occursin("sdiv_int",Tylo._call_hint("gpu_report_exception",["k.jl:1 (f) → int.jl:295 (div)"]))
end
