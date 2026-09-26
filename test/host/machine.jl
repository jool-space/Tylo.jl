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
    @test occursin("unchecked_div",Tylo._call_hint("gpu_report_exception",["k.jl:1 (f) → int.jl:295 (div)"]))
    @test unchecked_div(Int32(-7),Int32(2)) === Int32(-3) && unchecked_rem(Int32(-7),Int32(2)) === Int32(-1)
    @test unchecked_div(UInt32(7),UInt32(2)) === UInt32(3) && unchecked_rem(UInt32(7),UInt32(2)) === UInt32(1)
end

# ptxas's performance warnings are informational: the code runs, slower. A
# kernel whose setmaxnreg ptxas ignored keeps every warp at the launch's
# budget, and the roles meant to receive more registers spill.
@testset "ptxas performance warnings" begin
    log = """
    ptxas info    : (C7508) Potential Performance Loss: 'setmaxnreg' ignored; unable to determine register count at entry.
    ptxas info    : Used 128 registers, used 16 barriers, 144 bytes cumulative stack size
        144 bytes stack frame, 172 bytes spill stores, 176 bytes spill loads
    """
    code = Tylo.MachineCode("k","sm_100a","",UInt8[],"",log,() -> "")
    findings = audit(code;spills=false)
    @test only(findings).kind === :ptxas
    @test occursin("C7508",only(findings).message) && occursin("maxregs",only(findings).hint)
    @test isempty(audit(code;spills=false,ptxas=false))
    @test Tylo._stack_frame(code) == 144 && Tylo.registers(code) == 128
end
