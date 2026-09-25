# TEST_TARGET: cc>=8.0
# `machine_code` and `audit`: what they find, and where they trace it.
include("wgmma_defs.jl")

function clean_scale!(y,x)
    i = threadIdx().x
    @inbounds y[i] = 2f0*x[i]
    nothing
end
# Line numbers matter: the audit traces calls to them.
function stepped_roots!(y,x,step)
    for i in threadIdx().x:step:length(x)            # StepRange: an out-of-line helper
        @inbounds y[i] = sqrt(x[i])                   # DomainError check
    end
    nothing
end
const STEPPED_LINE = 12
function indexed_tuple!(y,x,i)
    values = ntuple(j -> @inbounds(x[j]),Val(16))
    @inbounds y[1] = values[i]                        # a tuple indexed at run time
    nothing
end
# A WGMMA kernel with a call: ptxas serializes every WGMMA instruction in it.
function serialized_wgmma!(out,p::WGMMAAtom,k0::Int32,n::Int32)
    wgmma_operand_probe!(out,p,k0)
    for i in threadIdx().x:n:Int32(64)                # StepRange with a runtime step
        @inbounds out[i] += 1f0
    end
    nothing
end
function wide_copy!(y,x)
    t = Int32(threadIdx().x)-Int32(1)
    @inbounds tile, out = GlobalTile(x,16), GlobalTile(y,16)
    @inbounds f = load_fragment(distribute((8,32),32;vector=8,axis=1),window(tile,(0,0),(8,32)),t)
    @inbounds store!(window(out,(0,0),(8,32)),f,t)
    nothing
end

@testset "machine_code and audit" begin
    vector = CuDeviceVector{Float32,1}
    code = compile_kernel(clean_scale!,Tuple{vector,vector};arch=CUDACore.SMVersion(12,1,:arch),threads=32)
    @test code isa Tylo.MachineCode && code.arch == "sm_121a"
    @test isempty(audit(code))
    @test Tylo.registers(code) > 0 && Tylo.sass_instructions(code) > 0
    @test Tylo.kernel_calls(code) == String[] && Tylo.spill_bytes(code) == (0,0)
    @test occursin("audit: clean",sprint(show,MIME("text/plain"),code))

    # Calls are traced to the lines that brought them, with a way out.
    code = compile_kernel(stepped_roots!,Tuple{vector,vector,Int32};arch=CUDACore.SMVersion(12,1,:arch),threads=32)
    findings = audit(code)
    calls = filter(f -> f.kind === :call,findings)
    @test !isempty(calls)
    here(line) = "machine.jl:$line (stepped_roots!)"
    step = only(filter(f -> occursin("overflow_case",f.message),calls))
    @test only(step.sources) == "$(here(STEPPED_LINE)) → range.jl:367 (steprange_last)"
    @test occursin("StepRange",step.hint)
    root = only(filter(f -> occursin("domainerror",f.message),calls))
    @test startswith(only(root.sources),here(STEPPED_LINE+1))
    @test occursin("sqrt_llvm",root.hint)
    @test all(f -> all(s -> startswith(s,"machine.jl:"),f.sources),calls)
    shown = sprint(show,MIME("text/plain"),code)
    @test occursin("✗ calls",shown) && occursin("hint:",shown)
    @test isempty(audit(code;calls=false,local_memory=false))

    code = compile_kernel(indexed_tuple!,Tuple{vector,vector,Int32};arch=CUDACore.SMVersion(12,1,:arch),threads=32)
    local_ = only(filter(f -> f.kind === :local_memory,audit(code)))
    @test Tylo.local_bytes(code) == 64 && occursin("runtime index",local_.hint)

    # ptxas serializes WGMMA silently; the audit reports it with the reason.
    p = WGMMAAtom(BFloat16,8,64)
    tt = Tuple{CuDeviceMatrix{Float32,1},typeof(p),Int32,Int32}
    code = compile_kernel(serialized_wgmma!,tt;arch=CUDACore.SMVersion(9,0,:arch))
    w = Tylo.wgmma_groups(code)
    @test w.instructions == 4 && w.fences == 1 && w.arrives == 4
    serialized = only(filter(f -> f.kind === :wgmma_serialized,audit(code)))
    @test occursin("C75",serialized.message)
    @test any(f -> f.kind === :call,audit(code))
    code = compile_kernel(wgmma_operand_probe!,Tuple{CuDeviceMatrix{Float32,1},typeof(p),Int32};arch=CUDACore.SMVersion(9,0,:arch))
    @test Tylo.wgmma_groups(code) == (; instructions=4, fences=1, arrives=1, waits=1)
    @test isempty(audit(code))

    # The widths the memory instructions became.
    matrix = CuDeviceMatrix{BFloat16,1}
    code = compile_kernel(wide_copy!,Tuple{matrix,matrix};arch=CUDACore.SMVersion(12,1,:arch),threads=32)
    memory = Dict(Tylo.memory_instructions(code))
    @test get(memory,"LDG.E.128",0) == 1 && get(memory,"STG.E.128",0) == 1
    @test isempty(audit(code))
end

if runtime_supported(@__FILE__)
@testset "@machine_code for host arguments" begin
    x = CuArray(rand(Float32,32)); y = similar(x)
    code = @machine_code threads=32 clean_scale!(y,x)
    @test isempty(audit(code))
    @test code.arch == "sm_$(device_capability().major)$(device_capability().minor)" ||
          startswith(code.arch,"sm_$(device_capability().major)$(device_capability().minor)")
    @test isempty(audit(@machine_code clean_scale!(y,x)))
end
end

const UNALIGNED_LINE = 106
function unaligned_sum!(y,x)
    scratch = CuStaticSharedArray(Float32,32)
    p = reinterpret(Core.LLVMPtr{Float32,3},pointer(scratch))
    i = threadIdx().x
    unsafe_store!(p,@inbounds(x[i]),i)                 # alignment 1 unless given one
    sync_threads()
    @inbounds y[i] = unsafe_load(p,i%Int32(32)+Int32(1),Val(4))
    nothing
end
@testset "Unaligned accesses" begin
    vector = CuDeviceVector{Float32,1}
    code = compile_kernel(unaligned_sum!,Tuple{vector,vector};arch=CUDACore.SMVersion(12,1,:arch),threads=32)
    split = only(filter(f -> f.kind === :unaligned,audit(code)))
    @test occursin("1 loads and stores",split.message)
    @test only(split.sources) == "machine.jl:$(UNALIGNED_LINE) (unaligned_sum!) → base.jl:39 (macro expansion)" ||
          startswith(only(split.sources),"machine.jl:$(UNALIGNED_LINE) (unaligned_sum!)")
    @test occursin("Val(sizeof(T))",split.hint)
end

# A closure local sharing a name with the kernel's variable: Julia boxes it,
# and machine_code names it instead of the compiler's invalid-IR report.
function boxed_kernel!(y)
    @inline scaled(j) = (v = 2f0*j; v)
    v = 0f0
    for i in 1:4
        v += scaled(Float32(i))
    end
    @inbounds y[1] = v
    nothing
end
@testset "Boxed variables" begin
    vector = CuDeviceVector{Float32,1}
    @test Tylo.boxed_variables(boxed_kernel!,Tuple{vector}) == [:v]
    @test Tylo.boxed_variables(clean_scale!,Tuple{vector,vector}) == Symbol[]
    err = try
        compile_kernel(boxed_kernel!,Tuple{vector};arch=CUDACore.SMVersion(12,1,:arch),threads=32)
        nothing
    catch e
        e
    end
    @test err isa ErrorException && occursin("`v` is captured by a closure",err.msg)
end
