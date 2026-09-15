# Loaded into every test worker. Fixtures, device predicates, kernel
# compilation helpers.
using CUDACore.GPUCompiler: CompilerJob, methodinstance
isdefined(@__MODULE__, :TestTargets) || include(joinpath(@__DIR__, "targets.jl"))
using .TestTargets
include(joinpath(@__DIR__, "coverage.jl"))

# Explicit ownership patterns reused across host and device tests.
@inline local_fragment(data::NTuple{N}) where N = Fragment(data,Tylo.Layouts.LocalOwnership{N,2}())
@inline striped_fragment(data::NTuple{N}) where N = Fragment(data,Tylo.Layouts.StripedOwnership{N,2}())
# Row-distributed fragments for the reduction tests, keyed by ownership kind.
row_fragment(::Val{:local},data) = local_fragment(data)
row_fragment(::Val{:warp},data) = striped_fragment(data)
row_fragment(a::MMAAtom,data) = Fragment(data,operand_layout(a,Accumulator()))
@inline row_fragment(p::TiledMMA,data) = Fragment(data,operand_layout(p,Accumulator()))
row_words(f) = f.data
# Independent reference coordinates; do not use the ownership being tested.
function reference_row(kind,n,t,e)
    kind isa Val{:local} && return t
    kind isa Val{:warp} && return t÷32
    rm = kind isa MMAAtom ? 1 : typeof(kind).parameters[3][1]
    atom,word=e÷4,e%4
    16rm*(t÷32)+(t%32)÷4+8*(word÷2)+16*(atom%rm)
end


# Device predicates. A file's banner gates its main runtime sections through
# `runtime_supported(@__FILE__)`; sections needing a different device use
# the explicit capability helpers.
const _DEVICE_CAPABILITY = Ref{Union{Nothing,VersionNumber}}(nothing)
function device_capability()
    cap = _DEVICE_CAPABILITY[]
    cap === nothing || return cap
    cap = CUDACore.functional() ? CUDACore.capability(CUDACore.device()) : v"0.0"
    _DEVICE_CAPABILITY[] = cap
end
runtime_supported(file::AbstractString) = TestTargets.runtime_supported(file, device_capability())
capability_at_least(v::VersionNumber) = device_capability() >= v
capability_is(v::VersionNumber) = device_capability() == v
capability_major(major::Integer) = device_capability().major == major

# Masked row softmax reference in Float64; rows with no valid entry stay zero.
function softmax_reference(input,mask)
    out=zeros(Float64,size(input))
    for row in axes(input,2)
        valid=findall(mask[:,row])
        isempty(valid) && continue
        x=Float64.(input[valid,row]); weights=exp.(x .- maximum(x))
        out[valid,row]=weights ./ sum(weights)
    end
    out
end

# Compile a kernel for an explicit target without a device, as `ptxas`
# evidence. Kernels compiled this way are never loaded.
function compile_kernel(f, tt; arch=CUDACore.SMVersion(10,0,:arch), threads=128)
    config = CUDACore.compiler_config(nothing;kernel=true,arch,minthreads=threads)
    job = CompilerJob(methodinstance(typeof(f),tt),config)
    image = CUDACore.invoke_frozen(CUDACore.compile,job).image
    io = IOBuffer()
    CUDACore.invoke_frozen(CUDACore.GPUCompiler.code_native,io,job;dump_module=true)
    (;image,ptx=String(take!(io)))
end

function save_code(name, code)
    haskey(ENV,"TYLO_EVIDENCE") || return
    # Type names spell BFloat16 the same way on every Julia version.
    name = replace(name, "Core.BFloat16"=>"BFloat16")
    dir = ENV["TYLO_EVIDENCE"]
    mkpath(dir)
    write(joinpath(dir,name*".ptx"),code.ptx)
    write(joinpath(dir,name*".cubin"),code.image)
end

function entry_body(ptx)
    split(split(ptx,".visible .entry";limit=2)[2],"// -- End function";limit=2)[1]
end

# Extract a named entry's machine-code bytes from a CUDA ELF64 little-endian
# cubin. Debug/source metadata are deliberately outside this comparison.
function kernel_text(image, prefix=".text._Z11fab_kernel_")
    @assert image[1:6] == UInt8[0x7f,0x45,0x4c,0x46,2,1]
    u16(off) = Int(ltoh(reinterpret(UInt16,image[off+1:off+2])[1]))
    u32(off) = Int(ltoh(reinterpret(UInt32,image[off+1:off+4])[1]))
    u64(off) = Int(ltoh(reinterpret(UInt64,image[off+1:off+8])[1]))
    table = u64(0x28)
    stride, count, names_idx = u16(0x3a), u16(0x3c), u16(0x3e)
    names = u64(table+names_idx*stride+0x18)
    for idx in 0:count-1
        header = table+idx*stride
        start = names+u32(header)+1
        stop = findnext(iszero,image,start)
        name = String(image[start:stop-1])
        if startswith(name,prefix)
            offset, bytes = u64(header+0x18), u64(header+0x20)
            return image[offset+1:offset+bytes]
        end
    end
    error("kernel code section not found: $prefix")
end
