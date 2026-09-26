module CUDACoreExt
using Tylo, PTX, CUDACore, Adapt
using Tylo: @rtuple, Fragment, BFloat16, TMATile
using CUDACore: @device_override
using PTX: @ptx_str

# Device-only implementations of host-callable generics. The host keeps the
# generic method (or raises); kernels compiled through CUDACore's method
# table use these.
# FP32 to BF16/FP16 converts pairs with one `cvt` each.
@device_override @inline function Tylo._convert_fragment(::Type{T}, f::Fragment{Float32,N}) where {T<:Union{BFloat16,Float16},N}
    iseven(N) || return Tylo.map(Base.Fix1(Tylo.convert_element,T),f)
    words = @rtuple(i -> Tylo._pack_mma_pair(T,f.data[2i-1],f.data[2i]), 1:N÷2)
    Tylo._from_words(T,words,Tylo.Layouts.layout(f))
end
# FP32 to FP8 converts pairs with one `cvt` each where the target has it
# (sm_89 on); a pair's 16 bits are half of a fragment word, so the words
# form without splitting pairs into bytes.
@inline _fp8_pair(::Type{Tylo.Float8_E4M3FN},lo,hi) = reinterpret(UInt16,ptx"cvt.rn.satfinite.e4m3x2.f32"(hi,lo))
@inline _fp8_pair(::Type{Tylo.Float8_E5M2},lo,hi) = reinterpret(UInt16,ptx"cvt.rn.satfinite.e5m2x2.f32"(hi,lo))
@inline _has_fp8() = (c = CUDACore.compute_capability(); c.major > 8 || (c.major == 8 && c.minor >= 9))
@device_override @inline function Tylo._convert_fragment(::Type{T}, f::Fragment{Float32,N}) where {T<:Union{Tylo.Float8_E4M3FN,Tylo.Float8_E5M2},N}
    (N % 4 == 0 && _has_fp8()) ||
        return Fragment(Tuple(Tylo.Microfloats.cvt(Tylo.Microfloats.SVector{N,T},f.data,RoundNearest,Tylo.SAT)),Tylo.Layouts.layout(f))
    pairs = @rtuple(i -> _fp8_pair(T,f.data[2i-1],f.data[2i]), 1:N÷2)
    words = @rtuple(i -> UInt32(pairs[2i-1]) | (UInt32(pairs[2i]) << 16), 1:N÷4)
    Tylo._from_words(T,words,Tylo.Layouts.layout(f))
end
@device_override @inline Tylo._smem_address(p::Core.LLVMPtr{T,3}) where T = PTX.smem_addr_u32(p)
@device_override @inline Tylo._shuffle_idx(x::UInt32,lane::Int32) =
    ptx"shfl.sync.idx.b32"(x,lane % UInt32,UInt32(0x1f),0xffffffff % UInt32)
@device_override @inline Tylo._ex2(x::Float32) = ccall("llvm.nvvm.ex2.approx.ftz.f",llvmcall,Float32,(Float32,),x)
@device_override @inline Tylo._warp_reduce(op::F,x,::Val{W}) where {F,W} = PTX.Warps.warp_reduce(op,x,Val(W))
# stmatrix exists from sm_90; the compile target decides, folded at compile time.
@device_override @inline Tylo._stmatrix_available() = CUDACore.compute_capability().major >= 9
@device_override @inline function Tylo._shuffle_xor(op::F,x::T,::Val{Offset}) where {F,T,Offset}
    partner = ptx"shfl.sync.bfly.b32"(reinterpret(UInt32,x),UInt32(Offset),UInt32(0x1f),0xffffffff % UInt32)
    op(x,reinterpret(T,partner))
end

# Descriptor element types name the bit width; FP8, INT8 and other unlisted
# types move as unsigned words of their size.
_tma_dtype(::Type{T}) where T = T === Tylo.BFloat16 ? :bf16 : T === Float16 ? :f16 : T === Float32 ? :f32 :
    sizeof(T) == 1 ? :u8 : sizeof(T) == 2 ? :u16 : :u32
function Tylo.prepare_tma(p::TMATile{T,S,A,W},array::CuArray{T,N};bounds=size(array)) where {T,S,A,W,N}
    N in (2,3) || throw(ArgumentError("TMA binds a matrix or a batch of matrices"))
    length(bounds) == N && all(map((b,s) -> 0 < b <= s,bounds,size(array))) || throw(ArgumentError("bounds exceed the array or are empty"))
    UInt(pointer(array)) % 16 == 0 && (size(array,1)*sizeof(T)) % 16 == 0 ||
        throw(ArgumentError("TMA source pointer and column stride must be 16-byte aligned"))
    strides = N == 2 ? (sizeof(T)*size(array,1),) : (sizeof(T)*size(array,1),sizeof(T)*size(array,1)*size(array,2))
    box = N == 2 ? Tylo._box(p) : (Tylo._box(p)...,1)     # one stripe per copy
    tmap = GC.@preserve array PTX.tensor_map_encode_tiled(_tma_dtype(T),UInt(pointer(array)),
        map(Int,bounds),strides,box;swizzle=Symbol(:B,W))
    uploaded = PTX.upload_tma_descriptor(tmap)
    Tylo.HostTMA(p,array,uploaded.ptr,uploaded.blob,Val(N))
end
function Adapt.adapt_structure(to::CUDACore.KernelAdaptor,b::Tylo.HostTMA{P,A,D,B,R}) where {P,A,D,B,R}
    # Register both hidden allocations with CUDA's stream/lifetime tracking.
    # Their device views need not become kernel parameters, but adaptation's
    # host-side bookkeeping is still required for cross-stream launches.
    Adapt.adapt(to,b.source)
    Adapt.adapt(to,b.storage)
    Tylo.TMABinding(b.plan,b.descriptor,Val(R))
end

# Machine code: compile as `@cuda` would, then assemble with ptxas in
# verbose mode, whose report (registers, spills, WGMMA serialization) CUDACore
# keeps only at debug level, and disassemble the cubin.
function Tylo.machine_code(@nospecialize(f),@nospecialize(types::Type{<:Tuple});arch=nothing,threads=nothing,maxregs=nothing)
    options = Pair{Symbol,Any}[:kernel => true]
    arch === nothing || push!(options,:arch => arch)
    threads === nothing || push!(options,:minthreads => threads)
    maxregs === nothing || push!(options,:maxregs => maxregs)
    config = CUDACore.compiler_config(arch === nothing ? CUDACore.device() : nothing;options...)
    job = CUDACore.GPUCompiler.CompilerJob(CUDACore.GPUCompiler.methodinstance(typeof(f),types),config)
    # Compiled exactly as CUDACore compiles it: stripping the debug
    # information changes the code LLVM generates (registers, schedules).
    asm, _ = try
        CUDACore.GPUCompiler.JuliaContext() do ctx
            CUDACore.invoke_frozen(CUDACore.GPUCompiler.compile,:asm,job)
        end
    catch err
        throw(Tylo._compile_failure(f,types,err))
    end
    source = String(asm)
    # The PTX kept for reading and searching, without the debug directives:
    # their file names and strings would match searches for instructions.
    ptx = replace(source,r"^[ \t]*\.section[ \t]+\.debug_\w+\s*\{.*?\}[ \t]*\n"ms => "",
                         r"^[ \t]*\.(?:file|loc)\b[^\n]*\n"m => "")
    entry = something(match(r"\.visible \.entry\s+([\w$]+)",ptx),(;captures=["?"])).captures[1]
    target = match(r"^\.target\s+(sm_\w+)"m,ptx)
    target === nothing && error("no target in the PTX")
    gpu = String(target[1])
    image, log = mktempdir() do dir
        input, output = joinpath(dir,"kernel.ptx"), joinpath(dir,"kernel.cubin")
        write(input,source)
        buffer = IOBuffer()
        # CUDACore's ptxas options at this debug level.
        level = Base.JLOptions().debug_level
        debug = level == 1 ? ["--generate-line-info"] : level >= 2 ? ["--device-debug"] : String[]
        process = run(pipeline(ignorestatus(`$(CUDACore.CUDA_Compiler.ptxas()) $debug --verbose --gpu-name $gpu --output-file $output $input`);
                               stdout=buffer,stderr=buffer))
        log = String(take!(buffer))
        success(process) || error("ptxas failed for $entry:\n$log")
        read(output), log
    end
    sass = mktempdir() do dir
        path = joinpath(dir,"kernel.cubin")
        write(path,image)
        read(`$(CUDACore.CUDA_Compiler.nvdisasm()) $path`,String)
    end
    cached = Ref{Union{Nothing,String}}(nothing)
    annotated = () -> something(cached[],begin
        io = IOBuffer()
        CUDACore.invoke_frozen(CUDACore.GPUCompiler.code_llvm,io,job;debuginfo=:source,dump_module=true)
        cached[] = String(take!(io))
    end)
    Tylo.MachineCode(String(entry),gpu,ptx,image,sass,log,annotated)
end
Tylo._machine_code_call(f,args::Tuple;kwargs...) =
    Tylo.machine_code(f,Tuple{map(a -> Core.Typeof(CUDACore.cudaconvert(a)),args)...};kwargs...)

# The machine code of a kernel compiled by `@cuda launch=false`. The kernel
# does not remember its compile options, so they are inferred from the
# loaded function, or taken from `kwargs`, and the result is checked against
# it: an audit of other code than the kernel runs would mislead.
function Tylo.machine_code(k::CUDACore.HostKernel{F,TT};kwargs...) where {F,TT}
    attributes = CUDACore.attributes(k.fun)
    registers = Int(attributes[CUDACore.FUNC_ATTRIBUTE_NUM_REGS])
    local_bytes = Int(attributes[CUDACore.FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES])
    limit = Int(attributes[CUDACore.FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK])
    candidates = isempty(kwargs) ? [(;),(;threads=limit),(;threads=limit,maxregs=registers)] : [(;kwargs...)]
    for options in candidates
        code = Tylo.machine_code(k.f,TT;options...)
        Tylo.registers(code) == registers && Tylo._stack_frame(code) == local_bytes && return code
    end
    throw(ArgumentError("the loaded kernel uses $registers registers and $local_bytes bytes of local memory, which recompiling $(isempty(kwargs) ? "with the options it may have been launched with" : "with the given options") does not reproduce; pass them: machine_code(kernel; threads=..., maxregs=...)"))
end

function Tylo.tune(prepare,candidates;samples::Integer=15,audit=true,target::Real=2e-4)
    trials = Tylo.Trial[]
    timed = []
    for c in candidates
        prepared = try
            prepare(c)
        catch err
            err isa ArgumentError || rethrow()
            push!(trials,Tylo.Trial(c,:invalid,sprint(showerror,err),NaN,NaN,0))
            continue
        end
        kernel, launch = prepared isa Tuple ? prepared : (nothing,prepared)
        registers = kernel === nothing ? 0 : CUDACore.registers(kernel)
        if kernel !== nothing && audit !== false
            findings = audit === true ? Tylo.audit(Tylo.machine_code(kernel)) : Tylo.audit(Tylo.machine_code(kernel);audit...)
            if !isempty(findings)
                push!(trials,Tylo.Trial(c,:audited,join(unique(string.(getfield.(findings,:kind))),", "),NaN,NaN,registers))
                continue
            end
        end
        launch(); CUDACore.synchronize()
        once = CUDACore.@elapsed launch()
        repeats = clamp(ceil(Int,target/max(once,1e-7)),1,1000)
        push!(timed,(; c, launch, repeats, registers, times=Float64[]))
    end
    # Rotating the order each round spreads clock and thermal drift over all.
    for round in 1:samples, i in 1:length(timed)
        x = timed[mod1(i+round,length(timed))]
        t = CUDACore.@elapsed for _ in 1:x.repeats
            x.launch()
        end
        push!(x.times,t/x.repeats)
    end
    for x in timed
        times = sort(x.times)
        push!(trials,Tylo.Trial(x.c,:timed,"",times[cld(length(times),2)],first(times),x.registers))
    end
    Tylo.Tuning(Tylo._ranked(trials))
end

# FP32 scores to an FP8 atom's register A: within each quad, thread q takes
# its four consecutive columns 4q..4q+3 of each 16 from threads 2(q%2) and
# 2(q%2)+1, which hold them as pairs in 8-column block q÷2 (and q÷2+2).
@generated function Tylo.pack_operand_a(atom::A,f::Fragment{Float32,R,L},thread::Integer) where {T<:Union{Tylo.Float8_E4M3FN,Tylo.Float8_E5M2},N,K,A<:WGMMAAtom{T,N,K},R,L}
    L === typeof(Tylo._operand_layout(WGMMAAtom{BFloat16,K,16,1}(),Tylo.Accumulator())) ||
        return :(throw(ArgumentError("the scores are not a 64×$K accumulator fragment")))
    # Accumulator slot (1-based) of pair p, row half h, 8-column block b.
    slot(p,h,b) = 1 + p + 2h + 4b
    words = Expr[]
    body = Expr[]
    for j in 0:K÷32-1, h in 0:1
        pair(b) = :(_fp8_pair($T,f.data[$(slot(0,h,4j+b))],f.data[$(slot(1,h,4j+b))]))
        x0, x1, u0, u1 = (Symbol(n,"_",j,"_",h) for n in (:x0,:x1,:u0,:u1))
        push!(body,:($x0 = UInt32($(pair(0))) | (UInt32($(pair(2))) << 16)))
        push!(body,:($x1 = UInt32($(pair(1))) | (UInt32($(pair(3))) << 16)))
        push!(body,:($u0 = ifelse(high,Tylo._shuffle_idx($x1,from0),Tylo._shuffle_idx($x0,from0))))
        push!(body,:($u1 = ifelse(high,Tylo._shuffle_idx($x1,from1),Tylo._shuffle_idx($x0,from1))))
    end
    for j in 0:K÷32-1, b in 0:1, h in 0:1
        u0, u1 = Symbol(:u0_,j,"_",h), Symbol(:u1_,j,"_",h)
        push!(words, b == 0 ? :(($u0 & 0x0000ffff) | ($u1 << 16)) : :(($u0 >> 16) | ($u1 & 0xffff0000)))
    end
    quote
        Base.@inline
        lane = thread % Int32(32)
        q = lane & Int32(3)
        high = q >= Int32(2)                         # wants blocks 1 and 3
        from0 = (lane & ~Int32(3)) | ((q & Int32(1)) << 1)
        from1 = from0 | Int32(1)
        $(body...)
        Tylo._from_words($T,($(words...),),Tylo.operand_layout(atom,Tylo.OperandA()))
    end
end
end
