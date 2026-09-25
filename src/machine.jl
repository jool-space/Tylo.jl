"""
    MachineCode

A kernel compiled for one target, without launching it: its PTX, the cubin
ptxas assembles, the SASS in that cubin and ptxas's own report. Obtained
from [`machine_code`](@ref) or [`@machine_code`](@ref); [`audit`](@ref)
checks it and `show` summarizes it.

The fields are `entry` (the kernel's PTX name), `arch` (`"sm_90a"`, say),
`ptx`, `image`, `sass` and `log`, the output of `ptxas --verbose`.
"""
struct MachineCode
    entry::String
    arch::String
    ptx::String
    image::Vector{UInt8}
    sass::String
    log::String
    # Recompiles to LLVM IR annotated with source locations; used to trace
    # calls back to the lines that caused them.
    annotated::Any
end

"""
    machine_code(f, types; arch=nothing, threads=nothing, maxregs=nothing) -> MachineCode

Compile the kernel `f` for argument `types` (device-side types, a `Tuple`
type) as `@cuda` would, without a device when `arch` names the target
(`CUDACore.SMVersion(9,0,:arch)`, say) and for the current device
otherwise. `threads` is the block size the kernel requires, which bounds
its registers. Requires CUDACore.

    @machine_code [arch=...] [threads=...] f(args...)

The same for a call, taking the types from host arguments as `@cuda` would.
"""
function machine_code end

"""
    @machine_code [arch=...] [threads=...] [maxregs=...] f(args...)

[`machine_code`](@ref) for the kernel call `f(args...)`, with argument types
converted from host values as `@cuda` would.
"""
macro machine_code(ex...)
    isempty(ex) && throw(ArgumentError("@machine_code needs a kernel call"))
    call = ex[end]
    call isa Expr && call.head === :call || throw(ArgumentError("@machine_code needs a kernel call, as in @machine_code f(args...)"))
    kwargs = map(ex[1:end-1]) do kw
        kw isa Expr && kw.head === :(=) || throw(ArgumentError("options are written name=value"))
        Expr(:kw,kw.args...)
    end
    esc(:($(@__MODULE__)._machine_code_call($(call.args[1]),($(call.args[2:end]...),);$(kwargs...))))
end
function _machine_code_call end

"""
    Finding

One way a kernel's machine code falls short of what Tylo's kernels promise:
`kind` is `:call`, `:local_memory`, `:spill`, `:unaligned` or
`:wgmma_serialized`;
`message` says what; `sources` lists the Julia frames responsible, when
they can be traced, as `"file.jl:line (function)"` chains; `hint` says how
such code is usually fixed, when the cause is a known one.
"""
struct Finding
    kind::Symbol
    message::String
    sources::Vector{String}
    hint::String
end

"""
    audit(code::MachineCode; calls=true, local_memory=true, spills=true, alignment=true, wgmma=true) -> Vector{Finding}

Check the machine code against the properties Tylo's kernels promise, and
return what fails (empty when it all holds):

- `calls`: the kernel calls no functions. Each call is out-of-line code and
  an ABI boundary; ptxas also serializes every WGMMA in a kernel that has
  one. Julia inserts them for exceptions (bounds, `DivideError`,
  `DomainError`, `InexactError`) and for Base helpers marked `@noinline`
  (a `StepRange`'s overflow check). Each call is traced to the Julia
  frames that produced it.
- `local_memory`: no per-thread local memory, which lives in DRAM:
  arrays that did not become registers, or tuples indexed dynamically.
- `spills`: ptxas spilled no registers.
- `alignment`: no value wider than a byte is loaded or stored with
  alignment 1, which moves it byte by byte. `unsafe_load` and
  `unsafe_store!` on a `Core.LLVMPtr` assume alignment 1 unless given one.
- `wgmma`: every WGMMA stays asynchronous. ptxas serializes them, waiting
  after each instruction, when it cannot prove the pipeline safe, and says
  so only in a performance warning; the SASS then has more
  `WARPGROUP.ARRIVE`s than the PTX has `wgmma.fence`s.
"""
function audit(code::MachineCode; calls::Bool=true, local_memory::Bool=true, spills::Bool=true, alignment::Bool=true, wgmma::Bool=true)
    findings = Finding[]
    if calls
        targets = kernel_calls(code)
        if !isempty(targets)
            # One finding per origin: a line usually brings several calls
            # (a helper, an exception's constructor, its reporting).
            traced = _trace_calls(code)
            origins = Pair{String,Vector{String}}[]
            for name in unique(_base_name.(targets))
                for origin in get(traced,name,["unknown origin"])
                    i = findfirst(p -> p.first == origin,origins)
                    i === nothing ? push!(origins,origin => [name]) : push!(origins[i].second,name)
                end
            end
            for (origin,names) in origins
                hint = _call_hint(join(names," "),[origin])
                push!(findings,Finding(:call,"calls $(join(names,", "))",[origin],hint))
            end
        end
    end
    if local_memory
        bytes = local_bytes(code)
        accesses = _sass_count(code,r"\b(LDL|STL)\b")
        if bytes > 0 || accesses > 0
            sources = bytes > 0 ? _local_origins(code) : String[]
            hint = any(s -> occursin(r"\((sincos|sin|cos|tan|sinpi|cospi)\)",s),sources) ?
                "libdevice's sin and cos keep a 28-byte array for reducing large arguments (beyond about 1e5 radians); it is touched only then. Tables of precomputed values, or reduced arguments, avoid it." :
                isempty(kernel_calls(code)) ?
                "Values that did not become registers: an array or tuple indexed with a runtime index, or a closure or tuple the compiler did not unroll." :
                "The exception reporting ptxas inlined keeps a local buffer; it goes away with the calls."
            push!(findings,Finding(:local_memory,"$bytes bytes of local memory declared, $accesses local loads and stores",sources,hint))
        end
    end
    if spills
        stores, loads = spill_bytes(code)
        stores + loads > 0 &&
            push!(findings,Finding(:spill,"ptxas spilled registers ($stores bytes stored, $loads loaded)",String[],
                "Too many values live at once for the register budget: fewer accumulators, smaller tiles, or a larger budget (fewer threads)."))
    end
    # Split accesses show as byte-wide ones in the PTX; only then is the IR
    # worth reading.
    if alignment && occursin(r"\b(?:ld|st)(?:\.\w+)*\.[bsu]8\b",entry_ptx(code))
        # Byte elements, alone or in vectors, move byte by byte anyway.
        sites = trace(code,r"(?:= load (?!i8,|<\d+ x i8>)[^,]+, ptr|store (?!i8 |<\d+ x i8> )[^,]+, ptr)[^\n]*, align 1\b")
        if !isempty(sites)
            n = sum(last,sites)
            push!(findings,Finding(:unaligned,"$n loads and stores of wider values with alignment 1, moved byte by byte",first.(sites),
                "unsafe_load and unsafe_store! on a Core.LLVMPtr assume alignment 1 unless given one: pass Val(sizeof(T)), or the alignment the pointer has, as the last argument."))
        end
    end
    if wgmma
        w = wgmma_groups(code)
        if w.instructions > 0 && w.arrives != w.fences
            reason = something(_wgmma_warning(code),"no reason reported")
            push!(findings,Finding(:wgmma_serialized,
                "ptxas serialized WGMMA: $(w.arrives) warpgroup arrives for $(w.fences) fences ($reason)",String[],
                "Any call in the kernel does this, even on another warp's path; fix those first. Otherwise registers of a pending accumulator were touched before its wait."))
        end
    end
    findings
end

# How a call from Julia is usually removed, from the callee and the frames
# it came from.
function _call_hint(name,sources)
    from(f) = any(s -> occursin(f,s),sources)
    from("(steprange") || occursin("overflow_case",name) ?
        "A StepRange (a:step:b) with runtime bounds checks its step and computes its last element out of line. Iterate a unit range and scale the index, or loop with while." :
    from("(div)") || from("(rem)") || from("(cld)") || from("(fld)") || from("(mod)") || occursin("diverror",lowercase(name)) ?
        "Integer division by a runtime divisor checks for zero. Divide by a constant, or use Core.Intrinsics.sdiv_int/srem_int (udiv_int/urem_int) when the divisor is known to be nonzero." :
    occursin("domainerror",lowercase(name)) ?
        "sqrt, log or ^ of a value not known to be nonnegative throws a DomainError. When it is, use Core.Intrinsics.sqrt_llvm (IEEE sqrt, NaN for negatives)." :
    occursin("inexacterror",lowercase(name)) || from("(checked_trunc") || from("(toUInt") || from("(toInt") ?
        "A checked integer conversion, such as Int32(x) or UInt32(x) of a runtime Int. Use x % Int32 when the value fits, or pass constants of the right type." :
    occursin("boundserror",lowercase(name)) || from("(checkbounds") ?
        "A bounds check. Index under @inbounds where the index is known to be valid." :
    occursin("exception",name) ?
        "Reports an exception thrown on this path; the other calls from the same line say which." :
        ""
end

# The kernel's own body in the PTX module, without the helper functions
# (exception reporting) that follow it.
function entry_ptx(code::MachineCode)
    parts = split(code.ptx,".visible .entry";limit=2)
    length(parts) == 2 || return ""
    first(split(parts[2],"// -- End function";limit=2))
end

"The functions the kernel calls, one entry per call site, as named in its PTX."
kernel_calls(code::MachineCode) =
    [String(m[1]) for m in eachmatch(r"\bcall(?:\.uni)?\s+(?:\([^)]*\),\s*)?([A-Za-z_$][\w$]*)",entry_ptx(code))]

"Bytes of local memory the kernel declares in its PTX."
local_bytes(code::MachineCode) =
    sum((parse(Int,m[1]) for m in eachmatch(r"\.local\s[^;]*\[(\d+)\]",entry_ptx(code))); init=0)

"Registers per thread, as ptxas reports them."
registers(code::MachineCode) = (m = match(r"Used (\d+) registers",code.log); m === nothing ? 0 : parse(Int,m[1]))

"Bytes of spill stores and spill loads, as ptxas reports them."
function spill_bytes(code::MachineCode)
    m = match(r"(\d+) bytes spill stores, (\d+) bytes spill loads",code.log)
    m === nothing ? (0,0) : (parse(Int,m[1]),parse(Int,m[2]))
end

"Instructions in the kernel's SASS."
sass_instructions(code::MachineCode) = _sass_count(code,r"^\s+/\*[0-9a-f]{4,}\*/\s+\S"m)

"""
    wgmma_groups(code) -> (; instructions, fences, arrives, waits)

WGMMA instructions in the SASS, `wgmma.fence`s in the PTX, and the
warpgroup arrives and waits ptxas emitted; asynchronous code has one arrive
per fence.
"""
wgmma_groups(code::MachineCode) = (; instructions=_sass_count(code,r"\bHGMMA\b"),
    fences=count("wgmma.fence",entry_ptx(code)),
    arrives=_sass_count(code,r"WARPGROUP\.ARRIVE"),
    waits=_sass_count(code,r"WARPGROUP\.DEPBAR"))

"""
    memory_instructions(code) -> Vector{Pair{String,Int}}

The kernel's memory instructions by SASS opcode (`"LDG.E.128"`,
`"LDSM.16.M88.4"`, `"UTMALDG.2D"`, ...), most frequent first: which widths
the loads and stores became.
"""
function memory_instructions(code::MachineCode)
    counts = Dict{String,Int}()
    for m in eachmatch(r"^\s+/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?((?:LD|ST|LDG|STG|LDS|STS|LDSM|STSM|LDGSTS|UTMALDG|UTMASTG|UBLKCP|RED|ATOM)\b[\w.]*)"m,kernel_sass(code))
        counts[m[1]] = get(counts,m[1],0)+1
    end
    sort!(collect(counts);by=p -> (-p.second,p.first))
end

_sass_count(code::MachineCode,r::Regex) = count(r,kernel_sass(code))

"The kernel's own SASS: its text section, without the helper functions."
function kernel_sass(code::MachineCode)
    header = findfirst(Regex("^//-+ \\.text\\.\\Q$(code.entry)\\E\\s","m"),code.sass)
    header === nothing && return code.sass
    rest = code.sass[last(header)+1:end]
    next = findfirst(r"^//-{5,} \."m,rest)
    next === nothing ? rest : rest[1:first(next)-1]
end
_base_name(name::AbstractString) = replace(name,r"_\d+$" => "")
function _wgmma_warning(code::MachineCode)
    m = match(r"\((C75\d\d)\)[^:]*:\s*wgmma[^']*?serialized due to ([^']*?)(?: in the function|$)"m,code.log)
    m === nothing ? nothing : "$(m[1]): $(strip(m[2]))"
end

"""
    trace(code::MachineCode, pattern::Regex) -> Vector{Pair{String,Int}}

Where the kernel's LLVM instructions matching `pattern` come from: each
distinct origin, as a `"kernel.jl:12 (f) → lib.jl:40 (g) → int.jl:88 (+)"`
chain, with how many matching instructions it produced, most first. The
instructions are those of the optimized LLVM IR, before PTX, for instance
`r"load i8, ptr addrspace\\(3\\)"` for byte loads from shared memory or
`r"= alloca"` for what may become local memory.
"""
function trace(code::MachineCode,pattern::Regex)
    ir = try
        code.annotated()
    catch
        return Pair{String,Int}[]
    end
    counts = Dict{String,Int}()
    _trace(ir) do line, chain
        occursin(pattern,line) && (counts[chain] = get(counts,chain,0)+1)
    end
    sort!(collect(counts);by=p -> (-p.second,p.first))
end

# Allocas sit in the entry block with the kernel's own location; the
# instructions that use them say where the memory comes from.
function _local_origins(code::MachineCode)
    ir = try
        code.annotated()
    catch
        return String[]
    end
    _local_origins(ir)
end
function _local_origins(ir::AbstractString)
    names = String[]
    _trace(ir) do line, chain
        m = match(r"^\s*(%[\w.\"$:]+) = alloca\b",line)
        m === nothing || push!(names,m[1])
    end
    isempty(names) && return String[]
    uses = Regex(join(("\\Q$(n)\\E(?![\\w.\"])" for n in names),"|"))
    # Address arithmetic gets hoisted with unrelated locations; lifetime
    # markers, loads, stores and calls keep the frames that own the memory.
    origins = String[]
    _trace(ir) do line, chain
        occursin(r"= alloca\b|getelementptr",line) && return
        occursin(uses,line) && !(chain in origins) && push!(origins,chain)
    end
    origins
end

# Trace calls through the annotated LLVM IR of the kernel function. Maps a
# callee (without its numeric suffix) to one line per distinct origin.
function _trace_calls(code::MachineCode)
    ir = try
        code.annotated()
    catch
        return Dict{String,Vector{String}}()
    end
    _trace_calls(ir)
end
function _trace_calls(ir::AbstractString)
    origins = Dict{String,Vector{String}}()
    _trace(ir) do line, chain
        m = match(r"\bcall\b[^@]*@\"?([A-Za-z_][\w.$]*)\"?\(",line)
        (m === nothing || startswith(m[1],"llvm.")) && return
        list = get!(origins,_base_name(m[1]),String[])
        chain in list || push!(list,chain)
    end
    origins
end
# Walk the kernel function's instructions in `code_llvm` output with source
# annotations: the inlining comments (`; │┌ @ file.jl:12 within `f``) before
# an instruction give the chain of frames it came from. `f(line, chain)` is
# called for each instruction.
function _trace(f,ir::AbstractString)
    stack = String[]
    inside = false
    for line in eachline(IOBuffer(ir))
        # A function's outermost frame is printed just before its `define`.
        if startswith(line,"define ")
            inside = occursin("ptx_kernel",line)
            continue
        end
        startswith(line,"}") && (inside = false; empty!(stack); continue)
        if startswith(line,";")
            m = match(r"^;\s*([│┌└]*)\s*@ (.*)$",line)
            if m === nothing
                m = match(r"^;\s*([│]*)(└+)\s*$",line)
                m === nothing || resize!(stack,min(length(stack),length(m[1])+1))   # closes to depth |bars|
                continue
            end
            bars = m[1]
            depth = count(==('│'),bars)
            opens = endswith(bars,"┌")
            level = depth + (opens ? 2 : 1)
            resize!(stack,min(length(stack),level-1))
            while length(stack) < level-1; push!(stack,""); end
            push!(stack,_innermost(m[2]))
            continue
        end
        (inside && startswith(line,"  ")) || continue
        f(line,_describe(filter(!isempty,stack)))
    end
    nothing
end

# A comment line can hold several frames, `a.jl:1 within `f` @ b.jl:2`,
# the last of which may lack its function; it is the previous one's.
function _innermost(frames)
    pieces = split(frames," @ ")
    last_piece = String(pieces[end])
    (occursin(" within `",last_piece) || length(pieces) < 2) && return last_piece
    fn = match(r" within `(.*)`",pieces[end-1])
    fn === nothing ? last_piece : "$(last_piece) within `$(fn[1])`"
end
# "file.jl:12 within `f`" frames, innermost last: the kernel's own line, the
# innermost frame of code being written (outside Julia and installed
# packages), and the innermost frame, each once.
function _describe(frames)
    isempty(frames) && return "unknown origin"
    parsed = map(frames) do f
        m = match(r"^(.*?):(\d+)(?: within `(.*)`)?$",f)
        m === nothing ? (f,"","") : (String(m[1]),String(m[2]),something(m[3],"?"))
    end
    label((file,line,fn)) = isempty(line) ? file : "$(basename(file)):$line ($fn)"
    user = findlast(p -> isabspath(p[1]) && !_in_julia(p[1]) && !_in_packages(p[1]),parsed)
    user === nothing && (user = findlast(p -> isabspath(p[1]) && !_in_julia(p[1]),parsed))
    picks = unique(filter(!isnothing,[1,user,length(parsed)]))
    join(map(i -> label(parsed[i]),picks)," → ")
end
_in_julia(path) = startswith(path,normpath(joinpath(Sys.BINDIR,"..","share","julia")))
# Installed (not developed) packages: CUDACore, LLVM.jl, ... Code being
# written lives elsewhere, so it is the frame worth naming.
_in_packages(path) = any(d -> startswith(path,joinpath(d,"packages")),DEPOT_PATH)

function Base.show(io::IO,::MIME"text/plain",code::MachineCode)
    print(io,"MachineCode for ",code.arch,": ",code.entry)
    stores, loads = spill_bytes(code)
    print(io,"\n  ",registers(code)," registers, ",sass_instructions(code)," SASS instructions",
          stores + loads > 0 ? ", spills $stores/$loads bytes" : "")
    w = wgmma_groups(code)
    if w.instructions > 0
        print(io,"\n  WGMMA: ",w.instructions," instructions, ",w.fences," fences, ",w.arrives," arrives",
              w.arrives == w.fences ? " (asynchronous)" : " (serialized)")
    end
    memory = memory_instructions(code)
    isempty(memory) || print(io,"\n  memory: ",join(("$op ×$n" for (op,n) in first(memory,8)),", "))
    findings = audit(code)
    if isempty(findings)
        print(io,"\n  audit: clean")
    else
        for f in findings
            print(io,"\n  ✗ ",f.message)
            for s in f.sources
                print(io,"\n      from ",s)
            end
            isempty(f.hint) || print(io,"\n      hint: ",f.hint)
        end
    end
end
Base.show(io::IO,code::MachineCode) = print(io,"MachineCode(",code.arch,", ",code.entry,")")
function Base.show(io::IO,::MIME"text/plain",f::Finding)
    print(io,f.kind,": ",f.message)
    for s in f.sources
        print(io,"\n    from ",s)
    end
    isempty(f.hint) || print(io,"\n    hint: ",f.hint)
end
