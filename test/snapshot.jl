# Text-section fingerprints for saved kernel images.
#
# `snapshot_manifest(directory)` walks an evidence directory produced by
# `save_code` and records, for every cubin, the SHA256 of its executable
# sections. `check_snapshot(name, image)` compares a freshly compiled image
# against the manifest named by `TYLO_SNAPSHOT`, when that variable is set.
#
# Executable sections are compared, never debug or source metadata. A changed
# toolchain, PTX revision or Julia version invalidates a manifest; the header
# records those so a mismatch can be attributed before it is investigated.
using SHA, TOML, Test, CUDACore

# Every `.text.*` section of a CUDA ELF64 little-endian image, sorted by
# section name so the fingerprint does not depend on emission order.
function text_sections(image::Vector{UInt8})
    @assert image[1:6] == UInt8[0x7f,0x45,0x4c,0x46,2,1]
    u16(off) = Int(ltoh(reinterpret(UInt16,image[off+1:off+2])[1]))
    u32(off) = Int(ltoh(reinterpret(UInt32,image[off+1:off+4])[1]))
    u64(off) = Int(ltoh(reinterpret(UInt64,image[off+1:off+8])[1]))
    table = u64(0x28)
    stride, count, names_idx = u16(0x3a), u16(0x3c), u16(0x3e)
    names = u64(table+names_idx*stride+0x18)
    sections = Pair{String,Vector{UInt8}}[]
    for idx in 0:count-1
        header = table+idx*stride
        start = names+u32(header)+1
        stop = findnext(iszero,image,start)
        name = String(image[start:stop-1])
        startswith(name,".text.") || continue
        offset, bytes = u64(header+0x18), u64(header+0x20)
        push!(sections,name=>image[offset+1:offset+bytes])
    end
    sort!(sections;by=first)
end

# Order-independent hash of the executable bytes. Section names are excluded
# so renaming a Julia type, which renames the mangled entry symbol, does not
# change the fingerprint of unchanged machine code.
function text_fingerprint(image::Vector{UInt8})
    hashes = sort!([sha256(bytes) for (_,bytes) in text_sections(image)])
    ctx = SHA256_CTX()
    for h in hashes
        update!(ctx,h)
    end
    bytes2hex(digest!(ctx))
end

# ptxas does not allocate registers deterministically for every kernel: two
# assemblies of identical PTX can differ only in register numbering. The
# normalized fingerprint hashes the disassembled instruction stream with
# register and predicate numbers removed, so such kernels still compare as
# equivalent while any change in opcodes, immediates or ordering is caught.
function _disassemble(image::Vector{UInt8})
    path = tempname()*".cubin"
    write(path,image)
    try
        read(`$(CUDACore.CUDA_Compiler.nvdisasm()) -c $path`,String)
    finally
        rm(path;force=true)
    end
end
function normalized_sass(image::Vector{UInt8})
    lines = String[]
    for line in split(_disassemble(image),'\n')
        m = match(r"^\s*/\*[0-9a-f]+\*/\s*(.*?)\s*$",line)
        m === nothing && continue
        # Register numbers, reuse-cache flags and symbol names (mangled
        # entry names carry Julia type names; LLVM helpers carry session
        # numbers) vary between assemblies of equivalent code.
        text = replace(m.captures[1], r"\bU?R\d+\b"=>"R", r"\bU?P\d\b"=>"P",
                       ".reuse"=>"", r"`\([^)]*\)"=>"`(SYM)")
        push!(lines,text)
    end
    lines
end
normalized_fingerprint(image::Vector{UInt8}) = bytes2hex(sha256(join(normalized_sass(image),'\n')))
# ptxas also reorders independent instructions between assemblies of some
# kernels; the multiset fingerprint ignores instruction order entirely.
multiset_fingerprint(image::Vector{UInt8}) = bytes2hex(sha256(join(sort(normalized_sass(image)),'\n')))

function _git(root,args::Vector{String})
    try
        strip(read(`git -C $root $args`,String))
    catch
        "unavailable"
    end
end

"""
    snapshot_manifest(directory; output=joinpath(directory,"manifest.toml"))

Fingerprint every `.cubin` in `directory` and write a manifest with the
toolchain and source revisions that produced them.
"""
function snapshot_manifest(directory; output=joinpath(directory,"manifest.toml"))
    tylo = normpath(joinpath(@__DIR__,".."))
    ptx = normpath(joinpath(tylo,"..","PTX"))
    kernels = Dict{String,Any}()
    for file in sort(readdir(directory))
        endswith(file,".cubin") || continue
        image = read(joinpath(directory,file))
        kernels[splitext(file)[1]] = Dict("text"=>text_fingerprint(image),
                                          "normalized"=>normalized_fingerprint(image),
                                          "multiset"=>multiset_fingerprint(image))
    end
    header = Dict(
        "julia"=>string(VERSION),
        "cudacore"=>string(pkgversion(CUDACore)),
        "cuda_compiler"=>string(CUDACore.compiler_version()),
        "tylo_commit"=>_git(tylo,["rev-parse","HEAD"]),
        "tylo_diff_sha256"=>bytes2hex(sha256(_git(tylo,["diff","HEAD","--","src","ext","examples","test"]))),
        "ptx_commit"=>_git(ptx,["rev-parse","HEAD"]),
        "kernel_count"=>length(kernels))
    open(output,"w") do io
        TOML.print(io,Dict("toolchain"=>header,"kernels"=>kernels))
    end
    output
end

const _SNAPSHOT = Ref{Union{Nothing,Dict{String,Any}}}(nothing)
function _snapshot()
    _SNAPSHOT[] === nothing || return _SNAPSHOT[]
    path = get(ENV,"TYLO_SNAPSHOT","")
    _SNAPSHOT[] = isempty(path) ? Dict{String,Any}() : TOML.parsefile(path)
end

"""
    snapshot_check(name, image)

Assert that `name` is unchanged relative to the selected manifest. Names
matched by the comma-separated patterns in `TYLO_SNAPSHOT_ALLOW` may change;
names absent from the manifest are new kernels and pass.
"""
const _SNAPSHOT_STATUSES = Dict{Symbol,Vector{String}}()
function snapshot_check(name, image::Vector{UInt8})
    status = check_snapshot(name,image)
    if status === :changed
        allowed = split(get(ENV,"TYLO_SNAPSHOT_ALLOW",""),",";keepempty=false)
        any(pattern -> occursin(Regex(pattern),name),allowed) && (status = :allowed)
    end
    push!(get!(_SNAPSHOT_STATUSES,status,String[]),name)
    # Test workers run in parallel; the runner aggregates their statuses.
    if haskey(ENV,"TYLO_SNAPSHOT_REPORT") && status !== :disabled
        open(ENV["TYLO_SNAPSHOT_REPORT"],"a") do io
            println(io,status," ",name)
        end
    end
    status in (:disabled,:absent,:identical,:equivalent,:reordered,:allowed) && return status
    @test status in (:identical,:equivalent,:reordered)
    status
end

"""
    snapshot_report([file])

Print how every saved kernel compared against the selected manifest, from
this process or from the statuses workers appended to `file`.
"""
function snapshot_report(file=nothing)
    statuses = _SNAPSHOT_STATUSES
    if file !== nothing
        statuses = Dict{Symbol,Vector{String}}()
        for line in eachline(file)
            status,name = split(line," ";limit=2)
            push!(get!(statuses,Symbol(status),String[]),name)
        end
    end
    isempty(get(ENV,"TYLO_SNAPSHOT","")) && return
    for status in (:identical,:equivalent,:reordered,:allowed,:absent,:changed)
        names = get(statuses,status,String[])
        isempty(names) && continue
        println("snapshot ",status,": ",length(names),
                status in (:identical,) ? "" : " ("*join(sort(names),", ")*")")
    end
end

"""
    check_snapshot(name, image)

Return `:identical` (same executable bytes), `:equivalent` (same instruction
stream up to register numbering), `:reordered` (same instruction multiset),
`:changed`, or `:absent` for `name` against the manifest selected by
`TYLO_SNAPSHOT`; `:disabled` when no manifest is selected.
"""
function check_snapshot(name, image::Vector{UInt8})
    manifest = _snapshot()
    isempty(manifest) && return :disabled
    kernels = manifest["kernels"]
    haskey(kernels,name) || return :absent
    kernels[name]["text"] == text_fingerprint(image) && return :identical
    kernels[name]["normalized"] == normalized_fingerprint(image) && return :equivalent
    kernels[name]["multiset"] == multiset_fingerprint(image) ? :reordered : :changed
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: julia --project=test test/snapshot.jl EVIDENCE_DIRECTORY")
    path = snapshot_manifest(abspath(only(ARGS)))
    manifest = TOML.parsefile(path)
    println(path,": ",manifest["toolchain"]["kernel_count"]," kernels")
end
