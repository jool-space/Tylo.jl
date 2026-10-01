# Compare the machine code of two evidence directories (`TYLO_EVIDENCE`),
# kernel by kernel. Does not load a kernel.
#
#     julia --project=test test/tools/sassdiff.jl BEFORE AFTER [NAME...]
#
# A kernel whose PTX is identical is unchanged. Otherwise its SASS is
# disassembled and normalized (addresses, encodings, and the numbering of
# registers and labels removed) and compared: identical, or reordered with
# the same instruction histogram, or different, with the opcodes whose counts
# changed. With NAMEs, the normalized SASS of those kernels is diffed in full.
using CUDACore

length(ARGS) >= 2 || error("usage: julia --project=test test/tools/sassdiff.jl BEFORE AFTER [NAME...]")
before, after = abspath(ARGS[1]), abspath(ARGS[2])
selected = ARGS[3:end]

kernels(dir) = Set(splitext(f)[1] for f in readdir(dir) if endswith(f, ".cubin"))

# Instructions of every function in the cubin, normalized: no addresses or
# encodings, registers and labels unnumbered, Julia's symbol numbering removed.
function sass(path)
    text = read(`$(CUDACore.CUDA_Compiler.nvdisasm()) -c $path`, String)
    lines = String[]
    for l in split(text, '\n')
        occursin(r"^\s+/\*[0-9a-f]{4,}\*/", l) || continue
        l = replace(l, r"/\*[0-9a-f]{4,}\*/" => "", r"/\*.*?\*/" => "", r"\s*;.*$" => "")
        l = replace(l, r"\b(U?R|U?P|B)\d+\b" => s"\1", r"\.L_x_\d+" => ".L", r"_\d{2,}" => "_N")
        l = strip(replace(l, r"\s+" => " "))
        isempty(l) || push!(lines, l)
    end
    lines
end
opcode(l) = (w = split(l); i = startswith(w[1], "@") ? 2 : 1; i <= length(w) ? w[i] : "")
function histogram(lines)
    h = Dict{String,Int}()
    for l in lines
        o = opcode(l)
        h[o] = get(h, o, 0) + 1
    end
    h
end

a, b = kernels(before), kernels(after)
for name in sort(collect(setdiff(a, b))); println(rpad(name, 60), "removed"); end
for name in sort(collect(setdiff(b, a))); println(rpad(name, 60), "added"); end
counts = Dict("same PTX" => 0, "same SASS" => 0, "reordered" => 0, "different" => 0)
for name in sort(collect(intersect(a, b)))
    ptx(dir) = (p = joinpath(dir, name * ".ptx"); isfile(p) ? read(p, String) : nothing)
    if ptx(before) !== nothing && ptx(before) == ptx(after)
        counts["same PTX"] += 1
        name in selected && println(rpad(name, 60), "same PTX")
        continue
    end
    x, y = sass(joinpath(before, name * ".cubin")), sass(joinpath(after, name * ".cubin"))
    if x == y
        counts["same SASS"] += 1
        println(rpad(name, 60), "same SASS (PTX differs)")
        continue
    end
    hx, hy = histogram(x), histogram(y)
    if hx == hy
        counts["reordered"] += 1
        println(rpad(name, 60), "reordered, same histogram ($(length(x)) instructions)")
    else
        counts["different"] += 1
        changes = sort([(o, get(hx, o, 0), get(hy, o, 0)) for o in union(keys(hx), keys(hy)) if get(hx, o, 0) != get(hy, o, 0)];
                       by = c -> -abs(c[3] - c[2]))
        println(rpad(name, 60), "different: $(length(x)) → $(length(y)) instructions")
        println("    ", join(["$o $p→$q" for (o, p, q) in changes], ", "))
    end
    if name in selected
        mktempdir() do dir
            write(joinpath(dir, "before"), join(x, '\n') * '\n')
            write(joinpath(dir, "after"), join(y, '\n') * '\n')
            run(ignorestatus(`diff -u $(joinpath(dir, "before")) $(joinpath(dir, "after"))`))
        end
    end
end
println(join(["$k: $v" for (k, v) in sort(collect(counts))], ", "))
