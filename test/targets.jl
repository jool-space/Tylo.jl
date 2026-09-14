# Capability gating for `gpu/` test files. A file states which live devices
# can execute its runtime sections in a banner within its first 20 lines:
#
#   # TEST_TARGET: cc>=8.0          any device from Ampere on
#   # TEST_TARGET: cc==9.0          exactly Hopper
#   # TEST_TARGET: cc==10|cc==11    any datacenter Blackwell family
#
# Assembly checks in the same file run wherever the CUDA compiler is
# available; only the sections guarded by `runtime_supported(@__FILE__)`
# depend on the device. A file without a banner defaults to `cc>=8.0`.
module TestTargets
export runtime_predicates, runtime_supported, describe_predicates

struct Predicate
    kind::Symbol
    version::VersionNumber
end
const BANNER = r"^\s*#\s*TEST_TARGET\s*:\s*(.*?)\s*$"
const DEFAULT = [Predicate(:minimum, v"8.0")]

function parse_predicate(text::AbstractString)
    m = match(r"^cc>=(\d+)(?:\.(\d+))?$", text)
    m !== nothing && return Predicate(:minimum,
        VersionNumber(parse(Int, m[1]), m[2] === nothing ? 0 : parse(Int, m[2])))
    m = match(r"^cc==(\d+)(?:\.(\d+))?$", text)
    m !== nothing && return m[2] === nothing ?
        Predicate(:major, VersionNumber(parse(Int, m[1]))) :
        Predicate(:exact, VersionNumber(parse(Int, m[1]), parse(Int, m[2])))
    throw(ArgumentError("invalid TEST_TARGET predicate $(repr(text)); expected cc>=X[.Y], cc==X.Y or cc==X"))
end
satisfied(p::Predicate, cap::VersionNumber) =
    p.kind === :minimum ? cap >= p.version :
    p.kind === :exact ? cap == p.version : cap.major == p.version.major

"The banner's predicates, or `nothing` when the file has no banner."
function runtime_predicates(file::AbstractString)
    for (i, line) in enumerate(eachline(file))
        i > 20 && break
        m = match(BANNER, line)
        m === nothing && continue
        return [parse_predicate(strip(t)) for t in split(m[1], '|')]
    end
    nothing
end
function runtime_supported(file::AbstractString, cap::VersionNumber)
    predicates = something(runtime_predicates(file), DEFAULT)
    any(p -> satisfied(p, cap), predicates)
end
describe_predicates(predicates) = join((p.kind === :minimum ? "cc>=$(p.version.major).$(p.version.minor)" :
    p.kind === :exact ? "cc==$(p.version.major).$(p.version.minor)" : "cc==$(p.version.major)"
    for p in predicates), "|")
describe_predicates(::Nothing) = describe_predicates(DEFAULT)
end
