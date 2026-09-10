# Pure coordinate machinery. Shapes and coordinates use logical elements,
# never byte addresses or packed TMEM columns.
module Layouts

using Static: StaticInt, static

export Layout, @Layout, Swizzle, compose, window, shape, cosize, static, coordinate

include("affine.jl")
include("notation.jl")
include("transforms.jl")

function layout end

"""
    LaneRows{N}()

Ownership of a 32×N tile: lane t owns row t, with N consecutive values.
This is a row distribution, independent of the load/store instruction.
"""
struct LaneRows{N}
    function LaneRows{N}() where N
        N isa Int && N > 0 || throw(ArgumentError("positive static row width required"))
        new{N}()
    end
end
Base.size(::LaneRows{N}) where N = (32, N)

@inline function coordinate(::LaneRows{N}, lane::Integer, ::Val{E}) where {N,E}
    E isa Int && 0 <= E < N || throw(BoundsError())
    (lane, oftype(lane, E))
end

# Compatibility slicing for the original lane-local register fragments.
# Static indexing keeps tuples in registers.
function check_columns(n, first, width)
    first isa Int && width isa Int && 0 <= first && 0 < width &&
        first <= n - width || throw(ArgumentError("column interval is outside the tile"))
    nothing
end

end
