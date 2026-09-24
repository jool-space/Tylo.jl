# Pure coordinate machinery. Shapes and coordinates use logical elements,
# never byte addresses or packed TMEM columns.
module Layouts

using Static: StaticInt, static

export Layout, @Layout, Swizzle, compose, window, shape, cosize, static, coordinate
export concat, coalesce, complement, right_inverse, left_inverse,
       logical_divide, zipped_divide, tiled_divide,
       logical_product, blocked_product, raked_product,
       agree, is_injective, is_compact

include("affine.jl")
include("notation.jl")
include("transforms.jl")
include("algebra.jl")

function layout end

"""
    LocalOwnership{N,Axis}()
    StripedOwnership{N,Axis}()

Two explicit warp ownership patterns. Each lane holds `N` local values along
logical `Axis` (1 or 2). `LocalOwnership` assigns one independent sequence to
each lane; `StripedOwnership` interleaves the 32 lanes within one sequence.
Neither describes memory strides. Coordinates take a zero-based lane (0:31),
local to this warp, and a zero-based value slot. Use `Ownership` for an explicit mapping.
"""
struct LocalOwnership{N,Axis}
    function LocalOwnership{N,Axis}() where {N,Axis}
        _check_ownership(N,Axis)
        new{N,Axis}()
    end
end
struct StripedOwnership{N,Axis}
    function StripedOwnership{N,Axis}() where {N,Axis}
        _check_ownership(N,Axis)
        new{N,Axis}()
    end
end
function _check_ownership(n,axis)
    n isa Int && n > 0 && axis isa Int && axis in (1,2) ||
        throw(ArgumentError("ownership requires a positive value count and axis 1 or 2"))
end
Base.size(::LocalOwnership{N,A}) where {N,A} = A == 2 ? (32,N) : (N,32)
Base.size(::StripedOwnership{N,A}) where {N,A} = A == 2 ? (1,32N) : (32N,1)
@inline function coordinate(::LocalOwnership{N,A},t::Integer,::Val{E}) where {N,A,E}
    E isa Int && 0 <= E < N || throw(BoundsError())
    A == 2 ? (t,oftype(t,E)) : (oftype(t,E),t)
end
@inline function coordinate(::StripedOwnership{N,A},t::Integer,::Val{E}) where {N,A,E}
    E isa Int && 0 <= E < N || throw(BoundsError())
    value = t+oftype(t,32E)
    A == 2 ? (zero(t),value) : (value,zero(t))
end
Base.permutedims(::LocalOwnership{N,A}) where {N,A} = LocalOwnership{N,3-A}()
Base.permutedims(::StripedOwnership{N,A}) where {N,A} = StripedOwnership{N,3-A}()

end
