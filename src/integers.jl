"""
    unchecked_div(a, b)
    unchecked_rem(a, b)

Integer `÷` and `%` without Julia's checks for a zero divisor (and for
`typemin ÷ -1`). Those checks throw through an out-of-line call, which
costs a branch and, in a kernel with WGMMA, makes ptxas serialize all of
it; [`audit`](@ref) reports them. The divisor must not be zero.
"""
@inline unchecked_div(a::T,b::T) where T<:Signed = Core.Intrinsics.sdiv_int(a,b)
@inline unchecked_div(a::T,b::T) where T<:Unsigned = Core.Intrinsics.udiv_int(a,b)
@doc (@doc unchecked_div) @inline unchecked_rem(a::T,b::T) where T<:Signed = Core.Intrinsics.srem_int(a,b)
@inline unchecked_rem(a::T,b::T) where T<:Unsigned = Core.Intrinsics.urem_int(a,b)
