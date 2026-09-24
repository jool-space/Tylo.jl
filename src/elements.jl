# Element types and their register packing widths. The 8-bit floating-point
# formats are the Microfloats types: `Float8_E4M3FN` has no infinities and one
# NaN pattern per sign; `Float8_E5M2` has IEEE-style infinities and NaNs.
# Tylo converts into them the way the hardware does (`cvt.rn.satfinite`):
# round to nearest even and saturate any larger magnitude, infinities
# included, to the largest finite value. Arithmetic on them goes through FP32.
using Microfloats: Microfloats, Microfloat, Float8_E4M3FN, Float8_E5M2, Float8_E8M0FNU, SAT

"""
    convert_element(T, x)

Numerically convert one value to element type `T`. Microfloat destinations
round to nearest even and saturate, whatever overflow policy the type
declares, so host and device agree with `cvt.rn.satfinite`. Every conversion
goes through Microfloats' conversion funnel, which lowers to the native
instruction in kernels compiled for a target that has it.
"""
@inline convert_element(::Type{T},x) where T<:Microfloat = Microfloats.cvt(T,x,RoundNearest,SAT)
@inline convert_element(::Type{T},x::T) where T<:Microfloat = x
@inline convert_element(::Type{T},x) where T = T(x)

_element_bits(::Type{Float8_E4M3FN}) = 8
_element_bits(::Type{Float8_E5M2}) = 8
_element_bits(::Type{Float8_E8M0FNU}) = 8 # block scales: unsigned powers of two
_element_bits(::Type{Int8}) = 8
_element_bits(::Type{UInt8}) = 8
_element_bits(::Type{BFloat16}) = 16
_element_bits(::Type{Float16}) = 16
_element_bits(::Type{Float32}) = 32
_element_bits(::Type{Int32}) = 32
_element_bits(::Type) = 0
_carrier(::Type{T}) where T = _element_bits(T) == 8 ? UInt8 : _element_bits(T) == 16 ? UInt16 : UInt32
