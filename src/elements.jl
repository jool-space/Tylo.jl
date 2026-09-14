# Element types and their register packing widths. The 8-bit floating-point
# formats are Microfloats types with the `cvt.rn.satfinite` policy: E4M3 has
# no infinities and one NaN pattern per sign; E5M2 has IEEE-style
# infinities and NaNs. Conversions from FP32 round to nearest even and
# saturate any larger magnitude, infinities included, to the largest finite
# value. Arithmetic on them goes through FP32.
using Microfloats: @microfloat, NanOnlyAllOnes, SAT

"""
    Float8E4M3

OCP FP8 E4M3 (no infinities, NaN at `0x7f`/`0xff`), converting from FP32 with
round-to-nearest-even and saturation like `cvt.rn.satfinite.e4m3x2.f32`.
"""
@microfloat Float8E4M3 exponent=4 significand=3 nonfinite=NanOnlyAllOnes overflow=SAT
"""
    Float8E5M2

FP8 E5M2 with IEEE-style infinities and NaNs, converting from FP32 with
round-to-nearest-even and saturation like `cvt.rn.satfinite.e5m2x2.f32`.
"""
@microfloat Float8E5M2 exponent=5 significand=2 overflow=SAT

_element_bits(::Type{Float8E4M3}) = 8
_element_bits(::Type{Float8E5M2}) = 8
_element_bits(::Type{Int8}) = 8
_element_bits(::Type{UInt8}) = 8
_element_bits(::Type{BFloat16}) = 16
_element_bits(::Type{Float16}) = 16
_element_bits(::Type{Float32}) = 32
_element_bits(::Type{Int32}) = 32
_element_bits(::Type) = 0
_carrier(::Type{T}) where T = _element_bits(T) == 8 ? UInt8 : _element_bits(T) == 16 ? UInt16 : UInt32
