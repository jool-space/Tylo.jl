# FP32 pairs. `<2 x float>` arithmetic is one instruction on sm_100 and
# later (FFMA2, FADD2, FMUL2), where LLVM selects the packed forms, and two
# scalar ones elsewhere: the same code serves every target. Fragment
# broadcasts and sums of FP32 values run on pairs of slots through it.
const _V2 = NTuple{2,VecElement{Float32}}
struct F32x2
    v::_V2
end
@inline F32x2(a::Float32,b::Float32) = F32x2((VecElement(a),VecElement(b)))
@inline F32x2(a::Float32) = F32x2(a,a)
@inline F32x2(p::F32x2) = p
@inline _lanes(p::F32x2) = (p.v[1].value,p.v[2].value)

for (name,op) in ((:_add2,"fadd"),(:_sub2,"fsub"),(:_mul2,"fmul"))
    @eval @inline $name(a::_V2,b::_V2) =
        Base.llvmcall($("%r = $op <2 x float> %0, %1\nret <2 x float> %r"),_V2,Tuple{_V2,_V2},a,b)
end
# muladd's contraction, as for scalars: a fused multiply-add where it pays.
@inline _muladd2(a::_V2,b::_V2,c::_V2) =
    Base.llvmcall("%p = fmul contract <2 x float> %0, %1\n%r = fadd contract <2 x float> %p, %2\nret <2 x float> %r",_V2,Tuple{_V2,_V2,_V2},a,b,c)

const _PairArg = Union{F32x2,Float32}
@inline Base.:+(a::F32x2,b::_PairArg) = F32x2(_add2(a.v,F32x2(b).v))
@inline Base.:+(a::Float32,b::F32x2) = b+a
@inline Base.:-(a::F32x2,b::_PairArg) = F32x2(_sub2(a.v,F32x2(b).v))
@inline Base.:-(a::Float32,b::F32x2) = F32x2(_sub2(F32x2(a).v,b.v))
@inline Base.:-(a::F32x2) = F32x2(_sub2(F32x2(-0f0).v,a.v))
@inline Base.:*(a::F32x2,b::_PairArg) = F32x2(_mul2(a.v,F32x2(b).v))
@inline Base.:*(a::Float32,b::F32x2) = b*a
@inline Base.muladd(a::F32x2,b::_PairArg,c::_PairArg) = F32x2(_muladd2(a.v,F32x2(b).v,F32x2(c).v))
@inline Base.muladd(a::Float32,b::F32x2,c::_PairArg) = muladd(b,a,c)
@inline Base.muladd(a::Float32,b::Float32,c::F32x2) = F32x2(_muladd2(F32x2(a).v,F32x2(b).v,c.v))

# How a function applies to a pair of FP32 values giving FP32 values:
# packed (the methods above), per lane, or not at all (the broadcast then
# runs value by value). Functions defined later add methods.
_pair_op(f) = :none
_pair_op(::Union{typeof(+),typeof(-),typeof(*),typeof(muladd)}) = :packed
_pair_op(::Union{typeof(exp2),typeof(max),typeof(min),typeof(abs),typeof(identity),typeof(inv),typeof(/)}) = :lane
