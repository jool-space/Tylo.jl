# Approximate exponentials and logarithms, as softmax kernels compute them.

"""
    exp2_approx(x::Float32)

2^x by the GPU's `ex2.approx.ftz` (a special-function unit instruction, 2
ulp, subnormal results flushed to zero); `exp2` on the host. Broadcast over
FP32 fragments it applies lane by lane to pairs of slots.

    exp2_approx(f::Fragment{Float32}, emulate::Tuple)

2^x of every value of `f`, the pairs of slots listed in `emulate`
(zero-based pair indices: slots 2k and 2k+1) on the FMA pipe instead, as
FlashAttention-4 computes part of each row: a degree-3 polynomial on packed
pairs, with the integer part added into the exponent (sm_100 and later).
Where the special-function unit would otherwise be as busy as the tensor
cores, moving part of its work to the FMA pipe shortens the softmax.
`emulate` must be a constant.
"""
@inline exp2_approx(x::Float32) = _ex2(x)
_pair_op(::typeof(exp2_approx)) = :lane
Base.@constprop :aggressive @inline exp2_approx(f::Fragment{Float32},emulate::Tuple) = _exp2_mixed(f,Val(emulate))
@generated function _exp2_mixed(f::Fragment{Float32,N},::Val{E}) where {N,E}
    isempty(E) && return :(Base.@inline; exp2_approx.(f))
    all(k -> k isa Integer && 0 <= k < N ÷ 2,E) || return :(throw(ArgumentError("emulated pairs index the fragment's slot pairs")))
    names = Symbol[]
    stmts = Any[:(v = f.data)]
    for k in 0:N÷2-1
        a, b = Symbol(:e,2k+1), Symbol(:e,2k+2)
        push!(stmts,k in E ? :(($a,$b) = _exp2_fma(v[$(2k+1)],v[$(2k+2)])) :
                             :(($a,$b) = (_ex2(v[$(2k+1)]),_ex2(v[$(2k+2)]))))
        push!(names,a,b)
    end
    isodd(N) && (push!(stmts,:($(Symbol(:e,N)) = _ex2(v[$N]))); push!(names,Symbol(:e,N)))
    :(Base.@inline; $(stmts...); Fragment(($(names...),),Layouts.layout(f)))
end
# 2^x of a pair on the FMA pipe; the host evaluates the same polynomial.
@inline function _exp2_fma(x0::Float32,x1::Float32)
    _exp2_poly(x0), _exp2_poly(x1)
end
@inline function _exp2_poly(x::Float32)
    x = max(x,-127f0)
    n = floor(x)
    f = x-n
    c3, c2, c1 = reinterpret(Float32,0x3D9DF09D), reinterpret(Float32,0x3E6906A4), reinterpret(Float32,0x3F31F519)
    q = muladd(muladd(muladd(f,c3,c2),f,c1),f,1f0)
    reinterpret(Float32,(reinterpret(UInt32,q)+(unsafe_trunc(Int32,n) % UInt32) << 23))
end

"""
    log2_approx(x::Float32)

log₂ x by the GPU's `lg2.approx.ftz`; `log2` on the host.
"""
@inline log2_approx(x::Float32) = log2(x)
