"""
    SoftmaxState(fragment_or_ownership)

FP32 running row maxima and unnormalized exponential sums in the fragment's
row distribution. The empty state is `(-Inf, 0)`. Scores must be finite or
`-Inf` (masked). State arithmetic does not allocate storage or synchronize.
"""
struct SoftmaxState{R<:RowValues}
    maximum::R
    sum::R
end
@inline function SoftmaxState(o::RowOwnership)
    SoftmaxState(RowValues(o,ntuple(_ -> -Inf32,Val(_row_count(o)))),
                 RowValues(o,ntuple(_ -> 0f0,Val(_row_count(o)))))
end
@inline SoftmaxState(f) = SoftmaxState(row_ownership(f))
row_ownership(s::SoftmaxState) = row_ownership(s.maximum)

# Explicit scalar calls keep replicated row state in registers. Matching types
# are intentional: arithmetic never redistributes results between owners.
@generated function _row_zip(f::F,a::RowValues{O,N},b::RowValues{O,N}) where {F,O,N}
    values=[:(f(a.data[$i],b.data[$i])) for i in 1:N]
    quote
        Base.@inline
        RowValues($O(),($(values...),))
    end
end
@generated function _row_zip(f::F,a::RowValues{O,N},b::RowValues{O,N},c::RowValues{O,N}) where {F,O,N}
    values=[:(f(a.data[$i],b.data[$i],c.data[$i])) for i in 1:N]
    quote
        Base.@inline
        RowValues($O(),($(values...),))
    end
end
@inline _softmax_rescale(old,new) = old == -Inf32 ? 0f0 : exp(old-new)
@inline _softmax_weight(x,m) = m == -Inf32 ? 0f0 : exp(x-m)

"""
    softmax_update(state, scores) -> (; state, weights, rescale)

Update running statistics with a score tile. `weights` are unnormalized
`exp(scores - new_maximum)` in the original fragment distribution. Multiply
an existing weighted numerator by `rescale` before adding this tile's weighted
values. Statistics use FP32 weights, before any conversion for another MMA.
Empty chunks produce zero weights; an empty previous state has zero rescale.
Warp/MMA fragments require all lanes to participate in their row collectives.
"""
@inline function softmax_update(s::SoftmaxState,f)
    m = _row_zip(max,s.maximum,row_max(f))
    alpha = _row_zip(_softmax_rescale,s.maximum,m)
    p = row_map(_softmax_weight,f,m)
    l = _row_zip(muladd,alpha,s.sum,row_sum(p))
    (;state=SoftmaxState(m,l),weights=p,rescale=alpha)
end

"""
    softmax_merge(left, right) -> (; state, left_rescale, right_rescale)

Combine independent summaries with identical row ownership. Combine their
unnormalized weighted numerators using the returned factors. Floating-point
merge order may change the result. Two empty summaries remain empty.
"""
@inline function softmax_merge(a::SoftmaxState{R},b::SoftmaxState{R}) where R
    m = _row_zip(max,a.maximum,b.maximum)
    left = _row_zip(_softmax_rescale,a.maximum,m)
    right = _row_zip(_softmax_rescale,b.maximum,m)
    l = _row_zip(muladd,left,a.sum,_row_zip(*,right,b.sum))
    (;state=SoftmaxState(m,l),left_rescale=left,right_rescale=right)
end

"Normalize using one FP32 reciprocal per row and multiplication; empty rows return zero."
@inline function softmax_normalize(f,s::SoftmaxState)
    reciprocal=map(l -> l == 0f0 ? 0f0 : inv(l),s.sum)
    row_map((x,r) -> r == 0f0 ? 0f0 : x*r,f,reciprocal)
end

"Final log-sum-exp in the state's row distribution; empty rows return -Inf."
@inline softmax_logsumexp(s::SoftmaxState) =
    _row_zip((m,l) -> l == 0f0 ? -Inf32 : m+log(l),s.maximum,s.sum)
