"""
    SoftmaxState(fragment_or_ownership)

FP32 running row maxima and unnormalized exponential sums in the fragment's
row distribution. The empty state is `(-Inf, 0)`. Scores must be finite or
`-Inf` (masked). State arithmetic does not allocate storage or synchronize.
"""
struct SoftmaxState{R<:RowValues}
    maximum::R
    sum::R
    function SoftmaxState(maximum::R, sum::R) where {O,N,R<:RowValues{O,N,Float32}}
        new{R}(maximum, sum)
    end
end
@inline function SoftmaxState(ownership::RowOwnership)
    row_count = Val(_row_count(ownership))
    SoftmaxState(
        RowValues(ownership, ntuple(_ -> -Inf32, row_count)),
        RowValues(ownership, ntuple(_ -> 0f0, row_count)),
    )
end
@inline SoftmaxState(f) = SoftmaxState(row_ownership(f))
row_ownership(state::SoftmaxState) = row_ownership(state.maximum)

@inline _softmax_rescale(old_maximum, new_maximum) =
    old_maximum == -Inf32 ? 0f0 : exp(old_maximum - new_maximum)
@inline _softmax_weight(score, new_maximum) =
    new_maximum == -Inf32 ? 0f0 : exp(score - new_maximum)

"""
    softmax_update(state, scores) -> (; state, weights, rescale)

Update running statistics with a score tile. `weights` are unnormalized
`exp(scores - new_maximum)` in the original fragment distribution. Multiply
an existing weighted numerator by `rescale` before adding this tile's weighted
values. Statistics use FP32 weights, before any conversion for another MMA.
Empty chunks produce zero weights; an empty previous state has zero rescale.
Warp/MMA fragments require all lanes to participate in their row collectives.
"""
@inline function softmax_update(state::SoftmaxState, scores)
    new_maximum = max.(state.maximum, maximum(scores; dims=2))
    rescale = _softmax_rescale.(state.maximum, new_maximum)
    weights = _softmax_weight.(scores, new_maximum)
    normalizer = muladd.(rescale, state.sum, sum(weights; dims=2))
    (; state=SoftmaxState(new_maximum, normalizer), weights, rescale)
end

"""
    softmax_merge(left, right) -> (; state, left_rescale, right_rescale)

Combine independent summaries with identical row ownership. Combine their
unnormalized weighted numerators using the returned factors. Floating-point
merge order may change the result. Two empty summaries remain empty.
"""
@inline function softmax_merge(left_state::SoftmaxState{R}, right_state::SoftmaxState{R}) where R
    new_maximum = max.(left_state.maximum, right_state.maximum)
    left_rescale = _softmax_rescale.(left_state.maximum, new_maximum)
    right_rescale = _softmax_rescale.(right_state.maximum, new_maximum)
    normalizer = muladd.(left_rescale, left_state.sum, right_rescale .* right_state.sum)
    (; state=SoftmaxState(new_maximum, normalizer), left_rescale, right_rescale)
end

"Normalize using one FP32 reciprocal per row and multiplication; empty rows return zero."
@inline function softmax_normalize(values, state::SoftmaxState)
    reciprocal = map(normalizer -> normalizer == 0f0 ? 0f0 : inv(normalizer), state.sum)
    ((value, scale) -> scale == 0f0 ? 0f0 : value * scale).(values, reciprocal)
end

"Final log-sum-exp in the state's row distribution; empty rows return -Inf."
@inline softmax_logsumexp(state::SoftmaxState) =
    ((row_maximum, normalizer) -> normalizer == 0f0 ? -Inf32 : row_maximum + log(normalizer)).(
        state.maximum, state.sum)
