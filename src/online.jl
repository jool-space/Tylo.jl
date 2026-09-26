"""
    SoftmaxState(fragment; dims=2)

FP32 running maxima and unnormalized exponential sums in the fragment's
reduced ownership. The empty state is `(-Inf, 0)`. Scores must be finite or
`-Inf` (masked). State arithmetic does not allocate storage or synchronize.
The state type depends on the axis, so `dims` must be a constant.
"""
struct SoftmaxState{D,R<:Fragment}
    maximum::R
    sum::R
    Base.@constprop :aggressive function SoftmaxState(maximum::R, sum::R; dims=2) where {R<:Fragment{Float32}}
        axis = _reduction_axis_argument(dims)
        _same_distribution(Layouts.layout(maximum),Layouts.layout(sum)) &&
            size(Layouts.layout(maximum))[axis] == 1 ||
            throw(ArgumentError("softmax statistics need matching reduced ownership"))
        new{axis,R}(maximum, sum)
    end
end
Base.@constprop :aggressive @inline function SoftmaxState(f; dims=2)
    reduced = _reduced_ownership(Layouts.layout(f),Val(_reduction_axis_argument(dims)))
    count = Val(_register_count(reduced))
    SoftmaxState(Fragment(ntuple(_ -> -Inf32,count),reduced),
                 Fragment(ntuple(_ -> 0f0,count),reduced); dims)
end

# Element operations select instead of branching: a per-element branch
# around the exponential costs more than the exponential itself. The
# exponential is `exp2` of a scaled argument, which the GPU evaluates with
# one `ex2` instruction (2 ulp) instead of a range-reduced series.
@inline _softmax_exp(x::Float32) = exp2(x * 1.442695f0)
# 2^x for weights, flushing subnormal results to zero: on the device the
# single `ex2.approx.ftz` instruction (CUDACoreExt), where exp2 would add a
# rescaling of arguments below -126 to every element.
@inline _ex2(x::Float32) = exp2(x)
@inline _softmax_exp(x) = exp(x)
@inline _softmax_rescale(old_maximum, new_maximum) =
    ifelse(old_maximum == -Inf32, 0f0, _softmax_exp(old_maximum - new_maximum))
@inline _softmax_weight(score, new_maximum) =
    ifelse(new_maximum == -Inf32, 0f0, _softmax_exp(score - new_maximum))

"""
    softmax_update(state, scores) -> (; state, weights, rescale)
    softmax_update(state, scores; scale) -> (; state, weights, rescale)

Update running statistics with a score tile, or with `scale · scores`
(`scale` positive; attention's `1/√d`). `weights` are unnormalized
`exp(scores - new_maximum)`, or `exp(scale · scores - new_maximum)`, in the
original fragment distribution.
Multiply an existing weighted numerator by `rescale` before adding this
tile's weighted values. Statistics use FP32 weights, before any conversion
for another MMA. Empty chunks produce zero weights; an empty previous state
has zero rescale. Distributed reductions require all lanes to participate.
The state retains the reduction axis selected by `SoftmaxState(scores; dims)`.

Without `scale`, each weight is `exp2((s - m) · log2 e)`, whose difference
is exact for nearby scores. With it, as in FlashAttention, each weight
costs one fused multiply-add and one `ex2`: the scale and the change to
base 2 fold into one factor, and a row that is entirely masked (maximum
`-Inf`) is handled once per row rather than per element. The exponent then
carries an absolute error of about `|scale · s| · 2⁻²³`, negligible for
attention's scaled scores, larger for scores in the thousands.
"""
@inline function softmax_update(state::SoftmaxState{D}, scores; scale::Union{Nothing,Float32}=nothing) where D
    scale === nothing || return _softmax_update(state,scores,scale)
    new_maximum = max.(state.maximum, maximum(scores; dims=D))
    rescale = _softmax_rescale.(state.maximum, new_maximum)
    weights = _softmax_weight.(scores, new_maximum)
    normalizer = muladd.(rescale, state.sum, sum(weights; dims=D))
    (; state=SoftmaxState(new_maximum, normalizer; dims=D), weights, rescale)
end
@inline function _softmax_update(state::SoftmaxState{D}, scores, scale::Float32) where D
    new_maximum = max.(state.maximum, maximum(scores; dims=D) .* scale)
    rescale = _softmax_rescale.(state.maximum, new_maximum)
    # exp(scale·s - m) = exp2(s·(scale·log2 e) - m·log2 e); a masked row's
    # offset is 0, so its -Inf scores give exp2(-Inf) = 0.
    factor = scale * 1.442695f0
    offset = map(m -> ifelse(m == -Inf32, 0f0, m * 1.442695f0), new_maximum)
    weights = ((s, o) -> _ex2(muladd(s, factor, -o))).(scores, offset)
    normalizer = muladd.(rescale, state.sum, sum(weights; dims=D))
    (; state=SoftmaxState(new_maximum, normalizer; dims=D), weights, rescale)
end

"""
    softmax_merge(left, right) -> (; state, left_rescale, right_rescale)

Combine independent summaries with identical reduced ownership. Combine their
unnormalized weighted numerators using the returned factors. Floating-point
merge order may change the result. Two empty summaries remain empty.
"""
@inline function softmax_merge(left_state::SoftmaxState{D,R}, right_state::SoftmaxState{D,R}) where {D,R}
    new_maximum = max.(left_state.maximum, right_state.maximum)
    left_rescale = _softmax_rescale.(left_state.maximum, new_maximum)
    right_rescale = _softmax_rescale.(right_state.maximum, new_maximum)
    normalizer = muladd.(left_rescale, left_state.sum, right_rescale .* right_state.sum)
    (; state=SoftmaxState(new_maximum, normalizer; dims=D), left_rescale, right_rescale)
end

"Normalize using one FP32 reciprocal per result and multiplication; empty rows return zero."
@inline function softmax_normalize(values, state::SoftmaxState)
    reciprocal = map(normalizer -> ifelse(normalizer == 0f0, 0f0, inv(normalizer)), state.sum)
    ((value, scale) -> ifelse(scale == 0f0, 0f0, value * scale)).(values, reciprocal)
end

"Final log-sum-exp in the state's reduced ownership; empty rows return -Inf."
@inline softmax_logsumexp(state::SoftmaxState) =
    ((running_maximum, normalizer) -> normalizer == 0f0 ? -Inf32 : running_maximum + log(normalizer)).(
        state.maximum, state.sum)
# The device's exp2 applies to each lane of a pair.
_pair_op(::typeof(_ex2)) = :lane
