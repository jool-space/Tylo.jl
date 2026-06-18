module ChainRulesCoreExt

using Tylo
import Tylo: shadow, _flex_attention, grad_shadow, ScoreMod

using ChainRulesCore
import ChainRulesCore: rrule, unthunk, NoTangent

# `rms_norm(X, W; eps, offset)` -> Y, differentiated w.r.t. X and W. Checkpoints
# (the forward→backward bridge) and the backward's scratch are each allocated
# once up front and reused by the pullback, so a captured graph reuses the memory.
function rrule(::typeof(rms_norm), X, W; eps, offset = 0.0f0)
    checkpoints = allocate_checkpoints(rms_norm!, X, W)
    Y = similar(X)
    rms_norm!(Y, X, W; eps, offset, checkpoints)

    scratch = allocate_scratchspace(∇rms_norm!, X, W)
    function rms_norm_pullback(Ȳ)
        Ȳ = unthunk(Ȳ)
        Xd = Duplicated(X)
        Wd = Duplicated(W)
        Yd = Duplicated(Y, Ȳ)
        ∇rms_norm!(Yd, Xd, Wd; offset, checkpoints, scratch)
        return (NoTangent(), shadow(Xd), shadow(Wd))
    end

    return Y, rms_norm_pullback
end

# `layer_norm(X, W, B; eps)` -> Y, differentiated w.r.t. X, W and B.
function rrule(::typeof(layer_norm), X, W, B; eps)
    checkpoints = allocate_checkpoints(layer_norm!, X, W, B)
    Y = similar(X)
    layer_norm!(Y, X, W, B; eps, checkpoints)

    scratch = allocate_scratchspace(∇layer_norm!, X, W, B)
    function layer_norm_pullback(Ȳ)
        Ȳ = unthunk(Ȳ)
        Xd = Duplicated(X)
        Wd = Duplicated(W)
        Bd = Duplicated(B)
        Yd = Duplicated(Y, Ȳ)
        ∇layer_norm!(Yd, Xd, Wd, Bd; checkpoints, scratch)
        return (NoTangent(), shadow(Xd), shadow(Wd), shadow(Bd))
    end

    return Y, layer_norm_pullback
end

# `softmax(X)` -> Y, differentiated w.r.t. X. No checkpoints/scratch — the
# backward is computed from the output Y (closed over by the pullback).
function rrule(::typeof(softmax), X)
    Y = softmax(X)

    function softmax_pullback(Ȳ)
        Xd = Duplicated(X)
        ∇softmax!(Duplicated(Y, unthunk(Ȳ)), Xd)
        return (NoTangent(), shadow(Xd))
    end

    return Y, softmax_pullback
end

# `attention(Q, K, V; causal, …)` -> O, differentiated w.r.t. Q/K/V.
# Checkpoints (M/L) and the backward scratch are allocated once and reused.
function rrule(::typeof(attention), Q, K, V; causal = false, kwargs...)
    checkpoints = allocate_checkpoints(attention!, Q, K, V)
    O = similar(Q, size(V, 1), size(Q, 2), size(Q, 3), size(Q, 4))
    attention!(O, Q, K, V; causal, checkpoints, kwargs...)

    scratch = allocate_scratchspace(∇attention!, Q, K, V, O)
    function attention_pullback(Ō)
        Ō = unthunk(Ō)
        Qd = Duplicated(Q)
        Kd = Duplicated(K)
        Vd = Duplicated(V)
        ∇attention!(Duplicated(O, Ō), Qd, Kd, Vd; checkpoints, scratch, causal, kwargs...)
        return (NoTangent(), shadow(Qd), shadow(Kd), shadow(Vd))
    end

    return O, attention_pullback
end

# `flex_attention(Q, K, V; score_mod, mask_mod, …)` -> O — the kwarg wrapper,
# differentiated w.r.t. Q/K/V only (mods fixed config). This is the path Mooncake
# uses (it can bridge array tangents but not the score-mod struct Tangent). For
# `score_mod`-parameter gradients use `_flex_attention` (ChainRules) below.
function rrule(::typeof(flex_attention), Q, K, V;
              score_mod = NoOpScore(), mask_mod = FullMask(), kwargs...)
    checkpoints = allocate_checkpoints(flex_attention!, Q, K, V)
    O = similar(Q, size(V, 1), size(Q, 2), size(Q, 3), size(Q, 4))
    flex_attention!(O, Q, K, V; score_mod, mask_mod, checkpoints, kwargs...)

    scratch = allocate_scratchspace(∇flex_attention!, Q, K, V, O)
    function flex_attention_pullback(Ō)
        Ō = unthunk(Ō)
        Qd = Duplicated(Q)
        Kd = Duplicated(K)
        Vd = Duplicated(V)
        ∇flex_attention!(Duplicated(O, Ō), Qd, Kd, Vd;
                         score_mod, mask_mod, checkpoints, scratch, kwargs...)
        return (NoTangent(), shadow(Qd), shadow(Kd), shadow(Vd))
    end

    return O, flex_attention_pullback
end

# Recursively convert an accumulated `grad_shadow` (same struct type as the mod,
# array fields holding grads, non-array fields junk) into a ChainRules `Tangent`.
_score_tangent(x::AbstractArray) = x
_score_tangent(m::ScoreMod) =
    Tangent{typeof(m)}(; (fn => _score_tangent(getfield(m, fn)) for fn in fieldnames(typeof(m)))...)
_score_tangent(::Any) = NoTangent()

# `_flex_attention(Q, K, V, score_mod; mask_mod, …)` -> O — the positional core,
# differentiated w.r.t. Q/K/V AND `score_mod`'s parameters (its `grad_shadow` is
# accumulated by ∇flex_attention!, then mapped to a Tangent). `mask_mod` stays
# fixed config (NoTangent). The kwarg `flex_attention` routes here, so autodiff
# through it differentiates the variant's parameters too.
function rrule(::typeof(_flex_attention), Q, K, V, score_mod;
              mask_mod = FullMask(), kwargs...)
    checkpoints = allocate_checkpoints(flex_attention!, Q, K, V)
    O = similar(Q, size(V, 1), size(Q, 2), size(Q, 3), size(Q, 4))
    flex_attention!(O, Q, K, V; score_mod, mask_mod, checkpoints, kwargs...)

    scratch = allocate_scratchspace(∇flex_attention!, Q, K, V, O)
    function _flex_attention_pullback(Ō)
        Ō = unthunk(Ō)
        ∂score_mod = grad_shadow(score_mod)
        Qd = Duplicated(Q)
        Kd = Duplicated(K)
        Vd = Duplicated(V)
        ∇flex_attention!(Duplicated(O, Ō), Qd, Kd, Vd;
                         score_mod, mask_mod, ∂score_mod, checkpoints, scratch, kwargs...)
        return (NoTangent(), shadow(Qd), shadow(Kd), shadow(Vd), _score_tangent(∂score_mod))
    end

    return O, _flex_attention_pullback
end

end
