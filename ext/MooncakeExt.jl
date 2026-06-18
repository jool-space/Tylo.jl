module MooncakeExt

using Tylo
using Mooncake

using Mooncake: DefaultCtx, @from_rrule, @is_primitive, CoDual, NoRData, NoFData, FData,
    primal, tangent, fdata, zero_tangent, lazy_zero_rdata, instantiate, increment!!
import Mooncake: rrule!!
import Tylo: _flex_attention_core, ScoreMod, MaskMod

# Mooncake gates rule registration on `Base.IEEEFloat` by default; we register
# over any `AbstractFloat` so BFloat16 (and other real-float storage types) flow
# through wherever Mooncake has a scalar tangent interface for them. (Note
# `ForwardDiff.Dual <: Real` but NOT `<: AbstractFloat`, so dual numbers are not
# accidentally captured.)
const ADFloat = AbstractFloat

# Register a Mooncake rule from the ChainRulesCore rrule using a real call
# SIGNATURE — `@tylo_from_rrule f(::T1, ::T2; kw...)` — instead of hand-writing a
# `Tuple{typeof(f), T1, T2}` type. The presence of a `;` parameters block (the
# kwarg names are documentation only) flags that `f` takes keyword arguments,
# forwarding the trailing flag `@from_rrule` expects.
macro tylo_from_rrule(call)
    Meta.isexpr(call, :call) || error("@tylo_from_rrule expects a call signature, got: $call")
    f = call.args[1]
    haskw = false
    argtypes = Any[]
    for a in call.args[2:end]
        if Meta.isexpr(a, :parameters)
            haskw = true                      # `; kw...` present → kwarg function
        elseif Meta.isexpr(a, :(::))
            push!(argtypes, last(a.args))     # `::T` → the annotated type
        else
            error("@tylo_from_rrule arguments must be `::Type`, got: $a")
        end
    end
    sig = Expr(:curly, :Tuple, :(typeof($f)), argtypes...)
    return haskw ? :(@from_rrule(DefaultCtx, $sig, true)) :
                   :(@from_rrule(DefaultCtx, $sig))
end

# Accumulate a `grad_shadow` (same struct as the mod, array fields = grads) into
# the mod's Mooncake fdata (`FData` of per-field accumulators), recursing through
# composed mods. Array leaves use `increment!!` (CuArray via MooncakeCUDAExt).
_accumulate_mod!(f::AbstractArray, g::AbstractArray) = (increment!!(f, g); nothing)
_accumulate_mod!(::NoFData, _) = nothing   # non-differentiable field (e.g. BiasScore.nheads)
function _accumulate_mod!(f::FData, ∂)
    foreach(k -> _accumulate_mod!(getfield(f.data, k), getfield(∂, k)), keys(f.data))
    return
end

# Native reverse rule for the positional flex core. Unlike `@from_rrule` (which
# zeroes struct tangents), this carries learnable `score_mod` parameters: Q/K/V
# grads accumulate into their fdata, and the score-mod `grad_shadow` accumulates
# into the mod's fdata. Reached by tracing `flex_attention`/`_flex_attention`.
@is_primitive(DefaultCtx,
    Tuple{typeof(_flex_attention_core), AbstractArray{<:ADFloat,4},
          AbstractArray{<:ADFloat,4}, AbstractArray{<:ADFloat,4},
          ScoreMod, MaskMod, NamedTuple})

function rrule!!(::CoDual{typeof(_flex_attention_core)},
                 Q::CoDual, K::CoDual, V::CoDual, sm::CoDual, mm::CoDual, cfg::CoDual)
    pQ, pK, pV = primal(Q), primal(K), primal(V)
    psm, pmm, pcfg = primal(sm), primal(mm), primal(cfg)

    checkpoints = Tylo.allocate_checkpoints(flex_attention!, pQ, pK, pV)
    O = similar(pQ, size(pV, 1), size(pQ, 2), size(pQ, 3), size(pQ, 4))
    flex_attention!(O, pQ, pK, pV; score_mod = psm, mask_mod = pmm, checkpoints, pcfg...)
    scratch = Tylo.allocate_scratchspace(∇flex_attention!, pQ, pK, pV, O)

    Ō = fdata(zero_tangent(O))                       # downstream accumulates the output cotangent
    lzr = map(lazy_zero_rdata, (pQ, pK, pV, psm, pmm, pcfg))

    function flex_core_pb!!(::NoRData)
        Qd = Duplicated(pQ, similar(pQ))
        Kd = Duplicated(pK, similar(pK))
        Vd = Duplicated(pV, similar(pV))
        ∂sm = grad_shadow(psm)
        ∇flex_attention!(Duplicated(O, Ō), Qd, Kd, Vd;
            score_mod = psm, mask_mod = pmm, ∂score_mod = ∂sm, checkpoints, scratch, pcfg...)
        increment!!(tangent(Q), Tylo.shadow(Qd))
        increment!!(tangent(K), Tylo.shadow(Kd))
        increment!!(tangent(V), Tylo.shadow(Vd))
        _accumulate_mod!(tangent(sm), ∂sm)
        return (NoRData(), map(instantiate, lzr)...)
    end

    return CoDual(O, Ō), flex_core_pb!!
end

# Reuse the ChainRulesCore rrules (see ChainRulesCoreExt) from Mooncake. Loaded
# only when both Mooncake and ChainRulesCore are present (see Project.toml).
@tylo_from_rrule rms_norm(::AbstractMatrix{<:ADFloat}, ::AbstractVector{<:ADFloat}; eps, offset)

@tylo_from_rrule layer_norm(::AbstractMatrix{<:ADFloat}, ::AbstractVector{<:ADFloat},
                          ::AbstractVector{<:ADFloat}; eps)

@tylo_from_rrule softmax(::AbstractMatrix{<:ADFloat})

@tylo_from_rrule attention(::AbstractArray{<:ADFloat,4}, ::AbstractArray{<:ADFloat,4},
                         ::AbstractArray{<:ADFloat,4}; causal)

# flex via the kwarg wrapper: differentiates Q/K/V only, mods fixed config — the
# cheap path. (`@from_rrule` can't carry the score-mod struct tangent; for
# learnable score_mod use `_flex_attention`, which routes to the native `rrule!!`
# on `_flex_attention_core` above.)
@tylo_from_rrule flex_attention(::AbstractArray{<:ADFloat,4}, ::AbstractArray{<:ADFloat,4},
                              ::AbstractArray{<:ADFloat,4}; score_mod, mask_mod)

end
