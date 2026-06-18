export PairFeatureScore, pair_feature, ∇pair_feature

# Fused pair-feature bias — attention bias computed on the fly from per-position
# feature vectors, fused into the flex forward as a ScoreMod (no kernel changes):
#
#   s[kv, q] += Σₚ pair_proj[h, p] · φₚ          φ = pair_feature(op, qvals, kvals)
#
# with qvals = q_features[:, q, b] and kvals = k_features[:, kv, b]. This covers
# pair biases that PyTorch FlexAttention can't express without materializing the
# (N, M) bias: distance/RBF features from coordinates, learned relative features…
#
# `pair_feature` is the user extension point. Write it with DOT BROADCASTS over
# the entries of `qvals`/`kvals` and the one definition runs in both worlds:
#   * device: entries are (1, TILE_M) / (TILE_N, 1) feature-row tiles, so each
#     φₚ broadcasts outer-product style to a (TILE_N, TILE_M) tile
#   * host:   entries are scalars, so each φₚ is a scalar (CPU references via
#     `hscore(adapt(Array, mod), …)`)
#
# This is BlockPos earning its keep a second time (after BiasScore): the mod
# reads `block_idx` to block-load each feature row coalesced — a gather on the
# materialized position tile would be per-element.

"""
    pair_feature(op, qvals::NTuple{F}, kvals::NTuple{F}) -> NTuple{PD}

Compute `PD` pair features from the per-position feature values of one
(query, key) pair. Implement this for your `op` type using broadcast dots
(`.+`, `.*`, `exp.`, …) over the entries — the entries are scalars on the
host and tiles on the device, and broadcasting lifts the same definition to
both. `PD` must equal `size(pair_proj, 2)` of the [`PairFeatureScore`](@ref).

NO generic fallback method on purpose: an `error("…\$(typeof(op))…")` fallback
drags vararg `string` MethodInstances into device-code inference, which
cuTile's compiler cache cannot handle (lattice error on `Vararg` argtypes).
A missing implementation surfaces as a plain MethodError instead.
"""
function pair_feature end

"""
    PairFeatureScore(op, q_features, k_features, pair_proj)

Adds a per-head projection of [`pair_feature`](@ref) outputs to the scores.
Features are indexed by local query position (`input_pos` does not shift them).

  * `q_features`: `(F, SeqLen_Q, Batch)`
  * `k_features`: `(F, SeqLen_K, Batch)`
  * `pair_proj`: `(Heads, PD)`
"""
struct PairFeatureScore{F, PD, Op, AQ, AK, P} <: ScoreMod
    nfeat::Val{F}       # feature rows F, static so device loops unroll
    npair::Val{PD}      # pair-feature outputs PD = size(pair_proj, 2)
    op::Op
    q_features::AQ      # (F, SeqLen_Q, Batch)
    k_features::AK      # (F, SeqLen_K, Batch)
    pair_proj::P        # (Heads, PD)
end

function PairFeatureScore(op, q_features::AbstractArray, k_features::AbstractArray,
                          pair_proj::AbstractMatrix)
    F = Int(size(q_features, 1))
    size(k_features, 1) == F || throw(DimensionMismatch(
        "q_features has $F feature rows, k_features has $(size(k_features, 1))"))
    PairFeatureScore(Val(F), Val(Int(size(pair_proj, 2))),
                     op, q_features, k_features, pair_proj)
end

# Unrolled broadcast sum of a tuple (no varargs — cuTile inference caching
# rejects Vararg argtypes).
@inline _bsum(t::Tuple{Any}) = t[1]
@inline _bsum(t::Tuple) = t[1] .+ _bsum(Base.tail(t))

# Device path: one extent-1-row block load per feature (F and PD are type
# parameters, so the loops unroll and the load shapes stay static).
@inline function (m::PairFeatureScore{F, PD})(s, b, h, q::BlockPos, kv::BlockPos) where {F, PD}
    padding_mode = ct.PaddingMode.Zero
    qvals = ntuple(Val(F)) do f
        ct.load(m.q_features, (f, q.block_idx + 1i32, b), (1, tile_size(q)); padding_mode)
    end
    kvals = ntuple(Val(F)) do f
        t = ct.load(m.k_features, (f, kv.block_idx + 1i32, b), (1, tile_size(kv)); padding_mode)
        reshape(t, (tile_size(kv), 1))
    end
    phi = pair_feature(m.op, qvals, kvals)
    bias = _bsum(ntuple(p -> m.pair_proj[h, p] .* phi[p], Val(PD)))
    # `convert`, not `eltype(s).(bias)`: the broadcast cast of a broadcast
    # RESULT trips a cuTile structurizer bug (QuoteNode BroadcastStyle yields).
    s .+ convert(Tile{eltype(s)}, bias)
end

# Host path (CPU references): same op on scalars, 0-based positions.
function (m::PairFeatureScore{F, PD})(s, b, h, q::Integer, kv::Integer) where {F, PD}
    qvals = ntuple(f -> m.q_features[f, q + 1, b], Val(F))
    kvals = ntuple(f -> m.k_features[f, kv + 1, b], Val(F))
    phi = pair_feature(m.op, qvals, kvals)
    s + sum(ntuple(p -> m.pair_proj[h, p] * phi[p], Val(PD)))
end

## --- backward ----------------------------------------------------------------

"""
    ∇pair_feature(op, qvals, kvals, dphi::NTuple{PD}) -> (dq, dk)

VJP of [`pair_feature`](@ref): given cotangents `dphi` for the `PD` pair
features, return cotangent tuples for `qvals` and `kvals` (length `F` each).
Write it in the same broadcast style as `pair_feature`; each entry must be a
broadcast result involving the inputs (`0f0 .* qvals[f]` for a zero gradient,
not a bare scalar).

Only required with `grad_shadow(m; feature_grads = true)`; projection
gradients never need it.
"""
function ∇pair_feature end

# Feature gradients are opt-in: they cost an extra `∇pair_feature` evaluation
# plus 2F atomic row-adds per block pair, and need `∇pair_feature` implemented.
grad_shadow(m::PairFeatureScore; feature_grads::Bool = false) = PairFeatureScore(
    m.nfeat, m.npair, m.op,
    feature_grads ? fill!(similar(m.q_features), 0) : nothing,
    feature_grads ? fill!(similar(m.k_features), 0) : nothing,
    fill!(similar(m.pair_proj), 0))

@inline function ∇score(m::PairFeatureScore{F, PD}, ∂m, s, b, h,
                        q::BlockPos, kv::BlockPos, s̄) where {F, PD}
    isnothing(∂m) && return s̄
    padding_mode = ct.PaddingMode.Zero
    qvals = ntuple(Val(F)) do f
        ct.load(m.q_features, (f, q.block_idx + 1i32, b), (1, tile_size(q)); padding_mode)
    end
    kvals = ntuple(Val(F)) do f
        t = ct.load(m.k_features, (f, kv.block_idx + 1i32, b), (1, tile_size(kv)); padding_mode)
        reshape(t, (tile_size(kv), 1))
    end
    phi = pair_feature(m.op, qvals, kvals)

    # ∂proj[h, p] += Σ s̄ ∘ φₚ  (full reduce, one scalar atomic per p)
    ntuple(Val(PD)) do p
        g = sum(sum(s̄ .* phi[p], dims=1), dims=2)
        atomic_add_tile(∂m.pair_proj, (h, p),
                        convert(Tile{eltype(∂m.pair_proj)}, g))
        nothing
    end

    if !isnothing(∂m.q_features)
        dphi = ntuple(p -> m.pair_proj[h, p] .* s̄, Val(PD))
        dq, dk = ∇pair_feature(m.op, qvals, kvals, dphi)
        ntuple(Val(F)) do f
            gq = sum(dq[f], dims=1)                       # reduce over kv axis
            atomic_add_tile(∂m.q_features, (f, q.block_idx + 1i32, b),
                            convert(Tile{eltype(∂m.q_features)}, gq))
            gk = reshape(sum(dk[f], dims=2), (1, tile_size(kv)))  # over q axis
            atomic_add_tile(∂m.k_features, (f, kv.block_idx + 1i32, b),
                            convert(Tile{eltype(∂m.k_features)}, gk))
            nothing
        end
    end
    s̄
end
