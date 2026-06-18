export flex_attention
export flex_attention!
export ∇flex_attention
export ∇flex_attention!

# FlexAttention — variant-agnostic forward kernel; the variant (score_mod /
# mask_mod, see mods.jl) is chosen by TYPE, so each combination compiles to a
# specialized kernel with the mod inlined.
#
# Layout (column-major), matching `attention!`:
#   Q (Dk, SeqLen_Q, Heads, Batch)
#   K (Dk, SeqLen_K, Heads_KV, Batch)
#   V (Dv, SeqLen_K, Heads_KV, Batch)
#   O (Dv, SeqLen_Q, Heads, Batch)
#
# Two SEPARATE single-loop kernels (analytic range vs BlockMask walk): a
# bisection showed no individual heavy construct is at fault, but co-locating
# BOTH loop bodies in one kernel tips a cuTile codegen/structurizer threshold
# (nondeterministic miscompiles). Each kernel carries the body once.

# In-kernel analytic block range + full/partial split (mods.jl lattice).
function flex_fwd(
    Q::TileArray4, K::TileArray4, V::TileArray4, O::TileArray4,
    M::Optional{TileArray3{Float32}}, L::Optional{TileArray3{Float32}},
    score_mod, mask_mod,
    qk_scale::Float32, input_pos::Int32,
    H::Int,
    Tc::Type,
    Dk::Int, Dv::Int,
    TILE_M::Int, TILE_N::Int,
    QUERY_GROUP_SIZE::Int,
    EVEN_K::Bool, BLOCK_SPARSE::Bool,
)
    padding_mode = ct.PaddingMode.Zero
    i, hb = ct.bid(1), ct.bid(2)
    b, h = fldmod1(hb, H)
    hₖ = fld1(h, QUERY_GROUP_SIZE)

    # 0-based positions, as lazy descriptors. Mods broadcast through
    # Broadcast.broadcastable; BiasScore reads block_idx for ct.load.
    q_pos = BlockPos{2, TILE_M}(i - 1i32, input_pos)

    m_i = fill(-Inf32, (1, TILE_M))
    l_i = zeros(Float32, (1, TILE_M))
    acc = zeros(Float32, (Dv, TILE_M))

    q = ct.load(Q, (1, i, h, b), (Dk, TILE_M); padding_mode)

    q_len = size(Q, 2)
    k_len = size(K, 2)
    num_kv = cld(k_len, Int32(TILE_N))
    iq = i - 1i32                             # 0-based query block
    TMi = Int32(TILE_M); TNi = Int32(TILE_N)

    if BLOCK_SPARSE
        lo, hi = kv_block_range(mask_mod, iq, TMi, TNi, q_len, k_len, input_pos)
        lo = max(lo, 0i32)
        hi = min(hi, num_kv)
    else
        lo = 0i32; hi = num_kv
    end

    for j in lo:hi-1i32
        k = ct.load(K, (1, j + 1i32, hₖ, b), (Dk, TILE_N); padding_mode, latency=2)

        s = muladd((k)ᵀ → Tc, q → Tc, zeros(Float32, (TILE_N, TILE_M)))
        s = s * Float32(qk_scale)                # mods see true scores S = QKᵀ/√dₖ

        kv_pos = BlockPos{1, TILE_N}(j)
        s = score_mod(s, b, h, q_pos, kv_pos)

        if BLOCK_SPARSE && kv_block_full(mask_mod, iq, j, TMi, TNi, q_len, k_len, input_pos)
            # provably no element masked ⇒ skip mask_mod (score_mod only).
            if !EVEN_K
                s = ifelse.(kv_pos .< k_len, s, Float32(-Inf32))
            end
        else
            umask = mask_mod(b, h, q_pos, kv_pos)
            if !EVEN_K
                umask = umask .& (kv_pos .< k_len)
            end
            s = ifelse.(umask, s, Float32(-Inf32))
        end

        m_ij = max.(m_i, maximum(s, dims=1))
        safe = m_ij .> -Inf32                 # fully-masked column ⇒ keep -Inf state
        p = ifelse.(safe, exp.(s .- m_ij), 0f0)
        l_ij = sum(p, dims=1)
        alpha = ifelse.(safe, exp.(m_i .- m_ij), 1f0)
        l_i = l_i .* alpha .+ l_ij
        acc = acc .* Float32.(alpha)

        v = ct.load(V, (1, j + 1i32, hₖ, b), (Dv, TILE_N); padding_mode, latency=4)
        acc = muladd(v → Tc, p → Tc, acc)

        m_i = m_ij
    end

    o = acc .* ifelse.(l_i .== 0f0, 0f0, 1f0 ./ l_i)
    ct.store(O, (1, i, h, b), o → eltype(O))
    isnothing(M) || ct.store(M, (i, h, b), reshape(m_i, TILE_M))
    isnothing(L) || ct.store(L, (i, h, b), reshape(l_i, TILE_M))

    return
end

# Walk a precomputed coarse BlockMask index list (mods.jl `build_block_mask`).
function flex_fwd_bm(
    Q::TileArray4, K::TileArray4, V::TileArray4, O::TileArray4,
    M::Optional{TileArray3{Float32}}, L::Optional{TileArray3{Float32}},
    bm_count::TileVector{Int32},
    bm_idx::TileMatrix{Int32},
    bm_full::TileMatrix{Int32},
    score_mod, mask_mod,
    qk_scale::Float32, input_pos::Int32,
    H::Int,
    Tc::Type,
    Dk::Int, Dv::Int,
    TILE_M::Int, TILE_N::Int,
    QUERY_GROUP_SIZE::Int,
    EVEN_K::Bool,
)
    padding_mode = ct.PaddingMode.Zero
    i, hb = ct.bid(1), ct.bid(2)
    b, h = fldmod1(hb, H)
    hₖ = fld1(h, QUERY_GROUP_SIZE)

    q_pos = BlockPos{2, TILE_M}(i - 1i32, input_pos)

    m_i = fill(-Inf32, (1, TILE_M))
    l_i = zeros(Float32, (1, TILE_M))
    acc = zeros(Float32, (Dv, TILE_M))

    q = ct.load(Q, (1, i, h, b), (Dk, TILE_M); padding_mode)
    k_len = size(K, 2)

    nb = bm_count[i]
    for t in 1i32:nb
        j = bm_idx[t, i]                      # 0-based KV block
        is_full = bm_full[t, i] != 0i32

        k = ct.load(K, (1, j + 1i32, hₖ, b), (Dk, TILE_N); padding_mode, latency=2)

        s = muladd((k)ᵀ → Tc, q → Tc, zeros(Float32, (TILE_N, TILE_M)))
        s = s * Float32(qk_scale)

        kv_pos = BlockPos{1, TILE_N}(j)
        s = score_mod(s, b, h, q_pos, kv_pos)

        if is_full
            if !EVEN_K
                s = ifelse.(kv_pos .< k_len, s, Float32(-Inf32))
            end
        else
            umask = mask_mod(b, h, q_pos, kv_pos)
            if !EVEN_K
                umask = umask .& (kv_pos .< k_len)
            end
            s = ifelse.(umask, s, Float32(-Inf32))
        end

        m_ij = max.(m_i, maximum(s, dims=1))
        safe = m_ij .> -Inf32
        p = ifelse.(safe, exp.(s .- m_ij), 0f0)
        l_ij = sum(p, dims=1)
        alpha = ifelse.(safe, exp.(m_i .- m_ij), 1f0)
        l_i = l_i .* alpha .+ l_ij
        acc = acc .* Float32.(alpha)

        v = ct.load(V, (1, j + 1i32, hₖ, b), (Dv, TILE_N); padding_mode, latency=4)
        acc = muladd(v → Tc, p → Tc, acc)

        m_i = m_ij
    end

    o = acc .* ifelse.(l_i .== 0f0, 0f0, 1f0 ./ l_i)
    ct.store(O, (1, i, h, b), o → eltype(O))
    isnothing(M) || ct.store(M, (i, h, b), reshape(m_i, TILE_M))
    isnothing(L) || ct.store(L, (i, h, b), reshape(l_i, TILE_M))

    return
end

#==============================================================================
 Backward — mha_bwd's structure (grid = Heads·Batch, outer KV loop, inner Q
 loop, Q̄ read-modify-write, register-accumulated K̄/V̄) with the variant hooks:
 score_mod re-applied for P, its VJP (`∇score`, mods.jl) for s̄, mask_mod in
 place of CAUSAL. Analytic block sparsity reuses the forward lattice
 transposed: for fixed kv block j, query block i is skipped when
 j ∉ kv_block_range(i) — sound because the range is a superset of touched
 blocks, so skipped pairs are fully masked (S̄ ≡ 0).

 The precomputed-BlockMask path has no backward yet (its index list is
 per-query-block; the outer-KV loop would need the transpose).
==============================================================================#

function flex_bwd(
    Q::TileArray4, K::TileArray4, V::TileArray4,
    Ō′::TileArray4,
    M::TileArray3{Float32},
    Δ::TileArray3{Float32},
    Q̄::TileArray4, K̄::TileArray4, V̄::TileArray4,
    score_mod, mask_mod, ∂score_mod,
    qk_scale::Float32, input_pos::Int32,
    H::Int,
    Tc::Type,
    Dk::Int, Dv::Int,
    TILE_M::Int, TILE_N::Int,
    QUERY_GROUP_SIZE::Int,
    EVEN_K::Bool, BLOCK_SPARSE::Bool,
)
    padding_mode = ct.PaddingMode.Zero
    hb = ct.bid(1)
    b, h = fldmod1(hb, H)
    hₖ = fld1(h, QUERY_GROUP_SIZE)

    q_len = size(Q, 2)
    k_len = size(K, 2)
    q_tiles = cld(q_len, Int32(TILE_M))
    kv_tiles = cld(k_len, Int32(TILE_N))
    TMi = Int32(TILE_M); TNi = Int32(TILE_N)

    for j in 0i32:kv_tiles-1i32
        k = ct.load(K, (1, j + 1i32, hₖ, b), (Dk, TILE_N); padding_mode)
        v = ct.load(V, (1, j + 1i32, hₖ, b), (Dv, TILE_N); padding_mode)

        k̄_acc = zeros(Float32, (Dk, TILE_N))
        v̄_acc = zeros(Float32, (Dv, TILE_N))

        kv_pos = BlockPos{1, TILE_N}(j)

        for i in 1i32:q_tiles
            iq = i - 1i32
            if BLOCK_SPARSE
                lo, hi = kv_block_range(mask_mod, iq, TMi, TNi, q_len, k_len, input_pos)
                visit = (j >= lo) & (j < hi)
            else
                visit = true
            end
            if visit
                q = ct.load(Q, (1, i, h, b), (Dk, TILE_M); padding_mode, allow_tma=false)
                ō = ct.load(Ō′, (1, i, h, b), (Dv, TILE_M); padding_mode, allow_tma=false)
                m = reshape(ct.load(M, (i, h, b), (TILE_M,), latency=1), (1, TILE_M))
                δ = reshape(ct.load(Δ, (i, h, b), (TILE_M,), latency=1), (1, TILE_M))

                s₀ = muladd((k)ᵀ → Tc, q → Tc, zeros(Float32, (TILE_N, TILE_M)))
                s₀ = s₀ * Float32(qk_scale)          # pre-mod scores — VJPs see these

                q_pos = BlockPos{2, TILE_M}(iq, input_pos)
                s = score_mod(s₀, b, h, q_pos, kv_pos)

                if BLOCK_SPARSE && kv_block_full(mask_mod, iq, j, TMi, TNi, q_len, k_len, input_pos)
                    if !EVEN_K
                        s = ifelse.(kv_pos .< k_len, s, Float32(-Inf32))
                    end
                else
                    umask = mask_mod(b, h, q_pos, kv_pos)
                    if !EVEN_K
                        umask = umask .& (kv_pos .< k_len)
                    end
                    s = ifelse.(umask, s, Float32(-Inf32))
                end

                safe = m .> -Inf32                # fully-masked column ⇒ p = 0
                p = ifelse.(safe, exp.(s .- Float32.(m)), 0f0)

                v̄_acc = muladd(ō → Tc, (p)ᵀ → Tc, v̄_acc)

                p̄ = muladd((v)ᵀ → Tc, ō → Tc, zeros(Float32, (TILE_N, TILE_M)))
                ds = p .* (p̄ .- Float32.(δ))         # S̄ w.r.t. post-mod scores

                s̄ = ∇score(score_mod, ∂score_mod, s₀, b, h, q_pos, kv_pos, ds)
                s̄ = s̄ * Float32(qk_scale)

                q̄ = ct.load(Q̄, (1, i, h, b), (Dk, TILE_M), allow_tma=false)
                q̄ = muladd(k → Tc, s̄ → Tc, q̄ → Float32)
                ct.store(Q̄, (1, i, h, b), q̄ → eltype(Q̄))

                k̄_acc = muladd(q → Tc, (s̄)ᵀ → Tc, k̄_acc)
            end
        end

        store = isone(QUERY_GROUP_SIZE) ? ct.store : atomic_add_tile
        store(K̄, (1, j + 1i32, hₖ, b), k̄_acc → eltype(K̄))
        store(V̄, (1, j + 1i32, hₖ, b), v̄_acc → eltype(V̄))
    end

    return
end

"""
    flex_attention!(O, Q, K, V; score_mod = NoOpScore(), mask_mod = FullMask(), checkpoints = nothing, kwargs...)

FlexAttention forward: fused multi-head attention whose variant is given by two
mods — `score_mod` rewrites the attention scores and `mask_mod` decides which
query-key pairs attend.

  * `Q`: `(Dk, SeqLen_Q, Heads, Batch)`
  * `K`: `(Dk, SeqLen_K, Heads_KV, Batch)`
  * `V`: `(Dv, SeqLen_K, Heads_KV, Batch)`
  * `O`: `(Dv, SeqLen_Q, Heads, Batch)`

`Heads` must be a multiple of `Heads_KV` (GQA). `O`/`Q`/`K`/`V` may be bare arrays
or [`Duplicated`](@ref) (only the primals are read). Pass `checkpoints` from
[`allocate_checkpoints`](@ref)`(flex_attention!, Q, K, V)` to save the softmax
statistics `M`/`L` for [`∇flex_attention!`](@ref).

Keywords: `block_mask`, a precomputed [`BlockMask`](@ref); `block_sparse = true`,
in-kernel block skipping for analytic masks; `input_pos = 0`; `qk_scale = 1/√Dk`;
`tensorcore` compute precision (Float32 accumulation); `TILE_M`/`TILE_N`. Returns `nothing`.
"""
function flex_attention!(O,
    Q, K, V;
    score_mod = NoOpScore(),
    mask_mod = FullMask(),
    block_mask::Optional{BlockMask} = nothing,
    block_sparse::Bool = true,
    input_pos::Integer = 0,
    qk_scale = nothing,
    tensorcore = tensorcore_type(eltype(primal(Q))),
    TILE_M = 64,
    TILE_N = 64,
    checkpoints = nothing,
)
    pO, pQ, pK, pV = primal(O), primal(Q), primal(K), primal(V)
    Dq, SeqLen_Q, Heads, Batch = size(pQ)
    Dk, SeqLen_K, Heads_KV, Batch_K = size(pK)
    Dv, SeqLen_V, Heads_V, Batch_V = size(pV)
    @assert Dq == Dk
    @assert SeqLen_K == SeqLen_V
    @assert Heads_KV == Heads_V
    @assert Batch == Batch_K == Batch_V
    @assert size(pO) == (Dv, SeqLen_Q, Heads, Batch)
    @assert iszero(Heads % Heads_KV)

    query_group_size = Heads ÷ Heads_KV
    qk_scale = Float32(something(qk_scale, 1 / sqrt(Dk)))
    even_k = iszero(SeqLen_K % TILE_N)
    Dk_pow2 = nextpow(2, Dk)
    Dv_pow2 = nextpow(2, Dv)

    M = checkpoints === nothing ? nothing : get(checkpoints, :M, nothing)
    L = checkpoints === nothing ? nothing : get(checkpoints, :L, nothing)

    grid = (cld(SeqLen_Q, TILE_M), Heads * Batch)

    if block_mask isa BlockMask
        @cutile(blocks=grid,
            flex_fwd_bm(
                pQ, pK, pV, pO, M, L,
                block_mask.count, block_mask.idx, block_mask.full,
                score_mod, mask_mod,
                qk_scale, Int32(input_pos), Heads,
                tensorcore,
                Constant(Dk_pow2),
                Constant(Dv_pow2),
                Constant(TILE_M),
                Constant(TILE_N),
                Constant(query_group_size),
                Constant(even_k),
            )
        )
    else
        eff_bs = block_sparse && analytic_useful(mask_mod)
        @cutile(blocks=grid,
            flex_fwd(
                pQ, pK, pV, pO, M, L,
                score_mod, mask_mod,
                qk_scale, Int32(input_pos), Heads,
                tensorcore,
                Constant(Dk_pow2),
                Constant(Dv_pow2),
                Constant(TILE_M),
                Constant(TILE_N),
                Constant(query_group_size),
                Constant(even_k),
                Constant(eff_bs),
            )
        )
    end

    return
end

"""
    allocate_checkpoints(flex_attention!, Q, K, V) -> (; M, L)

The forward's checkpoints — softmax row max `M` and sum `L`, each
`(SeqLen_Q, Heads, Batch)` `Float32` — saved for [`∇flex_attention!`](@ref).
"""
allocate_checkpoints(::typeof(flex_attention!), Q, K, V) =
    (; M = similar(Q, Float32, (size(Q, 2), size(Q, 3), size(Q, 4))),
       L = similar(Q, Float32, (size(Q, 2), size(Q, 3), size(Q, 4))))

#=
"""
    _flex_attention(Q, K, V, score_mod; mask_mod = FullMask(), kwargs...) -> O

Positional-`score_mod` core of [`flex_attention`](@ref) — the differentiable
primitive the `ChainRulesCore`/`Mooncake` rules target. Because `score_mod` is a
positional argument here (not a kwarg), autodiff differentiates **its parameters**
too (e.g. `AliBiScore` slopes, `BiasScore` bias), alongside `Q`/`K`/`V`; `mask_mod`
stays fixed config. Wrap this directly when you want the variant's parameters
trained.
"""
=#
function _flex_attention(Q, K, V, score_mod; mask_mod = FullMask(), kwargs...)
    return _flex_attention_core(Q, K, V, score_mod, mask_mod, (; kwargs...))
end

# Fully-positional, no-kwargs core — the target of Mooncake's native `rrule!!`
# (in MooncakeExt), which carries learnable `score_mod` parameters that
# `@from_rrule` cannot (it zeroes non-array/struct tangents). `cfg` holds the
# remaining config (`input_pos`, `block_sparse`, …) as a NamedTuple. Mooncake
# reaches this by tracing through `_flex_attention`/`flex_attention`.
function _flex_attention_core(Q, K, V, score_mod, mask_mod, cfg::NamedTuple)
    O = similar(Q, size(V, 1), size(Q, 2), size(Q, 3), size(Q, 4))
    flex_attention!(O, Q, K, V; score_mod, mask_mod, cfg...)
    return O
end

"""
    flex_attention(Q, K, V; score_mod = NoOpScore(), mask_mod = FullMask(), kwargs...) -> O

Allocating forward; see [`flex_attention!`](@ref). A thin kwarg wrapper over the
differentiable core [`_flex_attention`](@ref) — autodiff through `flex_attention`
yields `Q`/`K`/`V` **and** `score_mod`-parameter gradients (`mask_mod` fixed
config). For score-mod param grads without the AD engine, call
[`∇flex_attention!`](@ref) with a [`grad_shadow`](@ref) `∂score_mod` directly.
"""
flex_attention(Q, K, V; score_mod = NoOpScore(), mask_mod = FullMask(), kwargs...) =
    _flex_attention(Q, K, V, score_mod; mask_mod, kwargs...)

"""
    ∇flex_attention!(O::Duplicated, Q::Duplicated, K::Duplicated, V::Duplicated;
                     checkpoints, scratch = …, score_mod, mask_mod, ∂score_mod = nothing, kwargs...)

In-place backward of [`flex_attention!`](@ref): reads `O.shadow` and the primals,
writes input gradients into `Q.shadow`/`K.shadow`/`V.shadow` (overwrite).
`checkpoints` must carry `M`/`L`; `scratch` is the transient workspace
([`allocate_scratchspace`](@ref)`(∇flex_attention!, …)`). `score_mod`/`mask_mod`
must match the forward. Score-mod parameter gradients accumulate into `∂score_mod`,
a [`grad_shadow`](@ref) of `score_mod` (`nothing` to skip). The precomputed-`BlockMask`
path has no backward. Returns `nothing`.
"""
function ∇flex_attention!(
    O::Duplicated, Q::Duplicated, K::Duplicated, V::Duplicated;
    score_mod = NoOpScore(),
    mask_mod = FullMask(),
    ∂score_mod = nothing,
    block_sparse::Bool = true,
    input_pos::Integer = 0,
    qk_scale = nothing,
    tensorcore = tensorcore_type(eltype(primal(Q))),
    TILE_M = 64,
    TILE_N = 64,
    checkpoints,
    scratch = allocate_scratchspace(∇flex_attention!, primal(Q), primal(K), primal(V), primal(O)),
)
    pO, pQ, pK, pV = primal(O), primal(Q), primal(K), primal(V)
    Ō = shadow(O)
    Q̄, K̄, V̄ = shadow(Q), shadow(K), shadow(V)
    (; M, L) = checkpoints
    (; Ō′, Δ) = scratch

    Dq, SeqLen_Q, Heads, Batch = size(pQ)
    Dk, SeqLen_K, Heads_KV, Batch_K = size(pK)
    Dv, SeqLen_V, Heads_V, Batch_V = size(pV)
    @assert Dq == Dk
    @assert SeqLen_K == SeqLen_V
    @assert Heads_KV == Heads_V
    @assert Batch == Batch_K == Batch_V
    @assert size(pO, 1) == Dv
    @assert size(Ō, 1) == Dv
    @assert size(M) == size(L) == (SeqLen_Q, Heads, Batch)
    @assert iszero(Heads % Heads_KV)

    query_group_size = Heads ÷ Heads_KV
    qk_scale = Float32(something(qk_scale, 1 / sqrt(Dk)))
    even_k = iszero(SeqLen_K % TILE_N)
    Dk_pow2 = nextpow(2, Dk)
    Dv_pow2 = nextpow(2, Dv)

    @cutile(blocks=(cld(SeqLen_Q, 32), Heads * Batch),
        mha_bwd_preprocess(
            Ō, pO, Ō′, L, Δ,
            Constant(Heads),
            Constant(Dv_pow2),
            Constant(32),
        )
    )

    eff_bs = block_sparse && analytic_useful(mask_mod)
    @cutile(blocks=Heads * Batch,
        flex_bwd(
            pQ, pK, pV, Ō′, M, Δ,
            fill!.((Q̄, K̄, V̄), 0)...,
            score_mod, mask_mod, ∂score_mod,
            qk_scale, Int32(input_pos), Heads,
            tensorcore,
            Constant(Dk_pow2),
            Constant(Dv_pow2),
            Constant(TILE_M),
            Constant(TILE_N),
            Constant(query_group_size),
            Constant(even_k),
            Constant(eff_bs),
        )
    )

    return
end

"""
    allocate_scratchspace(∇flex_attention!, Q, K, V, O) -> (; Ō′, Δ)

The backward's transient workspace: `Ō′` (row-normalized output gradient, same
shape/eltype as `O`) and `Δ` (per-query dot, `(SeqLen_Q, Heads, Batch)` `Float32`).
"""
function allocate_scratchspace(::typeof(∇flex_attention!), Q, K, V, O; kwargs...)
    return (; Ō′ = similar(O),
              Δ = similar(O, Float32, (size(O, 2), size(O, 3), size(O, 4))))
end

"""
    ∇flex_attention(Ō, Q, K, V, O; checkpoints, score_mod, mask_mod, ∂score_mod = nothing, kwargs...) -> (Q̄, K̄, V̄)

Allocating backward of [`flex_attention!`](@ref). Thin wrapper over
[`∇flex_attention!`](@ref); score-mod parameter grads go through `∂score_mod`.
"""
function ∇flex_attention(Ō, Q, K, V, O; checkpoints, kwargs...)
    Qd = Duplicated(Q)
    Kd = Duplicated(K)
    Vd = Duplicated(V)
    Od = Duplicated(O, Ō)

    ∇flex_attention!(Od, Qd, Kd, Vd; checkpoints, kwargs...)

    return shadow(Qd), shadow(Kd), shadow(Vd)
end
