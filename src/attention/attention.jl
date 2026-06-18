export attention
export attention!
export ∇attention
export ∇attention!

function mha_fwd(
    Q::TileArray4, K::TileArray4, V::TileArray4, O::TileArray4,
    M::Optional{TileArray3{Float32}}, L::Optional{TileArray3{Float32}},
    k_lengths::Optional{TileVector{Int32}},
    q_lengths::Optional{TileVector{Int32}},
    qk_scale::Float32,
    input_pos::Int32,
    H::Int,
    Tc::Type,
    Dk::Int, Dv::Int,
    TILE_M::Int, TILE_N::Int,
    QUERY_GROUP_SIZE::Int,
    CAUSAL::Bool,
)
    padding_mode = ct.PaddingMode.Zero
    i, hb = ct.bid(1), ct.bid(2)
    b, h = fldmod1(hb, H)
    hₖ = fld1(h, QUERY_GROUP_SIZE)

    q_len = isnothing(q_lengths) ? size(Q, 2) : q_lengths[b]
    (i - 1i32) * TILE_M >= q_len && return

    offs_m = (i - 1i32) * TILE_M .+ ct.arange(TILE_M) .- 1i32 .+ input_pos
    offs_n_tile = ct.arange(TILE_N) .- 1i32

    m_i = fill(-Inf32, (1, TILE_M))
    l_i = zeros(Float32, (1, TILE_M))
    acc = zeros(Float32, (Dv, TILE_M))

    q = ct.load(Q, (1, i, h, b), (Dk, TILE_M); padding_mode)

    k_len = isnothing(k_lengths) ? size(K, 2) : k_lengths[b]
    m_end = input_pos + i * TILE_M

    if CAUSAL
        mask_start = min(fld(input_pos + (i - 1i32) * TILE_M, TILE_N), fld(k_len, TILE_N))
        kv_tiles = cld(min(Int32(m_end), k_len), TILE_N)
    else
        mask_start = fld(k_len, TILE_N)
        kv_tiles = cld(k_len, TILE_N)
    end

    for j in 1i32:kv_tiles
        k = ct.load(K, (1, j, hₖ, b), (Dk, TILE_N); padding_mode, latency=2)

        s = muladd((k)ᵀ → Tc, q → Tc, zeros(Float32, (TILE_N, TILE_M)))
        # exp2 softmax: fold log2(e) = inv(log 2) into the scale so the row-max
        # and exps run in base-2 (hardware ex2). `M` is stored in these log2
        # units; the backward recomputes `p` the same way but keeps the TRUE
        # `qk_scale` for `s̄` (the ln2 / log2(e) factors cancel in the gradient).
        s = s * qk_scale * inv(log(2f0))

        if j > mask_start
            offs_n = (j - 1i32) * TILE_N .+ offs_n_tile
            mask = offs_n .< k_len
            CAUSAL && (mask = mask .& (offs_n .<= (offs_m)ᵀ))
            s = ifelse.(mask, s, -Inf32)
        end

        m_ij = max.(m_i, maximum(s, dims=1))
        ct.@fpmode flush_to_zero=true begin
            p = exp2.(s .- m_ij)
            l_ij = sum(p, dims=1)
            alpha = exp2.(m_i .- m_ij)
            l_i = l_i .* alpha .+ l_ij
            acc = acc .* alpha
        end

        v = ct.load(V, (1, j, hₖ, b), (Dv, TILE_N); padding_mode, latency=4)
        acc = muladd(v → Tc, p → Tc, acc)

        m_i = m_ij
    end

    ct.@fpmode rounding_mode=ct.Rounding.Approx flush_to_zero=true begin
        o = acc ./ l_i
    end
    ct.store(O, (1, i, h, b), o → eltype(O))
    isnothing(M) || ct.store(M, (i, h, b), reshape(m_i, TILE_M))
    isnothing(L) || ct.store(L, (i, h, b), reshape(l_i, TILE_M))

    return
end

function mha_bwd_preprocess(
    Ō::TileArray4,
    O::TileArray4,
    Ō′::TileArray4,
    L::TileArray3{Float32},
    Δ::TileArray3{Float32},
    H::Int, Dv::Int, TILE_M::Int
)
    padding_mode = ct.PaddingMode.Zero
    i, hb = ct.bid(1), ct.bid(2)
    b, h = fldmod1(hb, H)

    #q_len = isnothing(q_lengths) ? size(Q, 2) : q_lengths[b]
    #(i - 1i32) * TILE_M >= q_len && return

    ō = ct.load(Ō, (1, i, h, b), (Dv, TILE_M); padding_mode)
    o  = ct.load(O, (1, i, h, b), (Dv, TILE_M); padding_mode)

    l = reshape(ct.load(L, (i, h, b), (TILE_M,)), (1, TILE_M))

    ō′ = ō .* ifelse.(l .== 0f0, 0f0, 1f0 ./ l)
    ct.store(Ō′, (1, i, h, b), ō′ → eltype(Ō′))

    δ = sum(ō′ .* o, dims=1)
    ct.store(Δ, (i, h, b), reshape(δ, TILE_M))

    return
end

function mha_bwd(
    Q::TileArray4, K::TileArray4, V::TileArray4,
    Ō′::TileArray4,
    M::TileArray3{Float32},
    Δ::TileArray3{Float32},
    Q̄::TileArray4, K̄::TileArray4, V̄::TileArray4,
    k_lengths::Optional{TileVector{Int32}},
    q_lengths::Optional{TileVector{Int32}},
    qk_scale::Float32,
    input_pos::Integer,
    H::Integer,
    Tc::Type,
    Dk::Int, Dv::Int,
    TILE_M::Int, TILE_N::Int,
    QUERY_GROUP_SIZE::Int,
    CAUSAL::Bool,
)
    padding_mode = ct.PaddingMode.Zero
    hb = ct.bid(1)
    b, h = fldmod1(hb, H)
    hₖ = fld1(h, QUERY_GROUP_SIZE)

    k_len = isnothing(k_lengths) ? size(K, 2) : k_lengths[b]
    q_len = isnothing(q_lengths) ? size(Q, 2) : q_lengths[b]

    q_tiles = cld(q_len, TILE_M)
    kv_tiles = cld(k_len, TILE_N)

    offs_n_base = ct.arange(TILE_N) .- 1i32

    for j in 1i32:kv_tiles
        k = ct.load(K, (1, j, hₖ, b), (Dk, TILE_N); padding_mode)
        v = ct.load(V, (1, j, hₖ, b), (Dv, TILE_N); padding_mode)

        k̄_acc = zeros(Float32, (Dk, TILE_N))
        v̄_acc = zeros(Float32, (Dv, TILE_N))

        offs_n = (j - 1i32) * TILE_N .+ offs_n_base
        pad_mask_needed = j > fld(k_len, TILE_N)

        for i in 1i32:q_tiles
            q = ct.load(Q, (1, i, h, b), (Dk, TILE_M); padding_mode, allow_tma=false)
            ō = ct.load(Ō′, (1, i, h, b), (Dv, TILE_M); padding_mode, allow_tma=false)

            m = reshape(ct.load(M, (i, h, b), (TILE_M,), latency=1), (1, TILE_M))
            δ = reshape(ct.load(Δ, (i, h, b), (TILE_M,), latency=1), (1, TILE_M))

            s = muladd((k)ᵀ → Tc, q → Tc, zeros(Float32, (TILE_N, TILE_M)))
            s = s * qk_scale * inv(log(2f0))

            if CAUSAL || pad_mask_needed
                offs_m = (i - 1i32) * TILE_M .+ ct.arange(TILE_M) .- 1i32 .+ input_pos
                mask = offs_n .< k_len
                CAUSAL && (mask = mask .& (offs_n .<= (offs_m)ᵀ))
                s = ifelse.(mask, s, -Inf32)
            end

            p = exp2.(s .- m)
            v̄_acc = muladd(ō → Tc, (p)ᵀ → Tc, v̄_acc)

            p̄ = muladd((v)ᵀ → Tc, ō → Tc, zeros(Float32, (TILE_N, TILE_M)))

            ds = p .* (p̄ .- δ)

            s̄ = ds * qk_scale

            q̄ = ct.load(Q̄, (1, i, h, b), (Dk, TILE_M), allow_tma=false)
            q̄ = muladd(k → Tc, s̄ → Tc, q̄ → Float32)
            ct.store(Q̄, (1, i, h, b), q̄ → eltype(Q̄))

            k̄_acc = muladd(q → Tc, (s̄)ᵀ → Tc, k̄_acc)
        end

        store = isone(QUERY_GROUP_SIZE) ? ct.store : atomic_add_tile
        store(K̄, (1, j, hₖ, b), k̄_acc → eltype(K̄))
        store(V̄, (1, j, hₖ, b), v̄_acc → eltype(V̄))
    end

    return
end

"""
    attention!(O, Q, K, V; causal = false, checkpoints = nothing, kwargs...)

Fused multi-head attention with GQA support. For additive bias or other arbitrary
variants use [`flex_attention!`](@ref) (e.g. `BiasScore`).

  * `Q`: `(Dk, SeqLen_Q, Heads, Batch)`
  * `K`: `(Dk, SeqLen_K, Heads_KV, Batch)`
  * `V`: `(Dv, SeqLen_K, Heads_KV, Batch)`
  * `O`: `(Dv, SeqLen_Q, Heads, Batch)`

`O`/`Q`/`K`/`V` may be bare arrays or [`Duplicated`](@ref) (only the primals
are read). Pass `checkpoints` from [`allocate_checkpoints`](@ref)`(attention!, Q, K, V)`
to save the softmax statistics `M`/`L` for [`∇attention!`](@ref). Compute precision
is set by `tensorcore` (default from `eltype(Q)`); accumulation is Float32. Other
keywords: `causal`; `input_pos`, absolute position of the first query;
`k_lengths`/`q_lengths`, optional per-batch valid lengths; `TILE_M`/`TILE_N`.
Returns `nothing`.
"""
function attention!(O,
    Q, K, V;
    causal = false,
    input_pos = 0,
    k_lengths = nothing,
    q_lengths = nothing,
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
    @assert iszero(Heads % Heads_KV)

    query_group_size = Heads ÷ Heads_KV
    qk_scale = Float32(1 / sqrt(Dk))
    Dk_pow2 = nextpow(2, Dk)
    Dv_pow2 = nextpow(2, Dv)

    M = checkpoints === nothing ? nothing : get(checkpoints, :M, nothing)
    L = checkpoints === nothing ? nothing : get(checkpoints, :L, nothing)

    @cutile(blocks=(cld(SeqLen_Q, TILE_M), Heads * Batch),
        mha_fwd(
            pQ, pK, pV, pO, M, L, k_lengths, q_lengths,
            qk_scale, Int32(input_pos), Heads,
            tensorcore,
            Constant(Dk_pow2),
            Constant(Dv_pow2),
            Constant(TILE_M),
            Constant(TILE_N),
            Constant(query_group_size),
            Constant(causal),
        )
    )

    return
end

"""
    allocate_checkpoints(attention!, Q, K, V) -> (; M, L)

The forward's checkpoints — the per-query softmax statistics `M` (row max) and `L`
(row sum), each `(SeqLen_Q, Heads, Batch)` `Float32`, bridging forward→backward.
The forward writes them; [`∇attention!`](@ref) reads them.
"""
allocate_checkpoints(::typeof(attention!), Q, K, V) =
    (; M = similar(Q, Float32, (size(Q, 2), size(Q, 3), size(Q, 4))),
       L = similar(Q, Float32, (size(Q, 2), size(Q, 3), size(Q, 4))))

"""
    attention(Q, K, V; causal = false, kwargs...) -> O

Allocating forward; see [`attention!`](@ref). Autodiff entry point
(`ChainRulesCore`/`Mooncake` rules differentiate `attention`).
"""
function attention(Q, K, V; kwargs...)
    O = similar(Q, size(V, 1), size(Q, 2), size(Q, 3), size(Q, 4))
    attention!(O, Q, K, V; kwargs...)
    return O
end

"""
    ∇attention!(O::Duplicated, Q::Duplicated, K::Duplicated, V::Duplicated;
                checkpoints, scratch = …, causal, kwargs...)

In-place backward of [`attention!`](@ref): reads the output gradient `O.shadow`
and the primals, and writes input gradients into `Q.shadow`/`K.shadow`/`V.shadow`
(overwrite). `checkpoints` must carry `M`/`L` from the forward; `scratch` is the
transient backward workspace ([`allocate_scratchspace`](@ref)`(∇attention!, …)`).
`causal`/`input_pos` must match the forward. Returns `nothing`.
"""
function ∇attention!(
    O::Duplicated, Q::Duplicated, K::Duplicated, V::Duplicated;
    causal,
    input_pos = 0,
    k_lengths = nothing,
    q_lengths = nothing,
    tensorcore = tensorcore_type(eltype(primal(Q))),
    TILE_M = 64,
    TILE_N = 64,
    checkpoints,
    scratch = allocate_scratchspace(∇attention!, primal(Q), primal(K), primal(V), primal(O)),
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
    @assert iszero(Heads % Heads_KV)

    query_group_size = Heads ÷ Heads_KV
    qk_scale = Float32(1 / sqrt(Dk))
    Dk_pow2 = nextpow(2, Dk)
    Dv_pow2 = nextpow(2, Dv)

    @cutile(blocks=(cld(SeqLen_Q, 32), Heads*Batch),
        mha_bwd_preprocess(
            Ō, pO, Ō′, L, Δ,
            Constant(Heads),
            Constant(Dv_pow2),
            Constant(32)
        )
    )

    @cutile(blocks=Heads*Batch,
        mha_bwd(
            pQ, pK, pV, Ō′, M, Δ,
            fill!.((Q̄, K̄, V̄), 0)...,
            k_lengths, q_lengths,
            qk_scale, Int32(input_pos), Heads,
            tensorcore,
            Constant(Dk_pow2),
            Constant(Dv_pow2),
            Constant(TILE_M),
            Constant(TILE_N),
            Constant(query_group_size),
            Constant(causal),
        )
    )

    return
end

"""
    allocate_scratchspace(∇attention!, Q, K, V, O) -> (; Ō′, Δ)

The backward's transient workspace: `Ō′` (the per-row-normalized output gradient,
same shape/eltype as `O`) and `Δ` (the per-query dot product, `(SeqLen_Q, Heads,
Batch)` `Float32`) — produced by the preprocess kernel, consumed by the main
backward kernel.
"""
function allocate_scratchspace(::typeof(∇attention!), Q, K, V, O; kwargs...)
    return (; Ō′ = similar(O),
              Δ = similar(O, Float32, (size(O, 2), size(O, 3), size(O, 4))))
end

"""
    ∇attention(Ō, Q, K, V, O; checkpoints, causal, kwargs...) -> (Q̄, K̄, V̄)

Allocating backward of [`attention!`](@ref): from the output gradient `Ō`, the
inputs, the output `O`, and the forward's `checkpoints` (`M`/`L`), return the input
gradients. Thin wrapper over [`∇attention!`](@ref).
"""
function ∇attention(Ō, Q, K, V, O; checkpoints, causal, kwargs...)
    Qd = Duplicated(Q)
    Kd = Duplicated(K)
    Vd = Duplicated(V)
    Od = Duplicated(O, Ō)

    ∇attention!(Od, Qd, Kd, Vd; checkpoints, causal, kwargs...)

    return shadow(Qd), shadow(Kd), shadow(Vd)
end
