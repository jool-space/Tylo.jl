export layer_norm
export layer_norm!
export ∇layer_norm
export ∇layer_norm!

function layer_norm_fwd(
    X::TileMatrix, W::TileVector,
    B::TileVector, Y::TileMatrix,
    Mean::Optional{TileVector{Float32}}, Rstd::Optional{TileVector{Float32}},
    eps::Float32, TILE_M::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)

    mean = zeros(Float32, TILE_M)
    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode) → Float32
        mean = mean .+ x
    end
    mean = sum(mean) / M
    isnothing(Mean) || (Mean[bid_n] = mean)

    var = zeros(Float32, TILE_M)
    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode) → Float32
        mask = ((i - 1i32) * Int32(TILE_M) .+ ct.arange(TILE_M)) .<= M
        centered_x = ifelse.(mask, x .- mean, 0.0f0)
        var = var .+ (centered_x .* centered_x)
    end
    var = sum(var) / M
    rstd = 1 / √(var + eps)
    isnothing(Rstd) || (Rstd[bid_n] = rstd)

    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,)) → Float32
        w = ct.load(W, i, (TILE_M,)) → Float32
        b = ct.load(B, i, (TILE_M,)) → Float32
        y = (x .- mean) .* rstd
        y = y .* w .+ b
        ct.store(Y, (i, bid_n), y → eltype(Y))
    end

    return
end

@inline function bwd_helper(X, W, Ȳ, bid_n, i, mean, rstd, TILE_M, M)
    padding_mode = ct.PaddingMode.Zero
    x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode) → Float32
    w = ct.load(W, i, (TILE_M,); padding_mode) → Float32
    ȳ = ct.load(Ȳ, (i, bid_n), (TILE_M,); padding_mode) → Float32
    xhat = (x .- mean) .* rstd
    wȳ = w .* ȳ

    # Mask for valid elements
    indices = ct.arange(TILE_M)
    offset = (i - 1i32) * Int32(TILE_M)
    global_indices = offset .+ indices
    mask = global_indices .<= M

    xhat_masked = ifelse.(mask, xhat, 0.0f0)
    wȳ_masked = ifelse.(mask, wȳ, 0.0f0)

    return ȳ, xhat_masked, wȳ_masked
end

function layer_norm_bwd_dx_partial_dwdb(
    X̄::TileMatrix, Ȳ::TileMatrix,
    W̄::TileMatrix{Float32}, B̄::TileMatrix{Float32},
    X::TileMatrix, W::TileVector,
    Mean::TileVector{Float32}, Rstd::TileVector{Float32},
    Locks::TileVector{Int},
    N_GROUPS::Int, TILE_M::Int
)
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)
    group_id = mod1(bid_n, Int32(N_GROUPS))

    mean = Mean[bid_n]
    rstd = Rstd[bid_n]

    c1 = zeros(Float32, TILE_M)
    c2 = zeros(Float32, TILE_M)
    for i in 1i32:num_tiles
        _, xhat, wȳ = bwd_helper(X, W, Ȳ, bid_n, i, mean, rstd, TILE_M, M)
        c1 = c1 .+ (xhat .* wȳ)
        c2 = c2 .+ wȳ
    end
    c1 = sum(c1) / M
    c2 = sum(c2) / M

    for i in 1i32:num_tiles
        ȳ, xhat, wȳ = bwd_helper(X, W, Ȳ, bid_n, i, mean, rstd, TILE_M, M)
        x̄ = (wȳ .- (xhat .* c1 .+ c2)) .* rstd
        ct.store(X̄, (i, bid_n), x̄ → eltype(X̄))

        partial_w̄ = ȳ .* xhat
        partial_b̄ = ȳ

        # Acquire spinlock
        while ct.atomic_cas(Locks, group_id, 0, 1;
                memory_order=ct.MemoryOrder.Acquire) == 1
            # spin
        end

        # Critical section: accumulate partial gradients
        partial_w̄ = partial_w̄ .+ ct.load(W̄, (i, group_id), (TILE_M,))
        partial_b̄ = partial_b̄ .+ ct.load(B̄, (i, group_id), (TILE_M,))
        ct.store(W̄, (i, group_id), partial_w̄)
        ct.store(B̄, (i, group_id), partial_b̄)

        # Release spinlock
        ct.atomic_xchg(Locks, group_id, 0;
                      memory_order=ct.MemoryOrder.Release)
    end

    return
end

function layer_norm_bwd_dwdb(
    W̄::TileMatrix{Float32}, B̄::TileMatrix{Float32},
    FINAL_W̄::TileVector, FINAL_B̄::TileVector,
    TILE_G::Int, TILE_F::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid = ct.bid(1)
    num_tiles = ct.num_tiles(W̄, 2, (TILE_F, TILE_G))

    w̄ = zeros(Float32, (TILE_F, TILE_G))
    b̄ = zeros(Float32, (TILE_F, TILE_G))
    for i in 1i32:num_tiles
        w̄ = w̄ .+ ct.load(W̄, (bid, i), (TILE_F, TILE_G); padding_mode)
        b̄ = b̄ .+ ct.load(B̄, (bid, i), (TILE_F, TILE_G); padding_mode)
    end
    ct.store(FINAL_W̄, bid, sum(w̄; dims=2) → eltype(FINAL_W̄))
    ct.store(FINAL_B̄, bid, sum(b̄; dims=2) → eltype(FINAL_B̄))

    return
end

"""
    layer_norm!(Y, X, W, B; eps, TILE_M = 256, checkpoints = nothing)

Layer-normalize each column of `X` in place: `y = (x - mean) * rstd * w + b` with
`rstd = 1/√(var + eps)`.

  * `X`, `Y`: `(M, N)`
  * `W`, `B`: `(M,)`

`Y`/`X`/`W`/`B` may be bare arrays or [`Duplicated`](@ref) (only the primals are
read). Pass `checkpoints` from [`allocate_checkpoints`](@ref)`(layer_norm!, X, W, B)`
to save `Mean`/`Rstd` for [`∇layer_norm!`](@ref); the default `nothing` saves
nothing (inference). Returns `nothing`.
"""
function layer_norm!(
    Y, X, W, B;
    eps, TILE_M = 256,
    checkpoints = nothing,
)
    px, pw, pb, py = primal(X), primal(W), primal(B), primal(Y)
    M, N = size(px)
    @assert length(pw) == length(pb) == M
    Mean = checkpoints === nothing ? nothing : get(checkpoints, :Mean, nothing)
    Rstd = checkpoints === nothing ? nothing : get(checkpoints, :Rstd, nothing)

    @cutile(blocks=N,
        layer_norm_fwd(px, pw, pb, py, Mean, Rstd, Constant(Float32(eps)), Constant(TILE_M))
    )

    return
end

"""
    allocate_checkpoints(layer_norm!, X, W, B) -> (; Mean, Rstd)

The forward's checkpoints — the per-column statistics (`Mean`, `Rstd`, each `(N,)`
`Float32`) bridging forward→backward. The forward writes them; [`∇layer_norm!`](@ref)
reads them. See [`allocate_scratchspace`](@ref) for the backward's workspace.
"""
allocate_checkpoints(::typeof(layer_norm!), X, W, B) =
    (; Mean = similar(W, Float32, size(X, 2)), Rstd = similar(W, Float32, size(X, 2)))

"""
    layer_norm(X, W, B; eps, kwargs...) -> Y

Allocating forward; see [`layer_norm!`](@ref). Autodiff entry point
(`ChainRulesCore`/`Mooncake` rules differentiate `layer_norm`).
"""
function layer_norm(X, W, B; eps, kwargs...)
    Y = similar(X)
    layer_norm!(Y, X, W, B; eps, kwargs...)
    return Y
end

"""
    ∇layer_norm!(Y::Duplicated, X::Duplicated, W::Duplicated, B::Duplicated; checkpoints, scratch = …, …)

In-place backward of [`layer_norm!`](@ref). Reads the output gradient from
`Y.shadow` and the inputs from `X.primal`/`W.primal`, and writes the input
gradients into `X.shadow`/`W.shadow`/`B.shadow` (overwrite). `checkpoints` must
carry `Mean`/`Rstd` from the forward; `scratch` is the transient backward
workspace ([`allocate_scratchspace`](@ref)`(∇layer_norm!, …)`), allocated on the fly
if not supplied. Returns `nothing`.
"""
function ∇layer_norm!(
    Y::Duplicated, X::Duplicated, W::Duplicated, B::Duplicated;
    N_GROUPS = 64, TILE_M = 256, TILE_F = 64, TILE_G = 64,
    checkpoints,
    scratch = allocate_scratchspace(∇layer_norm!, primal(X), primal(W), primal(B); N_GROUPS),
)
    (; Mean, Rstd) = checkpoints
    (; W̄_partial, B̄_partial, Locks) = scratch
    px, pw = primal(X), primal(W)
    M, N = size(px)

    @cutile(blocks=N,
        layer_norm_bwd_dx_partial_dwdb(
            shadow(X), shadow(Y), fill!(W̄_partial, 0), fill!(B̄_partial, 0), px, pw,
            Mean, Rstd, fill!(Locks, 0), Constant(N_GROUPS), Constant(TILE_M)
        )
    )

    @cutile(blocks=cld(M, TILE_F),
        layer_norm_bwd_dwdb(W̄_partial, B̄_partial, shadow(W), shadow(B), Constant(TILE_G), Constant(TILE_F))
    )

    return
end

"""
    allocate_scratchspace(∇layer_norm!, X, W, B; N_GROUPS = 64) -> (; W̄_partial, B̄_partial, Locks)

The backward's transient workspace: the reduction's `W̄_partial`/`B̄_partial`
`(M, N_GROUPS)` partial accumulators (always `Float32`) and per-group `Locks`
`(N_GROUPS,)`. See [`allocate_checkpoints`](@ref) for the forward→backward bridge.
"""
function allocate_scratchspace(::typeof(∇layer_norm!), X, W, B; N_GROUPS = 64)
    M, _ = size(X)
    return (;
        W̄_partial = similar(W, Float32, M, N_GROUPS),
        B̄_partial = similar(B, Float32, M, N_GROUPS),
        Locks = similar(X, Int, N_GROUPS),
    )
end

"""
    ∇layer_norm(Ȳ, X, W, B; checkpoints, kwargs...) -> (X̄, W̄, B̄)

Allocating backward of [`layer_norm!`](@ref). `checkpoints` must carry `Mean`/`Rstd`
from the forward. Thin wrapper over [`∇layer_norm!`](@ref).
"""
function ∇layer_norm(
    Ȳ::AbstractMatrix, X::AbstractMatrix,
    W::AbstractVector, B::AbstractVector;
    checkpoints,
    N_GROUPS = 64, TILE_M = 256, TILE_F = 64, TILE_G = 64,
)
    Xd = Duplicated(X)
    Wd = Duplicated(W)
    Bd = Duplicated(B)
    Yd = Duplicated(Ȳ, Ȳ)  # backward reads only the shadow (Ȳ)

    ∇layer_norm!(Yd, Xd, Wd, Bd; N_GROUPS, TILE_M, TILE_F, TILE_G, checkpoints)

    return shadow(Xd), shadow(Wd), shadow(Bd)
end
