export rms_norm
export rms_norm!
export ∇rms_norm
export ∇rms_norm!

function rms_norm_fwd(
    X::TileMatrix, W::TileVector,
    Y::TileMatrix, Rstd::Optional{TileVector{Float32}},
    offset::Float32, eps::Float32,
    TILE_M::Int
)
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)

    ss = zeros(Float32, TILE_M)
    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode=ct.PaddingMode.Zero) → Float32
        ss = ss .+ x .* x
    end
    rstd = 1 / √(sum(ss) / M .+ eps)
    isnothing(Rstd) || (Rstd[bid_n] = rstd)

    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,)) → Float32
        w = ct.load(W, i, (TILE_M,)) → Float32
        y = x .* rstd .* (w .+ offset)
        ct.store(Y, (i, bid_n), y → eltype(Y))
    end

    return
end

function rms_norm_bwd_dx_partial_dw(
    X̄::TileMatrix, Ȳ::TileMatrix,
    W̄::TileMatrix,
    X::TileMatrix, W::TileVector,
    Rstd::TileVector,
    Locks::TileVector{Int},
    offset::Float32, N_GROUPS::Int, TILE_M::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)
    group_id = mod1(bid_n, Int32(N_GROUPS))

    dd = zeros(Float32, TILE_M)
    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode) → Float32
        w = ct.load(W, i, (TILE_M,); padding_mode) → Float32
        ȳ = ct.load(Ȳ, (i, bid_n), (TILE_M,); padding_mode) → Float32
        dd = dd .+ (ȳ .* (w .+ offset) .* x)
    end
    dd = sum(dd) / M

    rstd = Rstd[bid_n]
    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,)) → Float32
        w = ct.load(W, i, (TILE_M,)) → Float32
        ȳ = ct.load(Ȳ, (i, bid_n), (TILE_M,)) → Float32

        x̄ = rstd .* (ȳ .* (w .+ offset) .- (rstd * rstd * dd) .* x)
        ct.store(X̄, (i, bid_n), x̄ → eltype(X̄))

        partial_w̄ = ȳ .* x .* rstd

        # Acquire spinlock
        while ct.atomic_cas(Locks, group_id, 0, 1;
                memory_order=ct.MemoryOrder.Acquire) == 1
            # spin
        end

        partial_w̄ = partial_w̄ .+ ct.load(W̄, (i, group_id), (TILE_M,))
        ct.store(W̄, (i, group_id), partial_w̄)

        # Release spinlock
        ct.atomic_xchg(Locks, group_id, 0; memory_order=ct.MemoryOrder.Release)
    end

    return
end

function rms_norm_bwd_dw(
    W̄::TileMatrix{Float32},
    FINAL_W̄::TileVector,
    TILE_G::Int, TILE_F::Int
)
    bid = ct.bid(1)
    num_tiles = ct.num_tiles(W̄, 2, (TILE_F, TILE_G))

    w̄ = zeros(Float32, (TILE_F, TILE_G))
    for i in 1i32:num_tiles
        w̄ = w̄ .+ ct.load(W̄, (bid, i), (TILE_F, TILE_G); padding_mode=ct.PaddingMode.Zero)
    end
    ct.store(FINAL_W̄, bid, sum(w̄; dims=2) → eltype(FINAL_W̄))

    return
end

"""
    rms_norm!(Y, X, W; eps, offset = 0f0, TILE_M = 256, checkpoints = nothing)

RMS-normalize each column of `X` in place: `y = x * rstd * (w + offset)` with
`rstd = 1/√(mean(x²) + eps)`.

  * `X`, `Y`: `(M, N)`
  * `W`: `(M,)`

`Y`/`X`/`W` may be bare arrays or [`Duplicated`](@ref) (only the primals are
read). Pass `checkpoints` from [`allocate_checkpoints`](@ref)`(rms_norm!, X, W)`
to save `rstd` into `checkpoints.Rstd` for [`∇rms_norm!`](@ref); the default
`nothing` saves nothing (inference). Returns `nothing`.
"""
function rms_norm!(
    Y, X, W;
    eps, offset = 0.0f0, TILE_M = 256,
    checkpoints = nothing,
)
    px, pw, py = primal(X), primal(W), primal(Y)
    _, N = size(px)
    Rstd = checkpoints === nothing ? nothing : get(checkpoints, :Rstd, nothing)

    @cutile(blocks=N,
        rms_norm_fwd(px, pw, py, Rstd, Float32(offset), Float32(eps), Constant(TILE_M)))

    return
end

"""
    allocate_checkpoints(rms_norm!, X, W) -> (; Rstd)

The forward's checkpoints — saved activations bridging forward→backward. For
RMSNorm that's `Rstd` `(N,)`, always `Float32`. Pass to [`rms_norm!`](@ref) as
`checkpoints` (the forward writes it) and on to [`∇rms_norm!`](@ref) (the backward
reads it). See [`allocate_scratchspace`](@ref) for a pass's transient workspace.
"""
allocate_checkpoints(::typeof(rms_norm!), X, W) = (; Rstd = similar(W, Float32, size(X, 2)))

"""
    rms_norm(X, W; eps, offset = 0f0, kwargs...) -> Y

Allocating forward; see [`rms_norm!`](@ref). This is the autodiff entry point
(`ChainRulesCore`/`Mooncake` rules differentiate `rms_norm`).
"""
function rms_norm(X, W; eps, offset = 0.0f0, kwargs...)
    Y = similar(X)
    rms_norm!(Y, X, W; eps, offset, kwargs...)
    return Y
end

"""
    ∇rms_norm!(Y::Duplicated, X::Duplicated, W::Duplicated; checkpoints, scratch = …, offset = 0f0, …)

In-place backward of [`rms_norm!`](@ref). Reads the output gradient from
`Y.shadow` and the inputs from `X.primal`/`W.primal`, and writes the input
gradients into `X.shadow`/`W.shadow` (overwrite). `checkpoints` must carry
`checkpoints.Rstd` from the forward; `scratch` is the transient backward
workspace ([`allocate_scratchspace`](@ref)`(∇rms_norm!, …)`), allocated on the fly
if not supplied. Returns `nothing`.
"""
function ∇rms_norm!(
    Y::Duplicated, X::Duplicated, W::Duplicated;
    offset = 0f0, N_GROUPS = 64, TILE_M = 256, TILE_F = 64, TILE_G = 64,
    checkpoints,
    scratch = allocate_scratchspace(∇rms_norm!, primal(X), primal(W); N_GROUPS),
)
    Rstd = checkpoints.Rstd
    (; W̄_partial, Locks) = scratch
    px, pw = primal(X), primal(W)
    M, N = size(px)

    @cutile(blocks=N,
        rms_norm_bwd_dx_partial_dw(
            shadow(X), shadow(Y), fill!(W̄_partial, 0), px, pw, Rstd, fill!(Locks, 0),
            Constant(Float32(offset)), Constant(N_GROUPS), Constant(TILE_M)
        )
    )

    @cutile(blocks=cld(M, TILE_F),
        rms_norm_bwd_dw(W̄_partial, shadow(W), Constant(TILE_G), Constant(TILE_F))
    )

    return
end

"""
    allocate_scratchspace(∇rms_norm!, X, W; N_GROUPS = 64) -> (; W̄_partial, Locks)

The backward's transient workspace: the reduction's `W̄_partial` `(M, N_GROUPS)`
partial accumulators (always `Float32`) and per-group `Locks` `(N_GROUPS,)`. Lives
only within the backward pass — see [`allocate_checkpoints`](@ref) for the
forward→backward bridge.
"""
function allocate_scratchspace(::typeof(∇rms_norm!), X, W; N_GROUPS = 64)
    M, _ = size(X)
    return (; W̄_partial = similar(W, Float32, M, N_GROUPS), Locks = similar(X, Int, N_GROUPS))
end

"""
    ∇rms_norm(Ȳ, X, W; checkpoints, offset = 0f0, kwargs...) -> (X̄, W̄)

Allocating backward of [`rms_norm!`](@ref). `checkpoints` must carry `Rstd` from
the forward; `offset` must match. Thin wrapper over [`∇rms_norm!`](@ref).
"""
function ∇rms_norm(
    Ȳ::AbstractMatrix, X::AbstractMatrix,
    W::AbstractVector;
    checkpoints,
    offset = 0f0, N_GROUPS = 64,
    TILE_M = 256, TILE_F = 64, TILE_G = 64,
)
    Xd = Duplicated(X)
    Wd = Duplicated(W)
    Yd = Duplicated(Ȳ, Ȳ)  # backward reads only the shadow (Ȳ)

    ∇rms_norm!(Yd, Xd, Wd; offset, N_GROUPS, TILE_M, TILE_F, TILE_G, checkpoints)

    return shadow(Xd), shadow(Wd)
end
