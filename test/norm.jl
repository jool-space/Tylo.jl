# Normalization over the contiguous axis (dim 1) of an (M, N) matrix: M is the
# feature/reduction length, N the number of rows (tokens) — one block per row.
# References in Float64; kernels run in Float32, so the gap is just accumulation
# order → tight tol. M values span multiples and non-multiples of TILE_M (256),
# including a partial single tile, to exercise the padding paths.

# RMSNorm: y = x · rstd · (w + offset),  rstd = 1/√(mean(x²) + eps).
function rms_norm_ref(X, W; eps, offset = 0.0)
    M = size(X, 1)
    rstd = 1 ./ sqrt.(sum(abs2, X; dims = 1) ./ M .+ eps)
    return (X .* rstd) .* (W .+ offset)
end

# LayerNorm: y = (x - μ)·rstd·w + b,  rstd = 1/√(var + eps), var = mean((x-μ)²).
function layer_norm_ref(X, W, B; eps)
    M = size(X, 1)
    xc = X .- sum(X; dims = 1) ./ M
    rstd = 1 ./ sqrt.(sum(abs2, xc; dims = 1) ./ M .+ eps)
    return (xc .* rstd) .* W .+ B
end

function gpu_rms_norm(X, W; eps, offset = 0.0f0, T = Float32, kwargs...)
    dX = CuArray(T.(X))
    dW = CuArray(T.(W))
    dY = similar(dX)
    rms_norm!(dY, dX, dW; eps = Float32(eps), offset = Float32(offset), kwargs...)
    return Array(dY)
end

function gpu_layer_norm(X, W, B; eps, T = Float32, kwargs...)
    dX = CuArray(T.(X))
    dW = CuArray(T.(W))
    dB = CuArray(T.(B))
    dY = similar(dX)
    layer_norm!(dY, dX, dW, dB; eps = Float32(eps), kwargs...)
    return Array(dY)
end

const NORM_ATOL = 1f-5
const NORM_RTOL = 2f-3

# (M = feature length, N = rows). 256/512/768 are multiples of TILE_M; 130 is a
# partial single tile; 300 is a non-multiple spanning two tiles.
const NORM_SHAPES = [(256, 32), (512, 16), (768, 8), (130, 16), (300, 8)]

@testset "rms_norm!" begin
    rng = MersenneTwister(3)
    @testset "M=$M N=$N offset=$offset" for (M, N) in NORM_SHAPES, offset in (0.0, 1.0)
        X = randn(rng, Float64, M, N)
        W = randn(rng, Float64, M)
        Y_ref = rms_norm_ref(X, W; eps = 1e-5, offset)
        Y_gpu = gpu_rms_norm(X, W; eps = 1e-5, offset)
        @test size(Y_gpu) == size(Y_ref)
        @test isapprox(Y_gpu, Y_ref; atol = NORM_ATOL, rtol = NORM_RTOL)
    end
end

@testset "layer_norm!" begin
    rng = MersenneTwister(4)
    @testset "M=$M N=$N" for (M, N) in NORM_SHAPES
        X = randn(rng, Float64, M, N)
        W = randn(rng, Float64, M)
        B = randn(rng, Float64, M)
        Y_ref = layer_norm_ref(X, W, B; eps = 1e-5)
        Y_gpu = gpu_layer_norm(X, W, B; eps = 1e-5)
        @test size(Y_gpu) == size(Y_ref)
        @test isapprox(Y_gpu, Y_ref; atol = NORM_ATOL, rtol = NORM_RTOL)
    end
end

# --- Backward ---------------------------------------------------------------
# rstd (and μ for LayerNorm) are produced by the forward pass; here we compute
# them in the reference and feed them in, so the backward kernel is tested in
# isolation. dX has subtractive terms, so tolerances are a touch looser than fwd.

# RMSNorm grads. dx_i = rstd·(ȳ_i·g_i - rstd²·x_i·(1/M)Σ_k ȳ_k g_k x_k), g=w+offset;
# dw_i = Σ_n ȳ_{i,n}·x_{i,n}·rstd_n.
function rms_norm_bwd_ref(Ȳ, X, W; eps, offset = 0.0)
    M = size(X, 1)
    g = W .+ offset
    rstd = 1 ./ sqrt.(sum(abs2, X; dims = 1) ./ M .+ eps)   # (1, N)
    dd = sum(Ȳ .* (g .* X); dims = 1) ./ M                  # (1, N)
    X̄ = rstd .* (Ȳ .* g .- (rstd .^ 2 .* dd) .* X)
    W̄ = vec(sum(Ȳ .* X .* rstd; dims = 2))
    return X̄, W̄, vec(rstd)
end

# LayerNorm grads. wȳ = w·ȳ, xhat = (x-μ)·rstd;
# dx_i = rstd·(wȳ_i - (1/M)Σwȳ - xhat_i·(1/M)Σ xhat·wȳ); dw = Σ_n ȳ·xhat; db = Σ_n ȳ.
function layer_norm_bwd_ref(Ȳ, X, W; eps)
    M = size(X, 1)
    xc = X .- sum(X; dims = 1) ./ M
    rstd = 1 ./ sqrt.(sum(abs2, xc; dims = 1) ./ M .+ eps)
    xhat = xc .* rstd
    wȳ = W .* Ȳ
    c1 = sum(xhat .* wȳ; dims = 1) ./ M
    c2 = sum(wȳ; dims = 1) ./ M
    X̄ = (wȳ .- (xhat .* c1 .+ c2)) .* rstd
    W̄ = vec(sum(Ȳ .* xhat; dims = 2))
    B̄ = vec(sum(Ȳ; dims = 2))
    return X̄, W̄, B̄, vec(sum(X; dims = 1) ./ M), vec(rstd)
end

function gpu_rms_norm_bwd(Ȳ, X, W, rstd; offset = 0.0f0, T = Float32, kwargs...)
    X̄, W̄ = ∇rms_norm(CuArray(T.(Ȳ)), CuArray(T.(X)), CuArray(T.(W));
        checkpoints = (; Rstd = CuArray(T.(rstd))), offset = Float32(offset), kwargs...)
    return Array(X̄), Array(W̄)
end

function gpu_layer_norm_bwd(Ȳ, X, W, B, mean, rstd; T = Float32, kwargs...)
    X̄, W̄, B̄ = ∇layer_norm(CuArray(T.(Ȳ)), CuArray(T.(X)), CuArray(T.(W)), CuArray(T.(B));
        checkpoints = (; Mean = CuArray(Float32.(mean)), Rstd = CuArray(Float32.(rstd))), kwargs...)
    return Array(X̄), Array(W̄), Array(B̄)
end

const NORM_BWD_ATOL = 1f-4
const NORM_BWD_RTOL = 5f-3

@testset "∇rms_norm" begin
    rng = MersenneTwister(5)
    @testset "M=$M N=$N offset=$offset" for (M, N) in NORM_SHAPES, offset in (0.0, 1.0)
        X = randn(rng, Float64, M, N)
        W = randn(rng, Float64, M)
        Ȳ = randn(rng, Float64, M, N)
        X̄_ref, W̄_ref, rstd = rms_norm_bwd_ref(Ȳ, X, W; eps = 1e-5, offset)
        X̄_gpu, W̄_gpu = gpu_rms_norm_bwd(Ȳ, X, W, rstd; offset)
        @test isapprox(X̄_gpu, X̄_ref; atol = NORM_BWD_ATOL, rtol = NORM_BWD_RTOL)
        @test isapprox(W̄_gpu, W̄_ref; atol = NORM_BWD_ATOL, rtol = NORM_BWD_RTOL)
    end
end

@testset "∇layer_norm" begin
    rng = MersenneTwister(6)
    @testset "M=$M N=$N" for (M, N) in NORM_SHAPES
        X = randn(rng, Float64, M, N)
        W = randn(rng, Float64, M)
        B = randn(rng, Float64, M)
        Ȳ = randn(rng, Float64, M, N)
        X̄_ref, W̄_ref, B̄_ref, mean, rstd = layer_norm_bwd_ref(Ȳ, X, W; eps = 1e-5)
        X̄_gpu, W̄_gpu, B̄_gpu = gpu_layer_norm_bwd(Ȳ, X, W, B, mean, rstd)
        @test isapprox(X̄_gpu, X̄_ref; atol = NORM_BWD_ATOL, rtol = NORM_BWD_RTOL)
        @test isapprox(W̄_gpu, W̄_ref; atol = NORM_BWD_ATOL, rtol = NORM_BWD_RTOL)
        @test isapprox(B̄_gpu, B̄_ref; atol = NORM_BWD_ATOL, rtol = NORM_BWD_RTOL)
    end
end
