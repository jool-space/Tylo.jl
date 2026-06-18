# Softmax over the contiguous axis (dim 1) of an (M, N) matrix — one column per
# block. `softmax!`/`∇softmax!` dispatch on M: ≤ SOFTMAX_SINGLE_TILE_MAX uses the
# single-tile kernel, above it the streaming (online) kernel. The M values below
# straddle that 4096 threshold so both paths are exercised, including partial
# final tiles (non-power-of-two M) which lean on NegInf/Zero padding.
#
# Softmax is all elementwise + Float32 reductions (no tensorcore), so the only
# gap to the Float64 reference is Float32 accumulation order — hence tight tol.

softmax_ref(X) = (e = exp.(X .- maximum(X; dims = 1)); e ./ sum(e; dims = 1))

# Backward of y = softmax(x): x̄ = y ⊙ (ȳ - Σ y·ȳ), per column.
softmax_bwd_ref(Ȳ, Y) = Y .* (Ȳ .- sum(Y .* Ȳ; dims = 1))

function gpu_softmax(X; T = Float32)
    dX = CuArray(T.(X))
    dY = similar(dX)
    softmax!(dY, dX)
    return Array(dY)
end

function gpu_softmax_bwd(Ȳ, Y; T = Float32)
    dY = CuArray(T.(Y))
    dȲ = CuArray(T.(Ȳ))
    dX̄ = ∇softmax(dȲ, dY)
    return Array(dX̄)
end

const SM_ATOL = 1f-5
const SM_RTOL = 2f-3

@testset "softmax!" begin
    rng = MersenneTwister(2)
    # (M = reduction length, N = number of columns). M spans the single-tile /
    # streaming boundary (SOFTMAX_SINGLE_TILE_MAX = 4096) and includes non-pow2
    # and partial-final-tile sizes.
    cases = [
        (1, 8),         # degenerate: softmax of one element is 1
        (17, 64),       # tiny, non-pow2 → single-tile padded to 32
        (256, 128),
        (4096, 16),     # largest single-tile size
        (4097, 16),     # just over the threshold → streaming, partial last tile
        (5000, 8),      # streaming, ragged
        (16384, 4),     # large streaming
    ]

    @testset "forward M=$M N=$N" for (M, N) in cases
        X = randn(rng, Float64, M, N)
        Y_ref = softmax_ref(X)
        Y_gpu = gpu_softmax(X)
        @test size(Y_gpu) == size(Y_ref)
        @test isapprox(Y_gpu, Y_ref; atol = SM_ATOL, rtol = SM_RTOL)
        # Each column is a probability distribution.
        @test all(isapprox.(vec(sum(Y_gpu; dims = 1)), 1f0; atol = 1f-4))
    end

    @testset "backward M=$M N=$N" for (M, N) in cases
        X = randn(rng, Float64, M, N)
        Y = softmax_ref(X)
        Ȳ = randn(rng, Float64, M, N)
        X̄_ref = softmax_bwd_ref(Ȳ, Y)
        X̄_gpu = gpu_softmax_bwd(Ȳ, Y)
        @test size(X̄_gpu) == size(X̄_ref)
        @test isapprox(X̄_gpu, X̄_ref; atol = SM_ATOL, rtol = SM_RTOL)
    end
end
