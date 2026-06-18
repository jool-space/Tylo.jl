# Training attention (attention! / ∇attention!) against Float64 references.
# The backward reference is the standard attention gradient: with
# S = KᵀQ·scale, P = softmax(S), O = V·P and incoming Ō:
#   P̄ = Vᵀ·Ō,  S̄ = P ∘ (P̄ - colsum(P ∘ P̄)),
#   Q̄ = K·S̄·scale,  K̄ = Q·S̄ᵀ·scale,  V̄ = Ō·Pᵀ.

using Tylo: ᵀ

function attn_ref(Q, K, V; causal = false)
    Dk, M, H, B = size(Q)
    Dv, N, Hkv, _ = size(V)
    g = H ÷ Hkv
    scale = 1 / sqrt(Dk)
    O = zeros(Float64, Dv, M, H, B)
    for b in 1:B, h in 1:H
        hk = cld(h, g)
        S = ((K[:, :, hk, b])ᵀ * Q[:, :, h, b]) .* scale
        if causal
            for m in 1:M, n in 1:N
                n > m && (S[n, m] = -Inf)
            end
        end
        P = exp.(S .- maximum(S; dims = 1))
        P ./= sum(P; dims = 1)
        O[:, :, h, b] = V[:, :, hk, b] * P
    end
    return O
end

function attn_bwd_ref(Ō, Q, K, V; causal = false)
    Dk, M, H, B = size(Q)
    Dv, N, Hkv, _ = size(V)
    g = H ÷ Hkv
    scale = 1 / sqrt(Dk)
    Q̄ = zero(Q); K̄ = zero(K); V̄ = zero(V)
    for b in 1:B, h in 1:H
        hk = cld(h, g)
        S = ((K[:, :, hk, b])ᵀ * Q[:, :, h, b]) .* scale
        if causal
            for m in 1:M, n in 1:N
                n > m && (S[n, m] = -Inf)
            end
        end
        P = exp.(S .- maximum(S; dims = 1))
        P ./= sum(P; dims = 1)

        ō = Ō[:, :, h, b]
        P̄ = (V[:, :, hk, b])ᵀ * ō
        S̄ = P .* (P̄ .- sum(P .* P̄; dims = 1))
        Q̄[:, :, h, b] = K[:, :, hk, b] * S̄ .* scale
        K̄[:, :, hk, b] .+= Q[:, :, h, b] * (S̄)ᵀ .* scale
        V̄[:, :, hk, b] .+= ō * (P)ᵀ
    end
    return Q̄, K̄, V̄
end

attn_inputs(rng; Dk = 64, Dv = 64, M = 96, N = 128, H = 4, Hkv = H, B = 2) = (
    randn(rng, Float64, Dk, M, H, B),
    randn(rng, Float64, Dk, N, Hkv, B),
    randn(rng, Float64, Dv, N, Hkv, B),
)

const ATTN_ATOL = 1f-2
const ATTN_RTOL = 3f-2

≈ᵦ(A, B) = isapprox(A, B; atol = ATTN_ATOL, rtol = ATTN_RTOL)

@testset "attention!" begin
    rng = MersenneTwister(8)

    @testset "causal=$causal" for causal in (false, true)
        Q, K, V = attn_inputs(rng)
        dQ, dK, dV = map(x -> CuArray(Float32.(x)), (Q, K, V))
        dO = similar(dQ, size(V, 1), size(Q, 2), size(Q, 3), size(Q, 4))
        attention!(dO, dQ, dK, dV; causal)
        @test Array(dO) ≈ᵦ attn_ref(Q, K, V; causal)
    end
end

@testset "∇attention!" begin
    rng = MersenneTwister(9)

    function run_bwd(Q, K, V; causal)
        dQ, dK, dV = map(x -> CuArray(Float32.(x)), (Q, K, V))
        Dv, M, H, B = size(V, 1), size(Q, 2), size(Q, 3), size(Q, 4)
        dO = similar(dQ, Dv, M, H, B)
        cp = allocate_checkpoints(attention!, dQ, dK, dV)
        attention!(dO, dQ, dK, dV; causal, checkpoints = cp)

        Ō = randn(rng, Float64, Dv, M, H, B)
        dŌ = CuArray(Float32.(Ō))
        dQ̄, dK̄, dV̄ = ∇attention(dŌ, dQ, dK, dV, dO; checkpoints = cp, causal)

        refs = attn_bwd_ref(Ō, Q, K, V; causal)
        outs = (Array(dQ̄), Array(dK̄), Array(dV̄))
        return outs, refs
    end

    @testset "MHA causal=$causal" for causal in (false, true)
        Q, K, V = attn_inputs(rng)
        (Q̄, K̄, V̄), (Q̄r, K̄r, V̄r) = run_bwd(Q, K, V; causal)
        @test Q̄ ≈ᵦ Q̄r
        @test K̄ ≈ᵦ K̄r
        @test V̄ ≈ᵦ V̄r
    end

    @testset "GQA (atomic K̄/V̄ accumulation)" begin
        Q, K, V = attn_inputs(rng; H = 4, Hkv = 2)
        (Q̄, K̄, V̄), (Q̄r, K̄r, V̄r) = run_bwd(Q, K, V; causal = true)
        @test Q̄ ≈ᵦ Q̄r
        @test K̄ ≈ᵦ K̄r
        @test V̄ ≈ᵦ V̄r
    end
end
