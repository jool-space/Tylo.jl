# FlexAttention tests: every variant is checked against a Float64 elementwise
# reference that applies `score(s, b, h, q, kv)` / `mask(b, h, q, kv)` with
# 0-based positions — the same convention the device mods see. Kernels run
# Float32 with TF32 matmuls, hence the loose tolerance.

using Tylo: ᵀ

function flex_ref(Q, K, V; score = nothing, mask = nothing, input_pos = 0)
    Dk, M, H, B = size(Q)
    Dv, N, Hkv, _ = size(V)
    g = H ÷ Hkv
    scale = 1 / sqrt(Dk)
    O = zeros(Float64, Dv, M, H, B)
    for b in 1:B, h in 1:H
        hk = cld(h, g)
        S = ((K[:, :, hk, b])ᵀ * Q[:, :, h, b]) .* scale
        for m in 1:M, n in 1:N
            qpos = m - 1 + input_pos
            kvpos = n - 1
            isnothing(score) || (S[n, m] = score(S[n, m], b, h, qpos, kvpos))
            if !isnothing(mask) && !mask(b, h, qpos, kvpos)
                S[n, m] = -Inf
            end
        end
        for m in 1:M
            mx = maximum(S[:, m])
            if isfinite(mx)
                p = exp.(S[:, m] .- mx)
                p ./= sum(p)
                O[:, m, h, b] = V[:, :, hk, b] * p
            end                                  # fully masked column ⇒ zeros
        end
    end
    return O
end

function gpu_flex(Q, K, V; T = Float32, kwargs...)
    dQ, dK, dV = CuArray(T.(Q)), CuArray(T.(K)), CuArray(T.(V))
    dO = similar(dV, size(V, 1), size(Q, 2), size(Q, 3), size(Q, 4))
    flex_attention!(dO, dQ, dK, dV; kwargs...)
    return Array(dO)
end

# Pair op for PairFeatureScore: broadcast-generic, so the one definition runs
# on device tiles and on host scalars (F = 3 features, PD = 3 outputs).
struct RBFPairOp
    c::Float32
end
function Tylo.pair_feature(op::RBFPairOp, qv::NTuple{3}, kv::NTuple{3})
    # (x .- y) .* (x .- y), not .^2: literal_pow constructs Ref(^) → no Tile IR
    dx = qv[1] .- kv[1]; dy = qv[2] .- kv[2]; dz = qv[3] .- kv[3]
    d² = dx .* dx .+ dy .* dy .+ dz .* dz
    (exp.(-op.c .* d²), 1f0 ./ (1f0 .+ d²), qv[1] .* kv[2])
end

function Tylo.∇pair_feature(op::RBFPairOp, qv::NTuple{3}, kv::NTuple{3},
                                          dphi::NTuple{3})
    dx = qv[1] .- kv[1]; dy = qv[2] .- kv[2]; dz = qv[3] .- kv[3]
    d² = dx .* dx .+ dy .* dy .+ dz .* dz
    φ1 = exp.(-op.c .* d²)
    φ2 = 1f0 ./ (1f0 .+ d²)
    dd = dphi[1] .* (-op.c) .* φ1 .- dphi[2] .* φ2 .* φ2   # ∂L/∂d²
    dq = (2f0 .* dd .* dx .+ dphi[3] .* kv[2],
          2f0 .* dd .* dy,
          2f0 .* dd .* dz)
    dk = (-2f0 .* dd .* dx,
          -2f0 .* dd .* dy .+ dphi[3] .* qv[1],
          -2f0 .* dd .* dz)
    (dq, dk)
end

# Backward reference. `dscore(s0, b, h, q, kv, s̄) -> s̄_pre` is the score
# mod's elementwise VJP at the PRE-mod score s0; closures accumulate param
# gradients into captured arrays.
function flex_bwd_ref(Ō, Q, K, V; score = nothing, dscore = nothing,
                      mask = nothing, input_pos = 0)
    Dk, Mq, H, B = size(Q)
    Dv, N, Hkv, _ = size(V)
    g = H ÷ Hkv
    scale = 1 / sqrt(Dk)
    Q̄ = zero(Q); K̄ = zero(K); V̄ = zero(V)
    for b in 1:B, h in 1:H
        hk = cld(h, g)
        S0 = ((K[:, :, hk, b])ᵀ * Q[:, :, h, b]) .* scale
        S = copy(S0)
        for m in 1:Mq, n in 1:N
            qpos = m - 1 + input_pos; kvpos = n - 1
            isnothing(score) || (S[n, m] = score(S0[n, m], b, h, qpos, kvpos))
            if !isnothing(mask) && !mask(b, h, qpos, kvpos)
                S[n, m] = -Inf
            end
        end
        P = zeros(N, Mq)
        for m in 1:Mq
            mx = maximum(S[:, m])
            isfinite(mx) || continue             # fully masked ⇒ P column = 0
            pcol = exp.(S[:, m] .- mx)
            P[:, m] = pcol ./ sum(pcol)
        end
        ō = Ō[:, :, h, b]
        P̄ = (V[:, :, hk, b])ᵀ * ō
        S̄ = P .* (P̄ .- sum(P .* P̄; dims = 1))
        S̄0 = similar(S̄)
        for m in 1:Mq, n in 1:N
            qpos = m - 1 + input_pos; kvpos = n - 1
            S̄0[n, m] = isnothing(dscore) ? S̄[n, m] :
                       dscore(S0[n, m], b, h, qpos, kvpos, S̄[n, m])
        end
        Q̄[:, :, h, b] = K[:, :, hk, b] * S̄0 .* scale
        K̄[:, :, hk, b] .+= Q[:, :, h, b] * (S̄0)ᵀ .* scale
        V̄[:, :, hk, b] .+= ō * (P)ᵀ
    end
    return Q̄, K̄, V̄
end

flex_inputs(rng; Dk = 64, Dv = 64, M = 128, N = 192, H = 2, Hkv = H, B = 2) = (
    randn(rng, Float64, Dk, M, H, B),
    randn(rng, Float64, Dk, N, Hkv, B),
    randn(rng, Float64, Dv, N, Hkv, B),
)

const FLEX_ATOL = 1f-2
const FLEX_RTOL = 1f-2

≈ₐ(A, B) = isapprox(A, B; atol = FLEX_ATOL, rtol = FLEX_RTOL)

@testset "flex_attention!" begin
    rng = MersenneTwister(7)

    @testset "masks (analytic + dense agree with reference)" begin
        Q, K, V = flex_inputs(rng)
        N = size(K, 2)
        cases = [
            ("full", FullMask(), (b,h,q,kv) -> true),
            ("causal", CausalMask(), (b,h,q,kv) -> q >= kv),
            ("sliding window", SlidingWindowMask(Int32(37)), (b,h,q,kv) -> kv <= q <= kv + 37),
            ("prefix", PrefixMask(Int32(50)), (b,h,q,kv) -> kv < 50),
            ("prefix-lm", prefix_lm(50), (b,h,q,kv) -> q >= kv || kv < 50),
            ("causal ∧ prefix", CausalMask() & PrefixMask(Int32(100)), (b,h,q,kv) -> q >= kv && kv < 100),
        ]
        @testset "$name" for (name, mod, ref) in cases
            O_ref = flex_ref(Q, K, V; mask = ref)
            @test gpu_flex(Q, K, V; mask_mod = mod) ≈ₐ O_ref
            @test gpu_flex(Q, K, V; mask_mod = mod, block_sparse = false) ≈ₐ O_ref
        end
    end

    @testset "input_pos (causal decode chunk)" begin
        Q, K, V = flex_inputs(rng; M = 64, N = 192)
        ip = size(K, 2) - size(Q, 2)            # queries are the last 64 positions
        O_ref = flex_ref(Q, K, V; mask = (b,h,q,kv) -> q >= kv, input_pos = ip)
        @test gpu_flex(Q, K, V; mask_mod = CausalMask(), input_pos = ip) ≈ₐ O_ref
    end

    @testset "uneven lengths (EVEN_K = false)" begin
        Q, K, V = flex_inputs(rng; M = 100, N = 150)
        @test gpu_flex(Q, K, V) ≈ₐ flex_ref(Q, K, V)
        O_ref = flex_ref(Q, K, V; mask = (b,h,q,kv) -> q >= kv)
        @test gpu_flex(Q, K, V; mask_mod = CausalMask()) ≈ₐ O_ref
    end

    @testset "GQA" begin
        Q, K, V = flex_inputs(rng; H = 4, Hkv = 2)
        O_ref = flex_ref(Q, K, V; mask = (b,h,q,kv) -> q >= kv)
        @test gpu_flex(Q, K, V; mask_mod = CausalMask()) ≈ₐ O_ref
    end

    @testset "fully masked (l = 0 guard)" begin
        Q, K, V = flex_inputs(rng; M = 64, N = 64)
        O_ref = zeros(size(V, 1), size(Q, 2), size(Q, 3), size(Q, 4))
        @test gpu_flex(Q, K, V; mask_mod = PrefixMask(Int32(0))) ≈ₐ O_ref
        @test gpu_flex(Q, K, V; mask_mod = PrefixMask(Int32(0)), block_sparse = false) ≈ₐ O_ref
    end

    @testset "score mods" begin
        Q, K, V = flex_inputs(rng)
        H = size(Q, 3)

        @testset "soft cap" begin
            cap = 5.0
            O_ref = flex_ref(Q, K, V; score = (s,b,h,q,kv) -> cap * tanh(s / cap))
            @test gpu_flex(Q, K, V; score_mod = SoftCapScore(Float32(cap))) ≈ₐ O_ref
        end

        @testset "alibi" begin
            slopes = Float32.(2.0 .^ (-(1:H)))
            O_ref = flex_ref(Q, K, V;
                score = (s,b,h,q,kv) -> s + slopes[h] * (q - kv),
                mask = (b,h,q,kv) -> q >= kv)
            @test gpu_flex(Q, K, V;
                score_mod = AliBiScore(CuArray(slopes)), mask_mod = CausalMask()) ≈ₐ O_ref
        end

        @testset "bias (broadcast over heads/batch)" begin
            M, N = size(Q, 2), size(K, 2)
            bias = randn(rng, Float64, N, M, 1, 1)
            O_ref = flex_ref(Q, K, V; score = (s,b,h,q,kv) -> s + bias[kv+1, q+1, 1, 1])
            @test gpu_flex(Q, K, V; score_mod = BiasScore(CuArray(Float32.(bias)))) ≈ₐ O_ref
        end

        @testset "compose (softcap ∘ alibi)" begin
            cap = 5.0
            slopes = Float32.(2.0 .^ (-(1:H)))
            mod = SoftCapScore(Float32(cap)) ∘ AliBiScore(CuArray(slopes))
            O_ref = flex_ref(Q, K, V;
                score = (s,b,h,q,kv) -> cap * tanh((s + slopes[h] * (q - kv)) / cap),
                mask = (b,h,q,kv) -> q >= kv)
            @test gpu_flex(Q, K, V; score_mod = mod, mask_mod = CausalMask()) ≈ₐ O_ref
        end
    end

    @testset "pair features (fused pair bias)" begin
        M, N = 96, 128
        Q, K, V = flex_inputs(rng; M, N)
        H, B = size(Q, 3), size(Q, 4)
        qf = randn(rng, Float64, 3, M, B)
        kf = randn(rng, Float64, 3, N, B)
        proj = randn(rng, Float64, H, 3) .* 0.5
        c = 0.7

        # reference computes the bias with explicit scalar math, NOT pair_feature
        score = (s, b, h, q, kv) -> begin
            d2 = sum(abs2, qf[:, q+1, b] .- kf[:, kv+1, b])
            phi = (exp(-c * d2), 1 / (1 + d2), qf[1, q+1, b] * kf[2, kv+1, b])
            s + sum(proj[h, p] * phi[p] for p in 1:3)
        end
        O_ref = flex_ref(Q, K, V; score, mask = (b,h,q,kv) -> q >= kv)

        mod = PairFeatureScore(RBFPairOp(Float32(c)),
                               CuArray(Float32.(qf)), CuArray(Float32.(kf)),
                               CuArray(Float32.(proj)))
        @test gpu_flex(Q, K, V; score_mod = mod, mask_mod = CausalMask()) ≈ₐ O_ref

        # the same mod evaluates on the host after adapt(Array, ·)
        hmod = adapt(Array, mod)
        @test hscore(hmod, 0.25, 5, 3; h = 2, b = 2) ≈ score(0.25, 2, 2, 5, 3) atol = 1e-4
    end

    @testset "host evaluation via adapt(Array, mod)" begin
        docs = Int32.(vcat(fill(1, 5), fill(2, 7)))
        dm = adapt(Array, DocumentMask(CuArray(docs)))
        @test all(hmask(dm, q, kv) == (docs[q+1] == docs[kv+1]) for q in 0:11, kv in 0:11)

        slopes = Float32.(2.0 .^ -(1:4))
        al = adapt(Array, AliBiScore(CuArray(slopes)))
        @test hscore(al, 0.5f0, 7, 3; h = 2) ≈ 0.5f0 + slopes[2] * 4

        bias = randn(rng, Float32, 8, 8, 1, 1)
        bs = adapt(Array, BiasScore(CuArray(bias)))
        @test hscore(bs, 0f0, 2, 5; h = 3, b = 2) ≈ bias[6, 3, 1, 1]

        # composition adapts recursively
        cm = adapt(Array, CausalMask() & DocumentMask(CuArray(docs)))
        @test hmask(cm, 6, 5) == (docs[7] == docs[6])
        @test !hmask(cm, 5, 6)
    end

    @testset "document mask (data-dependent)" begin
        M = N = 192
        Q, K, V = flex_inputs(rng; M, N)
        docs = Int32.(vcat(fill(1, 70), fill(2, 80), fill(3, 42)))
        keep = (q, kv) -> docs[q+1] == docs[kv+1]
        O_ref = flex_ref(Q, K, V; mask = (b,h,q,kv) -> keep(q, kv))
        dm = DocumentMask(CuArray(docs))

        # data-dependent geometry ⇒ host wrapper runs dense
        @test gpu_flex(Q, K, V; mask_mod = dm) ≈ₐ O_ref

        # same mask through the precomputed coarse BlockMask
        bm = build_block_mask(keep, M, N)
        @test gpu_flex(Q, K, V; mask_mod = dm, block_mask = bm) ≈ₐ O_ref
    end

    @testset "block mask path (causal)" begin
        Q, K, V = flex_inputs(rng; M = 100, N = 150)
        M, N = size(Q, 2), size(K, 2)
        O_ref = flex_ref(Q, K, V; mask = (b,h,q,kv) -> q >= kv)
        bm = build_block_mask((q, kv) -> q >= kv, M, N)
        @test gpu_flex(Q, K, V; mask_mod = CausalMask(), block_mask = bm) ≈ₐ O_ref
    end
end

@testset "∇flex_attention!" begin
    rng = MersenneTwister(11)

    ≈ᵧ(A, B) = isapprox(A, B; atol = 1f-2, rtol = 3f-2)

    function run_flex_bwd(Q, K, V; score_mod = NoOpScore(), mask_mod = FullMask(),
                          ∂score_mod = nothing, kwargs...)
        dQ, dK, dV = map(x -> CuArray(Float32.(x)), (Q, K, V))
        Dv, Mq, H, B = size(V, 1), size(Q, 2), size(Q, 3), size(Q, 4)
        dO = similar(dV, Dv, Mq, H, B)
        cp = allocate_checkpoints(flex_attention!, dQ, dK, dV)
        flex_attention!(dO, dQ, dK, dV; score_mod, mask_mod, checkpoints = cp, kwargs...)
        Ō = randn(rng, Float64, Dv, Mq, H, B)
        dŌ = CuArray(Float32.(Ō))
        dQ̄, dK̄, dV̄ = ∇flex_attention(dŌ, dQ, dK, dV, dO;
                         checkpoints = cp, score_mod, mask_mod, ∂score_mod, kwargs...)
        return Ō, Array(dQ̄), Array(dK̄), Array(dV̄)
    end

    @testset "causal (block_sparse=$bs)" for bs in (true, false)
        Q, K, V = flex_inputs(rng; M = 96, N = 128)
        Ō, Q̄, K̄, V̄ = run_flex_bwd(Q, K, V; mask_mod = CausalMask(), block_sparse = bs)
        Q̄r, K̄r, V̄r = flex_bwd_ref(Ō, Q, K, V; mask = (b,h,q,kv) -> q >= kv)
        @test Q̄ ≈ᵧ Q̄r
        @test K̄ ≈ᵧ K̄r
        @test V̄ ≈ᵧ V̄r
    end

    @testset "GQA + sliding window (analytic skip, atomic K̄/V̄)" begin
        Q, K, V = flex_inputs(rng; M = 128, N = 128, H = 4, Hkv = 2)
        w = 37
        Ō, Q̄, K̄, V̄ = run_flex_bwd(Q, K, V; mask_mod = SlidingWindowMask(Int32(w)))
        Q̄r, K̄r, V̄r = flex_bwd_ref(Ō, Q, K, V; mask = (b,h,q,kv) -> kv <= q <= kv + w)
        @test Q̄ ≈ᵧ Q̄r
        @test K̄ ≈ᵧ K̄r
        @test V̄ ≈ᵧ V̄r
    end

    @testset "uneven lengths (EVEN_K = false)" begin
        Q, K, V = flex_inputs(rng; M = 100, N = 150)
        Ō, Q̄, K̄, V̄ = run_flex_bwd(Q, K, V; mask_mod = CausalMask())
        Q̄r, K̄r, V̄r = flex_bwd_ref(Ō, Q, K, V; mask = (b,h,q,kv) -> q >= kv)
        @test Q̄ ≈ᵧ Q̄r
        @test K̄ ≈ᵧ K̄r
        @test V̄ ≈ᵧ V̄r
    end

    @testset "soft cap (non-additive VJP)" begin
        Q, K, V = flex_inputs(rng; M = 96, N = 128)
        cap = 5.0
        Ō, Q̄, K̄, V̄ = run_flex_bwd(Q, K, V; score_mod = SoftCapScore(Float32(cap)))
        Q̄r, K̄r, V̄r = flex_bwd_ref(Ō, Q, K, V;
            score = (s0,b,h,q,kv) -> cap * tanh(s0 / cap),
            dscore = (s0,b,h,q,kv,s̄) -> s̄ * (1 - tanh(s0 / cap)^2))
        @test Q̄ ≈ᵧ Q̄r
        @test K̄ ≈ᵧ K̄r
        @test V̄ ≈ᵧ V̄r
    end

    @testset "bias grad ($label)" for (label, bh, bb) in
            [("full", nothing, nothing), ("broadcast", 1, 1)]
        Q, K, V = flex_inputs(rng; M = 96, N = 128)
        Mq, N, H, B = size(Q, 2), size(K, 2), size(Q, 3), size(Q, 4)
        bias = randn(rng, Float64, N, Mq, something(bh, H), something(bb, B))
        mod = BiasScore(CuArray(Float32.(bias)))
        ∂mod = grad_shadow(mod)
        B̄ref = zero(bias)
        dscore = (s0,b,h,q,kv,s̄) -> begin
            B̄ref[kv+1, q+1, mod1(h, size(bias, 3)), mod1(b, size(bias, 4))] += s̄
            s̄
        end
        Ō, Q̄, K̄, V̄ = run_flex_bwd(Q, K, V; score_mod = mod, ∂score_mod = ∂mod)
        Q̄r, K̄r, V̄r = flex_bwd_ref(Ō, Q, K, V;
            score = (s0,b,h,q,kv) -> s0 + bias[kv+1, q+1, mod1(h, size(bias, 3)), mod1(b, size(bias, 4))],
            dscore)
        @test Q̄ ≈ᵧ Q̄r
        @test Array(∂mod.bias) ≈ᵧ B̄ref
    end

    @testset "alibi slopes grad" begin
        Q, K, V = flex_inputs(rng; M = 96, N = 128)
        H = size(Q, 3)
        slopes = Float32.(2.0 .^ -(1:H))
        mod = AliBiScore(CuArray(slopes))
        ∂mod = grad_shadow(mod)
        slopes_ref = zeros(H)
        dscore = (s0,b,h,q,kv,s̄) -> (slopes_ref[h] += s̄ * (q - kv); s̄)
        Ō, Q̄, K̄, V̄ = run_flex_bwd(Q, K, V;
            score_mod = mod, ∂score_mod = ∂mod, mask_mod = CausalMask())
        Q̄r, K̄r, V̄r = flex_bwd_ref(Ō, Q, K, V;
            score = (s0,b,h,q,kv) -> s0 + slopes[h] * (q - kv),
            dscore, mask = (b,h,q,kv) -> q >= kv)
        @test Q̄ ≈ᵧ Q̄r
        @test Array(∂mod.slopes) ≈ᵧ slopes_ref
    end

    @testset "compose chain (softcap ∘ alibi)" begin
        Q, K, V = flex_inputs(rng; M = 96, N = 128)
        H = size(Q, 3)
        cap = 5.0
        slopes = Float32.(2.0 .^ -(1:H))
        mod = SoftCapScore(Float32(cap)) ∘ AliBiScore(CuArray(slopes))
        ∂mod = grad_shadow(mod)                  # ∂mod.a is the AliBi shadow
        slopes_ref = zeros(H)
        dscore = (s0,b,h,q,kv,s̄) -> begin
            s2 = s0 + slopes[h] * (q - kv)
            s̄2 = s̄ * (1 - tanh(s2 / cap)^2)
            slopes_ref[h] += s̄2 * (q - kv)
            s̄2
        end
        Ō, Q̄, K̄, V̄ = run_flex_bwd(Q, K, V;
            score_mod = mod, ∂score_mod = ∂mod, mask_mod = CausalMask())
        Q̄r, K̄r, V̄r = flex_bwd_ref(Ō, Q, K, V;
            score = (s0,b,h,q,kv) -> cap * tanh((s0 + slopes[h] * (q - kv)) / cap),
            dscore, mask = (b,h,q,kv) -> q >= kv)
        @test Q̄ ≈ᵧ Q̄r
        @test K̄ ≈ᵧ K̄r
        @test Array(∂mod.a.slopes) ≈ᵧ slopes_ref
    end

    @testset "pair feature grads (∂proj + features)" begin
        # spot-check ∇pair_feature itself against central differences
        op = RBFPairOp(0.7f0)
        qv = (0.3, -1.1, 0.7); kv = (-0.4, 0.2, 1.5); dphi = (0.9, -0.5, 0.4)
        dq, dk = Tylo.∇pair_feature(op, qv, kv, dphi)
        ϵ = 1e-5
        for f in 1:3
            bump = ntuple(i -> i == f ? ϵ : 0.0, 3)
            num_q = sum(dphi .* (Tylo.pair_feature(op, qv .+ bump, kv) .-
                                 Tylo.pair_feature(op, qv .- bump, kv))) / 2ϵ
            num_k = sum(dphi .* (Tylo.pair_feature(op, qv, kv .+ bump) .-
                                 Tylo.pair_feature(op, qv, kv .- bump))) / 2ϵ
            @test dq[f] ≈ num_q atol = 1e-4
            @test dk[f] ≈ num_k atol = 1e-4
        end

        Q, K, V = flex_inputs(rng; M = 96, N = 128)
        Mq, N, H, B = size(Q, 2), size(K, 2), size(Q, 3), size(Q, 4)
        qf = randn(rng, Float64, 3, Mq, B)
        kf = randn(rng, Float64, 3, N, B)
        proj = randn(rng, Float64, H, 3) .* 0.5
        mod = PairFeatureScore(op, CuArray(Float32.(qf)), CuArray(Float32.(kf)),
                               CuArray(Float32.(proj)))
        ∂mod = grad_shadow(mod; feature_grads = true)

        proj_ref = zero(proj); qf_ref = zero(qf); kf_ref = zero(kf)
        pair = (q, kv, b) -> (ntuple(f -> qf[f, q+1, b], 3), ntuple(f -> kf[f, kv+1, b], 3))
        score = (s0,b,h,q,kv) -> begin
            qv, kvv = pair(q, kv, b)
            s0 + sum(proj[h, :] .* Tylo.pair_feature(op, qv, kvv))
        end
        dscore = (s0,b,h,q,kv,s̄) -> begin
            qv, kvv = pair(q, kv, b)
            phi = Tylo.pair_feature(op, qv, kvv)
            proj_ref[h, :] .+= s̄ .* phi
            dphi = ntuple(p -> proj[h, p] * s̄, 3)
            dq, dk = Tylo.∇pair_feature(op, qv, kvv, dphi)
            for f in 1:3
                qf_ref[f, q+1, b] += dq[f]
                kf_ref[f, kv+1, b] += dk[f]
            end
            s̄
        end
        Ō, Q̄, K̄, V̄ = run_flex_bwd(Q, K, V; score_mod = mod, ∂score_mod = ∂mod)
        Q̄r, K̄r, V̄r = flex_bwd_ref(Ō, Q, K, V; score, dscore)
        @test Q̄ ≈ᵧ Q̄r
        @test K̄ ≈ᵧ K̄r
        @test Array(∂mod.pair_proj) ≈ᵧ proj_ref
        @test Array(∂mod.q_features) ≈ᵧ qf_ref
        @test Array(∂mod.k_features) ≈ᵧ kf_ref
    end

    @testset "fully masked ⇒ zero grads" begin
        Q, K, V = flex_inputs(rng; M = 64, N = 64)
        Ō, Q̄, K̄, V̄ = run_flex_bwd(Q, K, V; mask_mod = PrefixMask(Int32(0)))
        @test all(iszero, Q̄)
        @test all(iszero, K̄)
        @test all(iszero, V̄)
    end
end
