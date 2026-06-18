# Autodiff extension tests: drive the public AD entry points — ChainRulesCore
# `rrule` and Mooncake `value_and_gradient!!` — and check they reproduce the
# gradients from the trusted functional `∇*` kernel path (which the per-kernel
# suites already validate against Float64 references). This guards the extension
# *wiring*: tangent slots/order, kwarg routing, the `@from_rrule` registrations,
# and the native Mooncake `rrule!!` that carries learnable `score_mod`
# parameters. The kernel math itself is covered by the other suites.
#
# References use a `ones` output cotangent so the Mooncake `sum`-loss closures
# and the ChainRules pullback share one reference per kernel.

using ChainRulesCore
using ChainRulesCore: rrule, NoTangent, Tangent
using Mooncake

const AD_ATOL = 1.0f-3
const AD_RTOL = 1.0f-2
≈ᵍ(a, b) = isapprox(Array(a), Array(b); atol = AD_ATOL, rtol = AD_RTOL)

cu(x) = CuArray(Float32.(x))
onelike(x) = fill!(similar(x), 1.0f0)

# Mooncake gradient of a `sum`-loss closure w.r.t. its positional args (drops
# the leading function-slot tangent).
function mooncake_grad(f, args...)
    cache = Mooncake.prepare_gradient_cache(f, args...)
    _, grads = Mooncake.value_and_gradient!!(cache, f, args...)
    return grads[2:end]
end

@testset "autodiff extensions" begin
    @test CUDA.functional()

    @testset "rms_norm" begin
        rng = MersenneTwister(1)
        X, W = cu(randn(rng, 64, 32)), cu(randn(rng, 64))
        Ō = cu(randn(rng, 64, 32))
        eps, offset = 1.0f-5, 1.0f0

        cp = allocate_checkpoints(rms_norm!, X, W)
        Y = similar(X); rms_norm!(Y, X, W; eps, offset, checkpoints = cp)
        X̄, W̄ = ∇rms_norm(Ō, X, W; checkpoints = cp, offset)
        X̄₁, W̄₁ = ∇rms_norm(onelike(Y), X, W; checkpoints = cp, offset)

        Yc, pb = rrule(rms_norm, X, W; eps, offset)
        _, X̄c, W̄c = pb(Ō)
        @test Yc ≈ᵍ Y
        @test X̄c ≈ᵍ X̄ && W̄c ≈ᵍ W̄

        dX, dW = mooncake_grad((x, w) -> sum(rms_norm(x, w; eps, offset)), X, W)
        @test dX ≈ᵍ X̄₁ && dW ≈ᵍ W̄₁
    end

    @testset "layer_norm" begin
        rng = MersenneTwister(2)
        X, W, B = cu(randn(rng, 64, 32)), cu(randn(rng, 64)), cu(randn(rng, 64))
        Ō = cu(randn(rng, 64, 32))
        eps = 1.0f-5

        cp = allocate_checkpoints(layer_norm!, X, W, B)
        Y = similar(X); layer_norm!(Y, X, W, B; eps, checkpoints = cp)
        X̄, W̄, B̄ = ∇layer_norm(Ō, X, W, B; checkpoints = cp)
        X̄₁, W̄₁, B̄₁ = ∇layer_norm(onelike(Y), X, W, B; checkpoints = cp)

        Yc, pb = rrule(layer_norm, X, W, B; eps)
        _, X̄c, W̄c, B̄c = pb(Ō)
        @test Yc ≈ᵍ Y
        @test X̄c ≈ᵍ X̄ && W̄c ≈ᵍ W̄ && B̄c ≈ᵍ B̄

        dX, dW, dB = mooncake_grad((x, w, b) -> sum(layer_norm(x, w, b; eps)), X, W, B)
        @test dX ≈ᵍ X̄₁ && dW ≈ᵍ W̄₁ && dB ≈ᵍ B̄₁
    end

    @testset "softmax" begin
        rng = MersenneTwister(3)
        X, Ō = cu(randn(rng, 64, 32)), cu(randn(rng, 64, 32))

        Y = softmax(X)
        X̄ = ∇softmax(Ō, Y)
        X̄₁ = ∇softmax(onelike(Y), Y)

        Yc, pb = rrule(softmax, X)
        _, X̄c = pb(Ō)
        @test Yc ≈ᵍ Y && X̄c ≈ᵍ X̄

        (dX,) = mooncake_grad(x -> sum(softmax(x)), X)
        @test dX ≈ᵍ X̄₁
    end

    @testset "attention (no bias)" begin
        rng = MersenneTwister(4)
        Q, K, V = cu(randn(rng, 64, 64, 2, 1)), cu(randn(rng, 64, 64, 2, 1)), cu(randn(rng, 64, 64, 2, 1))
        Ō = cu(randn(rng, 64, 64, 2, 1))
        causal = true

        cp = allocate_checkpoints(attention!, Q, K, V)
        O = similar(Q); attention!(O, Q, K, V; causal, checkpoints = cp)
        Q̄, K̄, V̄ = ∇attention(Ō, Q, K, V, O; checkpoints = cp, causal)
        Q̄₁, K̄₁, V̄₁ = ∇attention(onelike(O), Q, K, V, O; checkpoints = cp, causal)

        Oc, pb = rrule(attention, Q, K, V; causal)
        _, Q̄c, K̄c, V̄c = pb(Ō)
        @test Oc ≈ᵍ O
        @test Q̄c ≈ᵍ Q̄ && K̄c ≈ᵍ K̄ && V̄c ≈ᵍ V̄

        dQ, dK, dV = mooncake_grad((q, k, v) -> sum(attention(q, k, v; causal)), Q, K, V)
        @test dQ ≈ᵍ Q̄₁ && dK ≈ᵍ K̄₁ && dV ≈ᵍ V̄₁
    end

    @testset "flex_attention (Q/K/V only)" begin
        rng = MersenneTwister(6)
        Q, K, V = cu(randn(rng, 64, 64, 2, 1)), cu(randn(rng, 64, 64, 2, 1)), cu(randn(rng, 64, 64, 2, 1))
        Ō = cu(randn(rng, 64, 64, 2, 1))
        mask_mod = CausalMask()

        cp = allocate_checkpoints(flex_attention!, Q, K, V)
        O = similar(Q); flex_attention!(O, Q, K, V; mask_mod, checkpoints = cp)
        Q̄, K̄, V̄ = ∇flex_attention(Ō, Q, K, V, O; checkpoints = cp, mask_mod)
        Q̄₁, K̄₁, V̄₁ = ∇flex_attention(onelike(O), Q, K, V, O; checkpoints = cp, mask_mod)

        Oc, pb = rrule(flex_attention, Q, K, V; mask_mod)
        _, Q̄c, K̄c, V̄c = pb(Ō)
        @test Oc ≈ᵍ O
        @test Q̄c ≈ᵍ Q̄ && K̄c ≈ᵍ K̄ && V̄c ≈ᵍ V̄

        dQ, dK, dV = mooncake_grad((q, k, v) -> sum(flex_attention(q, k, v; mask_mod)), Q, K, V)
        @test dQ ≈ᵍ Q̄₁ && dK ≈ᵍ K̄₁ && dV ≈ᵍ V̄₁
    end

    # The learnable-score_mod path: ChainRules carries the struct Tangent,
    # Mooncake routes through the native `rrule!!` on `_flex_attention_core`.
    @testset "flex_attention (learnable score_mod)" begin
        rng = MersenneTwister(7)
        Q, K, V = cu(randn(rng, 64, 64, 2, 1)), cu(randn(rng, 64, 64, 2, 1)), cu(randn(rng, 64, 64, 2, 1))
        Ō = cu(randn(rng, 64, 64, 2, 1))
        slopes = cu(randn(rng, 2))
        mask_mod = CausalMask()
        mod = AliBiScore(slopes)

        cp = allocate_checkpoints(flex_attention!, Q, K, V)
        O = similar(Q); flex_attention!(O, Q, K, V; score_mod = mod, mask_mod, checkpoints = cp)
        ∂mod = grad_shadow(mod)
        Q̄, K̄, V̄ = ∇flex_attention(Ō, Q, K, V, O;
            checkpoints = cp, score_mod = mod, mask_mod, ∂score_mod = ∂mod)
        slopes̄ = copy(∂mod.slopes)

        # ChainRules on the positional core (carries the score_mod Tangent).
        Oc, pb = rrule(Tylo._flex_attention, Q, K, V, mod; mask_mod)
        _, Q̄c, K̄c, V̄c, modc = pb(Ō)
        @test Oc ≈ᵍ O
        @test Q̄c ≈ᵍ Q̄ && K̄c ≈ᵍ K̄ && V̄c ≈ᵍ V̄
        @test modc.slopes ≈ᵍ slopes̄

        # Mooncake: differentiate w.r.t. the slopes array, reconstructing the mod
        # inside the closure so the native rrule!!'s struct tangent flows back.
        ∂mod₁ = grad_shadow(mod)
        Q̄₁, K̄₁, V̄₁ = ∇flex_attention(onelike(O), Q, K, V, O;
            checkpoints = cp, score_mod = mod, mask_mod, ∂score_mod = ∂mod₁)
        slopes̄₁ = copy(∂mod₁.slopes)
        dQ, dK, dV, dS = mooncake_grad(
            (q, k, v, s) -> sum(Tylo._flex_attention(q, k, v, AliBiScore(s); mask_mod)),
            Q, K, V, slopes)
        @test dQ ≈ᵍ Q̄₁ && dK ≈ᵍ K̄₁ && dV ≈ᵍ V̄₁
        @test dS ≈ᵍ slopes̄₁
    end

    # BFloat16: the AD rules now register over `AbstractFloat`, not just
    # `Base.IEEEFloat`, so bf16 (which is `<: AbstractFloat` but not `IEEEFloat`)
    # routes through the rules instead of falling through to differentiating the
    # kernel internals. ChainRules rrules are untyped → always exercise bf16.
    # The Mooncake path additionally needs a scalar tangent interface for the
    # bf16 *storage* type, which only exists when `BFloat16s.BFloat16 ===
    # Core.BFloat16` (gated on arch + LLVM version inside BFloat16s); guard the
    # Mooncake asserts on that capability. Loose tol: bf16 has ~3 decimal digits.
    @testset "BFloat16 (registered over AbstractFloat, not IEEEFloat)" begin
        bf(x) = CuArray(Tylo.BFloat16.(x))
        ≈ᵇ(a, b) = isapprox(Float32.(Array(a)), Float32.(Array(b)); atol = 5.0f-2, rtol = 5.0f-2)
        rng = MersenneTwister(8)
        mooncake_bf16 = try
            Mooncake.tangent_type(Tylo.BFloat16) === Tylo.BFloat16
        catch
            false
        end

        X, W = bf(randn(rng, 64, 32)), bf(randn(rng, 64))
        Ō = bf(randn(rng, 64, 32))
        eps, offset = 1.0f-5, 1.0f0
        cp = allocate_checkpoints(rms_norm!, X, W)
        Y = similar(X); rms_norm!(Y, X, W; eps, offset, checkpoints = cp)
        X̄, W̄ = ∇rms_norm(Ō, X, W; checkpoints = cp, offset)

        # ChainRules (untyped — always fires for bf16).
        _, X̄c, W̄c = rrule(rms_norm, X, W; eps, offset)[2](Ō)
        @test X̄c ≈ᵇ X̄ && W̄c ≈ᵇ W̄

        if mooncake_bf16
            X̄₁, W̄₁ = ∇rms_norm(onelike(Y), X, W; checkpoints = cp, offset)
            dX, dW = mooncake_grad((x, w) -> sum(rms_norm(x, w; eps, offset)), X, W)
            @test dX ≈ᵇ X̄₁ && dW ≈ᵇ W̄₁

            # native learnable-score_mod path in bf16
            Q, K, V = bf(randn(rng, 64, 64, 2, 1)), bf(randn(rng, 64, 64, 2, 1)), bf(randn(rng, 64, 64, 2, 1))
            slopes = bf(randn(rng, 2))
            mask_mod = CausalMask()
            mod = AliBiScore(slopes)
            cpf = allocate_checkpoints(flex_attention!, Q, K, V)
            O = similar(Q); flex_attention!(O, Q, K, V; score_mod = mod, mask_mod, checkpoints = cpf)
            ∂mod = grad_shadow(mod)
            Q̄₁, K̄₁, V̄₁ = ∇flex_attention(onelike(O), Q, K, V, O;
                checkpoints = cpf, score_mod = mod, mask_mod, ∂score_mod = ∂mod)
            slopes̄₁ = copy(∂mod.slopes)
            dQ, dK, dV, dS = mooncake_grad(
                (q, k, v, s) -> sum(Tylo._flex_attention(q, k, v, AliBiScore(s); mask_mod)),
                Q, K, V, slopes)
            @test dQ ≈ᵇ Q̄₁ && dK ≈ᵇ K̄₁ && dV ≈ᵇ V̄₁
            @test dS ≈ᵇ slopes̄₁
        else
            @test_skip false
        end
    end
end
