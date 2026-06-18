# Split-KV (Flash-Decoding) attention — `decode_attention!`.
#
# CPU reference: single-token (decode) attention with GQA.
#
#   Q (Dk, Heads, Batch)              — one query vector per (head, sequence)
#   K (Dk, SeqLen, Heads_KV, Batch)
#   V (Dv, SeqLen, Heads_KV, Batch)
#   lengths (Batch,)                — valid KV length per sequence
#   O (Dv, Heads, Batch)
#
# Query head h attends to KV head hₖ = (h-1) ÷ G + 1 (G consecutive query heads
# share one KV head — the same fld1 mapping the kernel uses). Computed in
# Float64 so the tolerance budget is spent on the kernel's reduced precision,
# not the reference.
function decode_reference(Q, K, V, lengths)
    Dk, Heads, Batch = size(Q)
    Dv, _, Heads_KV, _ = size(V)
    G = Heads ÷ Heads_KV
    scale = 1 / sqrt(Dk)
    O = zeros(Float64, Dv, Heads, Batch)
    for b in 1:Batch, h in 1:Heads
        L = Int(lengths[b])
        L == 0 && continue                      # empty cache → zero output
        hₖ = (h - 1) ÷ G + 1
        q = Float64.(@view Q[:, h, b])
        scores = [scale * dot(q, Float64.(@view K[:, n, hₖ, b])) for n in 1:L]
        w = exp.(scores .- maximum(scores))
        w ./= sum(w)
        acc = zeros(Float64, Dv)
        for n in 1:L
            acc .+= w[n] .* Float64.(@view V[:, n, hₖ, b])
        end
        O[:, h, b] .= acc
    end
    return O
end

# Run the kernel for host inputs and bring the result back to the host.
function gpu_decode(Q, K, V, lengths; T = Float32, kwargs...)
    Dv = size(V, 1)
    _, Heads, Batch = size(Q)
    dQ = CuArray(T.(Q))
    dK = CuArray(T.(K))
    dV = CuArray(T.(V))
    dO = CuArray(zeros(T, Dv, Heads, Batch))
    dlen = CuArray(Int32.(collect(lengths)))
    decode_attention!(dO, dQ, dK, dV; lengths = dlen, kwargs...)
    return Array(dO)
end

function decode_inputs(rng; Dk, Dv, Heads, Heads_KV, SeqLen, Batch)
    Q = randn(rng, Float64, Dk, Heads, Batch)
    K = randn(rng, Float64, Dk, SeqLen, Heads_KV, Batch)
    V = randn(rng, Float64, Dv, SeqLen, Heads_KV, Batch)
    return Q, K, V
end

# TF32 tensorcore (the default for Float32 input) carries ~10 mantissa bits, so
# the kernel/reference gap is ~1e-2. Loose enough for that, tight enough to
# catch any structural error (wrong indexing, broken combine) — those show up at
# order 0.1–1, not 0.01.
const DECODE_ATOL = 3f-2
const DECODE_RTOL = 3f-2

@testset "decode_attention!" begin
    @testset "vs CPU reference" begin
        rng = MersenneTwister(0)
        # name => (dims, lengths, n_splits, TILE_N)
        cases = [
            # MHA (group size 1), cache fully used, exactly two tiles.
            ("mha, full, 4 splits",
                (Dk=64, Dv=64, Heads=8, Heads_KV=8, SeqLen=128, Batch=2),
                [128, 128], 4, 64),
            # MHA, ragged lengths not aligned to TILE_N.
            ("mha, ragged",
                (Dk=64, Dv=64, Heads=4, Heads_KV=4, SeqLen=200, Batch=3),
                [200, 137, 65], 4, 64),
            # GQA group size 4 packed into the M dimension.
            ("gqa G=4",
                (Dk=64, Dv=64, Heads=8, Heads_KV=2, SeqLen=512, Batch=2),
                [512, 301], 8, 64),
            # GQA group size 8.
            ("gqa G=8",
                (Dk=128, Dv=128, Heads=8, Heads_KV=1, SeqLen=384, Batch=2),
                [384, 200], 6, 128),
            # Short sequences: most splits are empty (exercises the combine
            # finite guard), plus a length-0 sequence (all splits empty → zero).
            ("short + empty",
                (Dk=64, Dv=64, Heads=4, Heads_KV=2, SeqLen=512, Batch=3),
                [30, 1, 0], 8, 64),
            # n_splits = 1 degenerates to one block per (kv_head, sequence).
            ("single split",
                (Dk=64, Dv=64, Heads=4, Heads_KV=4, SeqLen=256, Batch=2),
                [256, 173], 1, 64),
            # More splits than tiles: trailing splits have no work.
            ("oversplit",
                (Dk=64, Dv=64, Heads=4, Heads_KV=2, SeqLen=96, Batch=2),
                [96, 50], 16, 64),
        ]
        for (name, dims, lengths, n_splits, TILE_N) in cases
            Q, K, V = decode_inputs(rng; dims...)
            O_ref = decode_reference(Q, K, V, lengths)
            O_gpu = gpu_decode(Q, K, V, lengths; n_splits, TILE_N)
            @testset "$name" begin
                @test size(O_gpu) == size(O_ref)
                @test isapprox(O_gpu, O_ref; atol = DECODE_ATOL, rtol = DECODE_RTOL)
            end
        end
    end

    # Splitting the KV axis must not change the result. Both runs use the same
    # on-device precision, so this is a tight, precision-robust check that the
    # split forward + log-sum-exp combine reassemble correctly.
    @testset "split invariance" begin
        rng = MersenneTwister(1)
        Q, K, V = decode_inputs(rng;
            Dk=64, Dv=64, Heads=8, Heads_KV=2, SeqLen=500, Batch=3)
        lengths = [500, 333, 64]
        base = gpu_decode(Q, K, V, lengths; n_splits = 1, TILE_N = 64)
        for n_splits in (2, 4, 8, 16)
            split = gpu_decode(Q, K, V, lengths; n_splits, TILE_N = 64)
            @test isapprox(split, base; atol = 1f-3, rtol = 1f-3)
        end
    end
end
