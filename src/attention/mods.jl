export FullMask, CausalMask, SlidingWindowMask, PrefixMask, DocumentMask
export AndMask, OrMask, prefix_lm
export NoOpScore, SoftCapScore, AliBiScore, BiasScore, ComposeScore
export BlockMask, build_block_mask
export hmask, hscore
export grad_shadow

# FlexAttention — variant library
#
# The attention *variant* is a `score_mod` / `mask_mod` value passed to the
# kernel. Each variant is a small CALLABLE STRUCT, not a raw closure:
#   * no-field struct  -> ghost type    -> zero kernel params, fully inlined
#   * struct w/ fields -> `_flatten_static!` recurses fields (a TileArray
#     field flattens to ptr/sizes/strides; scalar field -> scalar param)
#   * it is a distinct TYPE -> dispatchable, so composition is struct nesting
#     and analytic block heuristics attach as methods (`geometry`, `kv_lo`, …)
#
# Device call convention (inside the kernel, all tiles):
#   score tile `s` : (TILE_N, TILE_M) accumulator eltype
#   q : (1, TILE_M)  Int32   0-based query positions (lazy BlockPos)
#   kv: (TILE_N,)    Int32   0-based key positions   (lazy BlockPos)
#   b, h : Int32 scalars (1-based batch, head)
#
#   score_mod(s, b, h, q, kv) -> (TILE_N, TILE_M) score tile
#   mask_mod(b, h, q, kv)     -> (TILE_N, TILE_M) Bool tile (via broadcast)
#
# Mods hold their arrays as PLAIN arrays (CuArray on the way to a launch);
# `cuTileconvert` recurses through `Adapt.adapt_structure` at launch time and
# wraps every array field in a `TileArray`. The same recursion runs in the
# other direction: `adapt(Array, mod)` yields a host-indexable copy of ANY
# mod, so `hmask`/`hscore` give CPU references for the exact mod the kernel
# ran (mods dispatch device vs host on tile-vs-scalar positions).

abstract type MaskMod end
abstract type ScoreMod end

# One adapt rule for every mod: rebuild the struct with adapted fields. Type
# parameters are re-inferred from the fields, so mods keep static information
# in `Val` FIELDS (see PairFeatureScore) rather than free type parameters.
# Ghost mods (no fields) reconstruct as themselves.
function Adapt.adapt_structure(to, m::Union{MaskMod, ScoreMod})
    T = typeof(m)
    T.name.wrapper(ntuple(i -> adapt(to, getfield(m, i)), Val(fieldcount(T)))...)
end

"""
    hmask(mask_mod, q, kv; b = 1, h = 1) -> Bool

Evaluate a mask mod on the host at 0-based scalar positions — for CPU
references and tests. For mods carrying device arrays, evaluate a host copy:
`hmask(adapt(Array, mod), q, kv)`.
"""
hmask(m::MaskMod, q, kv; b = 1, h = 1)     = m(Int32(b), Int32(h), q, kv)

"""
    hscore(score_mod, s, q, kv; b = 1, h = 1)

Evaluate a score mod on the host: the modified score for scalar score `s` at
0-based positions `(q, kv)`. For mods carrying device arrays, evaluate a host
copy: `hscore(adapt(Array, mod), s, q, kv)`.
"""
hscore(m::ScoreMod, s, q, kv; b = 1, h = 1) = m(s, Int32(b), Int32(h), q, kv)

# Indexed read that is `ct.gather` on a device TileArray and plain `getindex`
# on a host array — lets one mod body serve both worlds.
@inline vgather(a::TileArray, i) = ct.gather(a, i)
@inline vgather(a::AbstractArray, i) = a[i]

# Lazy block position descriptor passed to mask_mod / score_mod in place of a
# concrete position tile. Mods that just broadcast (`q .>= kv`, `kv .+ w`, …)
# get materialization for free via `Base.broadcastable`. Mods that want the
# coalesced `ct.load` path (e.g. BiasScore) peek at `block_idx` directly.
#
# Axis ∈ (1, 2):  1 ⇒ kv-axis ⇒ shape (TileSize,)
#                 2 ⇒ q-axis  ⇒ shape (1, TileSize)
# TileSize is a type parameter so `ct.arange(TileSize)` stays static.
struct BlockPos{Axis, TileSize}
    block_idx::Int32        # 0-based block index
    offset::Int32           # added to every position (input_pos for q, 0 for kv)
end
BlockPos{Axis, TileSize}(block_idx) where {Axis, TileSize} =
    BlockPos{Axis, TileSize}(Int32(block_idx), 0i32)

@inline function as_tile(p::BlockPos{Axis, TileSize}) where {Axis, TileSize}
    v = p.block_idx * Int32(TileSize) .+ ct.arange(TileSize) .- 1i32 .+ p.offset
    Axis == 1 ? v : reshape(v, (1, TileSize))
end

Base.Broadcast.broadcastable(p::BlockPos) = as_tile(p)

tile_size(::BlockPos{Axis, TileSize}) where {Axis, TileSize} = TileSize

## --- mask modifiers ---------------------------------------------------------

"""
    FullMask()

No masking — every query attends to every key. The default `mask_mod`.
"""
struct FullMask <: MaskMod end
(::FullMask)(b, h, q, kv) = (kv .>= 0i32) .& (q .>= 0i32)  # 0-based ⇒ always true

"""
    CausalMask()

Each query attends to keys at or before its own position (`q ≥ kv`).
Positions are absolute, so `input_pos` shifts the queries.
"""
struct CausalMask <: MaskMod end
(::CausalMask)(b, h, q, kv) = q .>= kv

"""
    SlidingWindowMask(w)

Causal attention within a window of `w`: `q - w ≤ kv ≤ q`.
"""
struct SlidingWindowMask <: MaskMod
    w::Int32
end
(m::SlidingWindowMask)(b, h, q, kv) = (q .>= kv) .& (q .<= kv .+ m.w)

"""
    PrefixMask(len)

Keys at positions `kv < len` are visible to every query — a building block;
see [`prefix_lm`](@ref).
"""
struct PrefixMask <: MaskMod
    len::Int32
end
(m::PrefixMask)(b, h, q, kv) = kv .< m.len

"""
    DocumentMask(doc)
    DocumentMask(doc_q, doc_kv)

Attention only within the same document: 0-based position `p` belongs to
document `doc[p + 1]`, and a pair attends iff its ids match. Pass separate
vectors when queries and keys come from different sequences.
"""
struct DocumentMask{A, B} <: MaskMod
    doc_q::A
    doc_kv::B
end
DocumentMask(doc::AbstractVector) = DocumentMask(doc, doc)
(m::DocumentMask)(b, h, q, kv) =
    vgather(m.doc_q, q .+ 1i32) .== vgather(m.doc_kv, kv .+ 1i32)

"""
    AndMask(a, b)

Conjunction of two masks — attend where both allow. Construct with `a & b`.
"""
struct AndMask{A<:MaskMod, B<:MaskMod} <: MaskMod
    a::A; b::B
end
(m::AndMask)(b, h, q, kv) = m.a(b, h, q, kv) .& m.b(b, h, q, kv)

"""
    OrMask(a, b)

Disjunction of two masks — attend where either allows. Construct with `a | b`.
"""
struct OrMask{A<:MaskMod, B<:MaskMod} <: MaskMod
    a::A; b::B
end
(m::OrMask)(b, h, q, kv) = m.a(b, h, q, kv) .| m.b(b, h, q, kv)

Base.:&(a::MaskMod, b::MaskMod) = AndMask(a, b)
Base.:|(a::MaskMod, b::MaskMod) = OrMask(a, b)

"""
    prefix_lm(len)

Prefix-LM mask: bidirectional attention over the first `len` positions,
causal after. Equals `CausalMask() | PrefixMask(len)`.
"""
prefix_lm(len) = CausalMask() | PrefixMask(Int32(len))

## --- score modifiers --------------------------------------------------------

"""
    NoOpScore()

Identity score mod — leaves scores untouched. The default `score_mod`.
"""
struct NoOpScore <: ScoreMod end
(::NoOpScore)(s, b, h, q, kv) = s

"""
    SoftCapScore(cap)

Logit soft-capping: `s -> cap * tanh(s / cap)`. `cap` receives no gradient.
"""
struct SoftCapScore <: ScoreMod
    cap::Float32
end
(m::SoftCapScore)(s, b, h, q, kv) = m.cap .* tanh.(s ./ m.cap)

"""
    AliBiScore(slopes)

ALiBi positional bias: adds `slopes[h] * (q - kv)` to the scores, one slope
per query head.
"""
struct AliBiScore{A} <: ScoreMod
    slopes::A
end
(m::AliBiScore)(s, b, h, q, kv) = s .+ m.slopes[h] .* Float32.(q .- kv)

"""
    BiasScore(bias)

Adds `bias` to the scores, broadcast over heads/batch when those dims are
smaller. Indexed by local query position (`input_pos` does not shift it).

  * `bias`: `(SeqLen_K, SeqLen_Q, BiasHeads, BiasBatch)`
"""
struct BiasScore{A} <: ScoreMod
    bias::A
    nheads::Int32
    nbatch::Int32
end
function BiasScore(bias::AbstractArray)
    @assert ndims(bias) == 4 "BiasScore expects (N_kv, M_q, BIAS_HEADS, BIAS_BATCH)"
    BiasScore(bias, Int32(size(bias, 3)), Int32(size(bias, 4)))
end
function (m::BiasScore)(s, b, h, q::BlockPos, kv::BlockPos)
    hᵇ = mod1(h, m.nheads)
    bᵇ = mod1(b, m.nbatch)
    bias = ct.load(m.bias, (kv.block_idx + 1i32, q.block_idx + 1i32, hᵇ, bᵇ),
                   (tile_size(kv), tile_size(q));
                   padding_mode=ct.PaddingMode.Zero)
    s .+ (bias → eltype(s))
end
(m::BiasScore)(s, b, h, q::Integer, kv::Integer) =
    s + m.bias[kv + 1, q + 1, mod1(h, m.nheads), mod1(b, m.nbatch)]

"""
    ComposeScore(a, b)

Apply score mod `a`, then `b`. Construct with `b ∘ a`.
"""
struct ComposeScore{A<:ScoreMod, B<:ScoreMod} <: ScoreMod
    a::A; b::B
end
(m::ComposeScore)(s, b, h, q, kv) = m.b(m.a(s, b, h, q, kv), b, h, q, kv)

Base.:∘(b::ScoreMod, a::ScoreMod) = ComposeScore(a, b)

## --- score modifier VJPs (for ∇flex_attention!) -----------------------------
#
# Recompute-based pullbacks: the backward kernel rebuilds the PRE-mod score
# tile `s` per block anyway, so a mod's VJP receives (s, s̄_out) and returns
# s̄_in — no tape, no closures. Parameter gradients accumulate atomically into
# `∂m`, a GRADIENT SHADOW of the mod (same struct type, arrays zeroed);
# `∂m === nothing` skips parameter gradients and folds away at compile time.
# Masked and padded entries arrive with s̄ = 0, so they contribute nothing.
#
# Custom mods with parameters implement
#   ∇score(m::MyMod, ∂m, s, b, h, q, kv, s̄) -> s̄_in
# (No generic fallback ON PURPOSE — see the `pair_feature` docstring.)

"""
    grad_shadow(x)

Build a zeroed gradient accumulator for `x` — the *accumulate* counterpart to a
[`Duplicated`](@ref) shadow (which is overwrite, hence uninitialized). The single
factory for gradient containers, dispatched on what `x` is:

  * an array → a zeroed copy (`fill!(similar(x), 0)`);
  * a score mod → the same struct with every field recursively `grad_shadow`ed
    (array fields zeroed, composed sub-mods recursed, non-differentiable fields
    such as `BiasScore.nheads` left as-is);
  * anything else → returned unchanged (a non-differentiable leaf).

Pass a score-mod shadow as `∂score_mod` to [`∇flex_attention!`](@ref) and read
the accumulated gradients from its fields afterwards. Gradients accumulate in
place, so a shadow is valid for ONE backward call — rebuild (or re-zero) to reuse.
"""
grad_shadow(x::AbstractArray) = fill!(similar(x), 0)
function grad_shadow(m::ScoreMod)
    T = typeof(m)
    T.name.wrapper(ntuple(i -> grad_shadow(getfield(m, i)), Val(fieldcount(T)))...)
end
grad_shadow(x) = x

∇score(::NoOpScore, ∂m, s, b, h, q, kv, s̄) = s̄

function ∇score(m::SoftCapScore, ∂m, s, b, h, q, kv, s̄)
    th = tanh.(s ./ m.cap)
    s̄ .* (1f0 .- th .* th)                    # `cap` itself gets no gradient
end

# Additive mods pass s̄ through unchanged; they only accumulate param grads.
function ∇score(m::AliBiScore, ∂m, s, b, h, q, kv, s̄)
    if !isnothing(∂m)
        g = sum(sum(s̄ .* Float32.(q .- kv), dims=1), dims=2)
        atomic_add_tile(∂m.slopes, (h,),
                        convert(Tile{eltype(∂m.slopes)}, reshape(g, (1,))))
    end
    s̄
end

function ∇score(m::BiasScore, ∂m, s, b, h, q::BlockPos, kv::BlockPos, s̄)
    if !isnothing(∂m)
        atomic_add_tile(∂m.bias,
            (kv.block_idx + 1i32, q.block_idx + 1i32,
             mod1(h, m.nheads), mod1(b, m.nbatch)),
            convert(Tile{eltype(∂m.bias)}, s̄))
    end
    s̄
end

function ∇score(m::ComposeScore, ∂m, s, b, h, q, kv, s̄)
    s₂ = m.a(s, b, h, q, kv)                  # recompute the intermediate
    s̄₂ = ∇score(m.b, isnothing(∂m) ? nothing : ∂m.b, s₂, b, h, q, kv, s̄)
    ∇score(m.a, isnothing(∂m) ? nothing : ∂m.a, s, b, h, q, kv, s̄₂)
end


#=============================================================================
 Analytic block classification — a soundness-preserving lattice.

 For query block `i` (0-based, rows [i*TM, i*TM+TM-1] .+ input_pos) decide,
 per KV block, "masked-out / partial / full". Soundness contract:
   kv_block_range : superset of blocks holding ANY unmasked element
                    (over-inclusion costs speed, never correctness).
   kv_block_full  : conservative — true only when PROVABLY full.

 LATTICE (tightest → most general; each tier is a sound fallback for the
 next, so a mask is correct with no annotation and faster once it adds one):

   AffineBand    leaf declares a per-query allowed-kv interval [kv_lo, kv_hi]
                 that is affine & monotone non-decreasing in q ⇒ exact
                 closed-form range & full from ONE generic driver.
   DataDependent geometry opaque (DocumentMask, anything unannotated) ⇒
                 visit-all / never-full. THIS is the seam the host-built
                 coarse BlockMask index list replaces.

 Combinators compose the lattice RECURSIVELY:
 AndMask = ∩ of children's block sets, OrMask = enclosing interval ⊇ ∪.
=============================================================================#

abstract type MaskGeometry end
struct AffineBand    <: MaskGeometry end
struct DataDependent <: MaskGeometry end

geometry(::Any)               = DataDependent()
geometry(::FullMask)          = AffineBand()
geometry(::CausalMask)        = AffineBand()
geometry(::SlidingWindowMask) = AffineBand()
geometry(::PrefixMask)        = AffineBand()

# Per-query allowed-kv interval [kv_lo(q), kv_hi(q)] (inclusive, 0-based),
# monotone non-decreasing in q — the "affine band" the driver consumes.
kv_lo(::FullMask, q, N)           = 0i32
kv_hi(::FullMask, q, N)           = N - 1i32
kv_lo(::CausalMask, q, N)         = 0i32
kv_hi(::CausalMask, q, N)         = q                  # kv ≤ q
kv_lo(m::SlidingWindowMask, q, N) = q - m.w            # q-w ≤ kv ≤ q
kv_hi(m::SlidingWindowMask, q, N) = q
kv_lo(::PrefixMask, q, N)         = 0i32
kv_hi(m::PrefixMask, q, N)        = m.len - 1i32       # kv < len

# Leaf entry points dispatch on the geometry trait.
# Args: i = 0-based query block, (TM, TN) tile sizes, M/N = q/kv lengths,
# ip = input_pos. Range is (lo, hi) in 0-based blocks, hi EXCLUSIVE.
kv_block_range(mask, args...) = kv_block_range(geometry(mask), mask, args...)
kv_block_full(mask, args...) = kv_block_full(geometry(mask), mask, args...)

# DataDependent: opaque ⇒ safe fallback.
kv_block_range(::DataDependent, mask, i, TM, TN, M, N, ip) = (0i32, cld(N, TN))
kv_block_full(::DataDependent, mask, i, j, TM, TN, M, N, ip) = false

# AffineBand: monotone band ⇒ union over the query block = [lo(qlo), hi(qhi)],
# intersection over the block = [lo(qhi), hi(qlo)] (opposite endpoints).
function kv_block_range(::AffineBand, mask, i, TM, TN, M, N, ip)
    qlo = i * TM + ip; qhi = qlo + TM - 1i32
    klo = max(0i32,     kv_lo(mask, qlo, N))
    khi = min(N - 1i32, kv_hi(mask, qhi, N))
    khi < klo ? (0i32, 0i32) : (fld(klo, TN), fld(khi, TN) + 1i32)
end
function kv_block_full(::AffineBand, mask, i, j, TM, TN, M, N, ip)
    qlo = i * TM + ip; qhi = qlo + TM - 1i32
    (j * TN >= kv_lo(mask, qhi, N)) &
    (j * TN + TN - 1i32 <= kv_hi(mask, qlo, N))
end

function kv_block_range(m::AndMask, args...)
    la, ha = kv_block_range(m.a, args...)
    lb, hb = kv_block_range(m.b, args...)
    (max(la, lb), min(ha, hb))                            # AND ⇒ intersection
end
kv_block_full(m::AndMask, args...) =
    kv_block_full(m.a, args...) & kv_block_full(m.b, args...)

function kv_block_range(m::OrMask, args...)
    la, ha = kv_block_range(m.a, args...)
    lb, hb = kv_block_range(m.b, args...)
    (min(la, lb), max(ha, hb))                            # OR ⇒ interval ⊇ union
end
kv_block_full(m::OrMask, args...) =
    kv_block_full(m.a, args...) | kv_block_full(m.b, args...)

# Does in-kernel analytic skip actually constrain anything for this mask?
# A `DataDependent` leaf gives visit-all/never-full — the analytic path is
# then pure overhead, so the host runs dense instead and expects a BlockMask
# for real sparsity. AndMask helps if EITHER child constrains (intersection);
# OrMask only if BOTH (enclosing interval).
analytic_useful(m)          = !(geometry(m) isa DataDependent)
analytic_useful(m::AndMask) = analytic_useful(m.a) || analytic_useful(m.b)
analytic_useful(m::OrMask)  = analytic_useful(m.a) && analytic_useful(m.b)


#=============================================================================
 General host-built coarse BlockMask.

 The `DataDependent` tier's job, made real: for masks with no analytic
 geometry (DocumentMask, paged, learned), precompute per query-block the
 list of KV blocks that hold ANY unmasked element (skip the rest) and which
 of those are FULLY unmasked (skip mask_mod there). Built on the HOST from a
 host predicate `keep(q, kv)` — pass a lambda, or reuse the device mod via
 `(q, kv) -> hmask(adapt(Array, mod), q, kv)`. Memory O(⌈M/B⌉·⌈N/B⌉) ≪ O(M·N).
=============================================================================#

"""
    BlockMask

Precomputed coarse block sparsity: per query block, the KV blocks holding any
unmasked element, and which of those are fully unmasked. Built with
[`build_block_mask`](@ref); pass as `block_mask` to [`flex_attention!`](@ref).
For masks with no analytic geometry (e.g. [`DocumentMask`](@ref)).
No backward support.
"""
struct BlockMask
    count :: CuArray{Int32,1}    # [n_qb]        KV blocks to visit per q-block
    idx   :: CuArray{Int32,2}    # [n_kb, n_qb]  0-based KV block ids (padded)
    full  :: CuArray{Int32,2}    # [n_kb, n_qb]  1 ⇒ that block is fully unmasked
end

"""
    build_block_mask(keep, n_q, n_kv; TILE_M = 64, TILE_N = 64, input_pos = 0) -> BlockMask

Build a [`BlockMask`](@ref) on the host from the predicate `keep(q, kv) -> Bool`
(0-based absolute positions). `keep` must agree with the device `mask_mod` —
e.g. `(q, kv) -> hmask(adapt(Array, mod), q, kv)` — and `TILE_M`/`TILE_N` must
match the kernel launch.
"""
function build_block_mask(keep, n_q, n_kv; TILE_M::Int=64, TILE_N::Int=64, input_pos::Int=0)
    nqb, nkb = cld(n_q, TILE_M), cld(n_kv, TILE_N)
    idx  = zeros(Int32, nkb, nqb)
    full = zeros(Int32, nkb, nqb)
    cnt  = zeros(Int32, nqb)
    for i in 1:nqb
        qlo = (i-1)*TILE_M; qhi = min(i*TILE_M, n_q) - 1
        slot = 0
        for j in 1:nkb
            klo = (j-1)*TILE_N; khi = min(j*TILE_N, n_kv) - 1
            anyk = false; allk = true
            for q in qlo:qhi, kv in klo:khi
                k = keep(q + input_pos, kv)::Bool
                anyk |= k; allk &= k
            end
            if anyk
                slot += 1
                idx[slot, i]  = j - 1                       # 0-based block id
                # boundary blocks are never "full": full skips the kernel's
                # kv-length mask along with mask_mod.
                full[slot, i] = (allk && i*TILE_M <= n_q && j*TILE_N <= n_kv) ? 1 : 0
            end
        end
        cnt[i] = slot
    end
    BlockMask(CuArray(cnt), CuArray(idx), CuArray(full))
end
