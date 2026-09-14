# Streaming forward attention

A complete single-head BF16 attention kernel for head dimension 64. Q/K/V use
BF16 storage; row statistics and the weighted output accumulate in FP32.
Sequence lengths are runtime integers. Masks, causal masking, query/key tails,
empty key sets, and fully masked rows are supported. Inputs must keep dot products and weighted sums in FP32 range; valid scores
must be finite.

Each CTA owns 64 query rows. Its four warps own 16 rows each and all columns
of those rows. The kernel holds Q in shared memory, then streams 32 keys/values
at a time. Its fixed shared allocation is 16 KiB; there is no global score or
probability workspace. A supplied arbitrary Boolean mask still occupies M×N
bytes—this example does not make the mask implicit or compress it.

The complete iteration is visible in [kernel.jl](kernel.jl):

```julia
scores = mma(config.scores, sq, sk, zero_accumulator(config.scores), tid)
update = softmax_update(state, mask_scores(scores, mask, tid, row, key, m, n, causal))
out = weighted_values(config.output, update.weights, sv,
                      out .* update.rescale, tid)
state = update.state
```

`weighted_values` converts adjacent FP32 score-result atoms into BF16 A operands
with `pack_operand_a`, then multiplies by V. That mapping requires no shuffles
or shared staging. Both accumulators use the same row ownership despite their
different widths (32 score columns versus 64 output columns). Copies, waits,
CTA barriers and buffer reuse remain explicit in this example.

Physical Julia arrays are Q `(64,M)`, K `(64,N)`, V `(N,64)`, mask `(N,M)`, and
output `(64,M)`. V may have a padded leading dimension ≥N; padding is outside
the logical key range. All arrays must be contiguous column-major arrays.
The wrapper checks these contracts before launching. Callers capturing a graph
must retain its input/output arrays for the full replay lifetime:

```julia
include("examples/streaming_attention/kernel.jl")
StreamingAttention.launch!(output, q, k, v, mask; causal=false)
```

An entirely masked row produces zero. With `causal=true`, key index j is valid
only if j≤query index i (top-left alignment for rectangular shapes). The
schedule still visits all key tiles; it does not skip future causal tiles.

## Rounding

The running denominator sums FP32 exponentials. The PV numerator uses those
unnormalized weights rounded to BF16, separately for each 32-key tile. After
streaming, divide the numerator by the final denominator. This is different
from rounding already normalized full-row probabilities to BF16.

[reference.jl](reference.jl) contains an independent Float64 attention reference
and a diagnostic reference with the same per-tile BF16 boundary. Tests use
absolute error bounds as well as relative criteria, including cancellation
near zero. No bitwise equivalence to materialized softmax is claimed.

## Run and measure

From Tylo's root with Julia 1.12 and compatible PTX checked out beside Tylo:

```sh
julia --project=examples/streaming_attention -e 'using Pkg; Pkg.instantiate()'
julia --project=examples/streaming_attention examples/streaming_attention/run.jl /tmp/tylo-attention-results
```

Use a fresh output directory. The dated validation report records the exact
PTX revision used for the GB10 results. The GPU test suite includes this kernel;
`test/tools/sanitize.jl` includes its copy/compute/reuse and replay workload.

The benchmark compares against materialized cuBLAS QK and PV with a scalar warp
softmax between them. It fixes `CUBLAS_COMPUTE_32F` and `DEFAULT_MATH`, BF16 inputs,
FP32 GEMM outputs, and a BF16 normalized-probability buffer. Scalar device
references are prepared before graph capture and retained throughout replay.
Both paths use the same inputs, effective mask, scale, and FP32 output.
It is a comparison with this baseline, not a tuned FlashAttention library.

Results retain all interleaved timing samples, numerical errors, registers,
local memory, and explicit workspace bytes. Compilation/first execution,
preparation and transfers are outside warm graph timings. The baseline's
explicit score/probability storage is 6MN bytes; library-internal workspace
and common inputs/output/mask are excluded from that number.

This first schedule prioritizes a visible dataflow. Register pressure, sparse
CTA grids for short queries, masked work, compact ragged V copies, and the
single-stage pipeline can limit performance. It does not implement backward,
dropout, multiple heads per launch, other head dimensions, or decode scheduling.
