# Completed: streaming row state and chained warp MMA

Completed 2026-09-10 in implementation commit `60dda11`, with validation and raw
evidence linked from [the checkpoint](../docs/src/validation.md). All required
milestones passed. The spill candidate was rejected after the paired experiment;
the optional split-attention migration was assessed and left unchanged. The
original scoped plan follows.

# Next batch: streaming row state and chained warp MMA

Scoped 2026-09-10. This is a proposed execution plan; implementation has not
started. Baselines are Tylo `f0315403edb2e44c7598aa0a6aff8b4d12ec3397` and
Megakernels `121875b24a78fc964b2b29c2c37a83dd4e42b5b4`.
The [completed batch](plans/2026-09-10-rows-boundaries.md) and
[validation checkpoint](docs/src/validation.md) remain the starting evidence.

## Outcome and scope

Build a readable forward attention example that executes on GB10 and keeps its
score/probability tiles off global memory. Use it to establish two concrete
primitives: stable row statistics across tiles, and a checked conversion from
warp MMA accumulators to a later MMA operand. Keep the dataflow visible in the
kernel. This batch needs no Hopper or datacenter Blackwell rental.

Start with BF16 Q/K/V, head dimension 64, FP32 statistics/output accumulation,
and one independent head per launch. Support runtime query/key lengths,
rectangular tails, finite valid inputs, and a Boolean validity mask. Add causal
masking only after the dense/masked path passes. The first schedule owns one
query tile per CTA and streams key/value tiles using warp MMA and explicit copy
waits/barriers. A small fixed tile choice is sufficient; sequence lengths must
not become type parameters or grow register tuples.

A complete, measured kernel is the main deliverable. General attention APIs,
backward kernels, dropout, paged KV caches, GQA, mixed head sizes, automatic
pipelines, and additional instruction families are outside this batch.

## 1. Bound the projection spill investigation

The current two-copy-warp persistent projection uses 80 registers, a 104-byte
stack frame, and 144/324 bytes of spill stores/loads according to ptxas. It is
already faster than the historical raw implementation. Fewer spills alone are
not an acceptance criterion for replacing it.

Use the newly committed projection as the before baseline. Keep the older
`7770d93` comparison available, but do not label that older implementation as
this batch's baseline. Attribute the spill instructions to producer, consumer,
or shared control flow before changing the code; register count alone does not
identify the cause. Examine at most two focused candidate families, such as
shorter producer address/value live ranges and moving invariant predicates out
of repeated copy work. Preserve 64-bit global offset safety and logical bounds.

Compare persistent and separate execution, both copy-warp counts and stage
counts, full/partial row tiles, and GEMM/gate-up operations. Retain task waits,
ready/release barriers, immutable-weight prefetch, and timeout drain behavior.
A candidate must pass raw-byte comparisons, representative sanitizers, and
interleaved paired timings against the committed Tylo-backed implementation.
Report register/stack/spill differences, resource residency and latency together.
If neither candidate wins convincingly, retain the current path and document
what was learned. This investigation must not delay the attention milestones.

## 2. Stable online row state

Design a small FP32 state around the existing row-ownership vocabulary. It
carries a row maximum and an unnormalized exponential sum. Updating it with a
score tile must expose the factor needed to rescale an existing weighted output
accumulator, as well as the new tile's unnormalized exponential weights.
Choose public names after the two consumers below establish the interface.

For a nonempty update, the mathematical contract is:

```text
m_new = max(m_old, row_max(scores))
alpha = exp(m_old - m_new)
p     = exp(scores - m_new), with masked entries set to zero
l_new = alpha * l_old + row_sum(p)
o_new = alpha * o_old + p * V
output = o / l
```

The empty state is `(m=-Inf, l=0)`. Define the empty-state and empty-chunk
branches explicitly: never evaluate `-Inf - -Inf` or divide by zero. Fully
masked rows produce zero output; log-sum-exp, if exposed, is `-Inf` there.
Valid scores are finite. The weighted numerator and final normalization must
be distinguished from already normalized probabilities.

State and rescale factors preserve row identity and replication. A row result
may apply to a differently wide accumulator only when its row ownership agrees.
Retain the one-N-warp restriction; no implicit shared-memory collective or
arbitrary-layout reduction compiler is required. Keep local arithmetic separate
from shuffles and caller-owned synchronization.

Acceptance:

- Independent Float64 references cover uneven chunk boundaries, changing maxima,
  cancellation in weighted sums, large finite score shifts, and empty chunks
  before/between/after valid chunks.
- Chunking and merge-order checks use justified tolerances rather than assuming
  floating-point associativity or byte identity.
- Exercise lane-local, warp-striped and MMA row ownership only through mappings
  already supported by Tylo; verify rescaling across differing score/output
  widths with the same logical rows.
- A standalone wide-row softmax uses fixed-capacity fragments: first stream to
  compute final statistics, then reread inputs to normalize and store. It must
  not claim to emit final probabilities before the complete denominator exists.
- Compare the two-pass kernel with the existing softmax implementations across
  small and wide rows. Report the extra pass and show bounded register use as
  logical row width grows. Preserve masks and BF16/FP16 storage conversions.

## 3. A checked accumulator-to-A conversion

For the existing m16n8k16 ownership, two horizontally adjacent 16x8 C atoms
supply a 16x16 A operand in the same lanes. A planning check compared all 256
lane/value coordinates: A slot `e` corresponds to C slot `e % 4` in tile
`e ÷ 4`, offset by eight columns for the second tile. This is a host coordinate
check, not yet a GPU conversion implementation or validation.

Implement the narrow typed conversion: FP32 values are rounded to BF16/FP16
and packed into the instruction's four A register words, with logical
coordinates preserved. It is numerical conversion and register packing, not a
bit reinterpretation of FP32 values. Start at the existing atom/fragment level;
add a plan-level overload only when the worked kernel needs it. Do not infer
register counts from tile area or let a metadata reshape imply redistribution.

Acceptance:

- An independent coordinate oracle checks every source/destination value,
  repeated M atoms, adjacent N atoms, and rejection of incompatible ownership.
- GPU round-trip probes compare packed words with the established shared-memory
  store/load route, including rounding ties, signed zero, finite extremes, and
  asymmetric per-lane values. Spell out the non-finite conversion contract.
- A chained MMA computes a first product, converts its result, and consumes it
  in a second product; compare with a reference that rounds at the same boundary.
- Representative generated code has the required conversions/packing, without
  unexpected lane shuffles, local-memory materialization or shared staging for
  this same-lane mapping. A mapping that needs communication must be rejected
  or use an explicitly implemented communication path.

This is the next concrete layout lesson: identify when a conversion is local,
prove the coordinate correspondence, and make any movement explicit. A generic
register redistribution engine or separate Laythe package is not a prerequisite.

## 4. A complete streaming attention consumer

Compose QK transpose, online softmax, output rescaling, the checked probability
conversion, and PV into a forward kernel in a new example directory. Preserve
the existing datacenter Blackwell attention example and its reference digest.
Use the current explicit storage views, bounded copies and caller-owned stage
protocol; begin with a simple schedule before attempting overlap optimization.

Keep row ownership compatible between score and output accumulators so the
online rescale factor applies without cross-warp communication. Store only the
final output to global memory; the score/probability workspace must not scale
as query length times key length. Query/key tails still participate in full
warp collectives and contribute masked values. A real validity mask should
exercise fully masked rows even when dense causal attention would not.

BF16 probability conversion before PV is an explicit numerical boundary.
Statistics remain FP32, while the numerator uses the converted weights. Use
both an independent high-precision attention reference and a diagnostic
reference with the same conversion boundary. Do not silently compare differently
rounded algorithms as if they were byte-identical. Use absolute error criteria
for cancellation near zero as well as relative/scale-aware checks elsewhere.

Acceptance:

- Execute multiple key-tile iterations, query/key tails, nonzero tile origins,
  changed-input graph replay, fully masked rows, and extreme finite inputs.
- Validate lengths below a tile and several uneven larger lengths. Keep head
  dimension 64 and BF16 inputs fixed for this batch's required attention path.
- Run memcheck, racecheck and synccheck on the entire copy/compute/reuse protocol.
- Compare steady-state kernel time and allocated workspace with a materialized
  QK/softmax/PV baseline using the same inputs, masks, scaling and documented
  rounding. Use cuBLAS for baseline matrix products where available; keep
  precision settings explicit. This is not a claim against a tuned FA library.
- Save PTX, cubins and ptxas/SASS resource evidence. Separate compilation,
  preparation/packing and transfers from warm kernel timings. Report regressions
  and resource limits alongside improvements.

Do not retune the Megakernels scheduler or replace decoder attention just to
install the new standalone kernel. Prefill tiles and single-query decode have
different reuse and parallelism constraints.

## 5. Optional second consumer: split-attention statistics

After the standalone kernel passes, assess `Megakernels.AttentionMerge` as a
consumer of the same stable summary/merge arithmetic. Its existing contract is
one maximum, exponential sum, and unnormalized weighted numerator per KV split.
It already demonstrates why the empty split and rescale rules matter.

Migrate only if the new helper expresses that arithmetic without widening its
ownership model or changing the task graph. Preserve scratch ownership, task
dependencies, final normalization, and empty-split behavior. Run the decoder
reference/replay suite, paired before/after timing, and the full sanitizer
workload. No decode speedup is assumed. If it needs a distinct communication
abstraction, document the gap and leave this consumer unchanged.

## Reference study and evidence

Read the specific operations that motivate this batch, not entire frameworks:

- CUTLASS `examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py`:
  online row state, output rescaling, and its accumulator-to-A layout conversion.
  Inspected checkout: `147295a3d4b75f3aeff247c25b8927cea9a7006a`.
- CUTLASS `examples/41_fused_multi_head_attention/kernel_forward.h`:
  `iterative_softmax`, empty-state handling and weighted-output rescaling.
- ThunderKittens `kernels/attention/mha_h100/mha_h100.cu`:
  row max/sum, probability conversion and the division of responsibility between
  register primitives and the surrounding pipeline. Its WGMMA schedule is a
  reference to study, not a schedule validated for GB10.
  Inspected checkout: `be0e7e57e90858dfa2bbeab7296ff252755f8a37`.
- Existing Tylo MMA/softmax and Megakernels `DecodeAttention`/`AttentionMerge`:
  the local ownership and numerical contracts that must remain coherent.

Keep the isolated PTX revision
`32e36c122bc1c7af5f171cf478324b628b06af3a` unless a concrete missing capability
justifies a separately validated update. Do not use the moving PTX working tree
as an accidental dependency. Record hashes of any modified reference source.

Finish focused checks while iterating, then run the relevant complete suites,
documentation and final sanitizers once code settles. Keep the old aligned GEMM,
TMA, Hopper assembly and datacenter Blackwell attention comparisons as regression
checks. H100/H200 WGMMA runtime and B200/B300 TMEM/attention runtime remain pending;
GB10 results must not relabel those as validated.

Produce a dated report with exact source/dependency revisions, raw paired timing
samples, numerical/error criteria, resource evidence and commands. Include a
short explanation of what row state and the operand conversion made reusable,
and which communication/lifetime choices remain in the kernel. Commit completed
milestones with their evidence before opening another batch.
