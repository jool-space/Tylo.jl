# Validation history

These are dated measurements and test receipts, not a description of every
current API. Read [Current status and validation](validation.md) first.
Compiler versions, kernel revisions and benchmark contracts belong to each
entry; counts and performance figures from different entries are not directly
comparable.

## 2026-09-10: logical fragments and TMEM transfer partitions

Fragments now retain explicit thread/value ownership through Julia broadcast,
reductions, slicing, BF16 packing and logical axis permutation. TMEM storage
uses `TmemTile(T, address, layout)` with explicit transfer partitions. See
[TMEM tiles and transfers](@ref) for the storage and participation contracts.

Validation uses Julia 1.13.0, pinned PTX
`32e36c122bc1c7af5f171cf478324b628b06af3a`, CUDACore 6.3.1, CUDA compiler
13.3.73 and a GB10. Host contracts also pass on Julia 1.11.9.

- Host suites and documentation/doctests pass.
- The full GPU suite passes 3,176 checks, with three expected architecture
  skips: WGMMA execution, TMEM execution, and datacenter FlashAttention execution.
- New GB10 tests exercise runtime address offsets, flat/hierarchical layouts,
  both logical axis orders, reductions, register windows and packed global
  stores against independent references. They execute no TMEM instructions.
- Actual TMEM round trips assemble for SM100a in both axis orders; runtime
  tests for both orders are prepared for B200/B300.
- All six complete attention variants still have byte-identical executable
  kernel sections to the pinned raw PTX reference. This includes its existing
  spills; the comparison excludes debug/source metadata.

This is correctness and code-generation evidence, not a performance or
B200/B300 runtime claim. Local logs, before/after resource reports and source
hashes are in the ignored `reports/tmem-api-2026-09-10/` directory. The earlier
fragment/broadcast batch has its own `reports/fragment-api-2026-09-10/` receipt.

## 2026-09-10: streaming row state and complete GB10 attention

Implementation: `60dda118bfc6d65a3b5b0b72fcba8c3d728eb68f`. This batch adds
`SoftmaxState`, stable tile updates/summary merges, a checked same-lane
accumulator-to-A conversion, fixed-capacity two-pass softmax, and complete
single-head BF16 forward attention with D=64. The dataflow and numerical
contract are described in [Streaming attention and online state](@ref).

The isolated PTX dependency remains
`32e36c122bc1c7af5f171cf478324b628b06af3a`; all 543 snapshot files were checked
against that commit. GB10 uses Julia 1.12.7, CUDACore/cuBLAS 6.3.1, CUDA
compiler 13.3.73 and runtime 13.3.0. Host checks also pass on Julia 1.11.9.

| Check | Result |
|:--|:--|
| Host contracts, coordinates and arithmetic | 30,235 passed on each Julia version |
| Host loading without CUDA; method ambiguities | Passed |
| Full Tylo GPU suite | 2,985 passed; three expected hardware skips |
| Offline / supported-runtime checks within that suite | 632 / 2,353 |
| New assembly checks with CUDA devices hidden | 20 passed |
| Megakernels complete suite | 238,348 passed; one expected Hopper skip |
| Both packages: memcheck / racecheck / synccheck | All six passed, zero errors and race warnings |
| Hopper runner tests | Nine passed using fixtures; no Hopper hardware claim |
| Documentation and isolated example environment | Built; instantiated/loaded successfully |

Runtime coverage includes empty chunks and key sets, changing maxima, weighted
cancellation, BF16/FP16 rounding ties and subnormals, signed zero/non-finite
conversion contracts, chained MMA, multiple key iterations, nonzero query tile
origins, masks, causal tails, compact/padded V and changed-input graph replay.
The 27,652 accumulator-conversion coordinate checks use an independent oracle.

### Attention performance and workspace

All timings below are microseconds: medians of 41 interleaved samples, each
containing eight warmed graph repetitions. Inputs, effective masks and scale
are identical within each pair. Compilation, preparation and transfers are
excluded. Clocks were not fixed. These compare the worked kernel against a
materialized cuBLAS QK / scalar warp softmax / cuBLAS PV baseline, not a tuned
FlashAttention library.

| Queries×keys | Mask/storage | Streaming | Materialized cuBLAS |
|:--|:--|--:|--:|
| 64×64 | masked | 22.15 | 11.98 |
| 129×257 | masked | 105.85 | 34.46 |
| 256×256 | masked | 61.20 | 23.05 |
| 1024×1024 | masked | 186.39 | 120.64 |
| 2048×2048 | masked | 355.90 | 429.65 |
| 1024×1024 | causal | 206.46 | 105.48 |
| 129×257 | masked, padded V | 76.25 | 34.38 |

The 2048×2048 case is about 17% faster in this run and avoids the baseline's
24 MiB of explicit score/probability buffers. Small/medium cases remain slower.
The arbitrary Boolean mask is common input and still occupies M×N bytes;
input/output storage and cuBLAS internal workspace are excluded from the
workspace comparison. The streaming kernel uses 16 KiB shared memory per CTA
and no explicit global score/probability workspace.

Runtime compilation reports 193 registers and zero local bytes. Offline SM121a
assembly reports 191 registers (masked) / 190 (causal), with zero stack frame
and zero spill loads/stores. These are different compiler configurations;
resource counters are kept with their corresponding artifacts. Runtime occupancy
queries allow two 128-thread CTAs per SM, or 16.7% theoretical warp occupancy.
GB10 has 48 SMs, while the benchmark grids contain only 1–32 query CTAs. This
schedule consequently leaves substantial parallelism unused. The causal kernel
also visits future key tiles, and compact ragged V often requires scalar copies.

FP32 statistics use unrounded exponential weights; the numerator consumes BF16
weights rounded per 32-key tile. The baseline rounds normalized probabilities
to BF16. Both use FP32 matrix-product outputs; cuBLAS uses explicit
`CUBLAS_COMPUTE_32F` and `DEFAULT_MATH`. The largest measured absolute error
against the Float64 reference was 0.00244 for streaming and 0.00381 for the
baseline (causal case). Runtime tests additionally use an independent reference
with the per-tile BF16 boundary. Cancellation is assessed with absolute bounds.
The baseline's device scalars are allocated before graph capture and retained
across replay, as are all input/output/workspace buffers.

### Fixed-capacity wide-row softmax

These are medians of 41 interleaved samples of 16-kernel graphs, on 1,024 FP32
rows with identical masks. All sequence widths use the same four-value-per-lane
streaming specialization.

| Width | Scalar three-pass | Full-row registers | Streaming two-pass | Registers: full / streaming |
|--:|--:|--:|--:|--:|
| 31 | 4.97 | 3.56 | 4.33 | 18 / 29 |
| 97 | 8.79 | 6.35 | 6.36 | 27 / 29 |
| 257 | 18.03 | 12.12 | 15.97 | 40 / 29 |
| 1024 | 81.25 | 63.05 | 54.15 | 96 / 29 |
| 4099 | 581.71 | 507.93 | 387.29 | 255 / 29 |

The full-row implementation is preferable for small rows. Streaming becomes
useful as register capacity limits the full-row approach. At width 4099, the
runtime full-row kernel reports 255 registers and 328 local bytes; the separately
emitted kernel has a 336-byte stack frame and 388/332 spill-store/load bytes.
Streaming remains at 29 registers and zero local/stack/spill storage on GB10.
Its extra read and per-chunk reductions are real costs, not hidden work.

### Regressions and the layer boundary

All 16 aligned GEMM assemblies retain 80 registers and zero stack/spill storage.
The existing TMA and Hopper assembly checks pass. All six paired datacenter
Blackwell attention comparisons retain identical executable bytes. H100/H200
WGMMA execution and B200/B300 TMEM/attention execution remain pending; GB10
results do not validate those paths.

The bounded Megakernels spill experiment moved fast-path eligibility to host
preparation. It removed spills in two-copy-warp cases but introduced them in a
small one-copy-warp case, with only modest persistent timing gains. It was
reverted; the candidate and paired data are preserved in Megakernels' report
commit `650716a`. `AttentionMerge` remains unchanged: its two-pass scalar factor
calculation is a different consumer than incremental weighted-output updates.

The reusable parts are row identity/statistics and the proven local register
conversion. Allocation, copy staging, collective participation and lifetime
remain explicit in the consumer. Next performance work should test smaller
query CTAs and copy overlap, with the same references and paired measurements,
before adding another general scheduling or layout abstraction.

Compact raw samples, resource tables, source hashes and commands are in
[the evidence directory](https://github.com/jool-space/Tylo.jl/tree/main/docs/validation/streaming-2026-09-10).
The local `reports/streaming-attention-2026-09-10/` archive additionally contains
full logs, exact source/environment snapshots, PTX/cubins/SASS and an artifact
SHA256 index. It stays outside Git; its receipt is included in the compact report.

## 2026-09-10: rows, boundaries, and two Megakernels consumers

The GB10 batch adds FP32 row sums/maxima and broadcasts for lane-local,
warp-striped, and tiled MMA ownership; masked standalone softmax and a softmax
epilogue over actual MMA output; bounded copies/stores and ragged BF16/FP16 GEMM.
Megakernels now uses the same primitives for normalization and `AsyncProjection`.
Its former private tile module has been removed.

The toolchain remains Julia 1.12.7, CUDACore 6.3.1, CUDA compiler 13.3.73,
and PTX revision `32e36c122bc1c7af5f171cf478324b628b06af3a` in an isolated
checkout. The active PTX development tree was not used.

| Check | Result |
|:--|:--|
| Host contracts, coordinates and arithmetic | 1,699 passed on Julia 1.11.9 and 1.12.7 |
| Host loading without CUDA and method ambiguities | Passed |
| Complete offline GPU suite, including attention | 612 passed |
| Complete supported GB10 runtime suite | 544 passed |
| Megakernels complete suite | 238,348 passed; one expected Hopper runtime skip |
| Both packages: memcheck / racecheck / synccheck | All six passed, zero errors; no racecheck warnings |

The runtime coverage includes independent row-coordinate references, cancellation,
masks and fully masked rows, irregular widths, negative copy origins, nonzero
swizzle windows, both copy axes, strided sources, output guards, K tails, and
stage reuse. The normalization integration preserves the distinct residual
rounding contracts and covers changed-input replay under four execution modes.
The smaller Megakernels assertion count follows removal of redundant private
layout checks; Tylo owns those primitive contracts now.

Six row probes, two softmax probes and the two checked ragged GEMM variants
have no stack/spill storage. The 16 aligned GEMM variants retain 80 registers
and zero stack/spill bytes. Their executable bytes differ from the saved
pre-batch binaries (register allocation changed), so no whole-batch binary
identity claim is made for GEMM. All six datacenter Blackwell attention
comparisons remain byte-identical to their paired reference.

The warmed standalone softmax measurements use a simple scalar warp baseline,
identical FP32 storage and masks, and 1,024 rows. Tylo's warp version measured
3.80 / 6.55 / 12.27 µs at widths 31 / 97 / 257, versus 5.03 / 8.98 / 18.12 µs
for that baseline. These are not cuDNN or FlashAttention comparisons. The
lane-local version was much slower with this storage; at width 257 it used
255 registers and 456 local bytes. Ownership remains a performance decision.

Megakernels normalization stayed effectively unchanged in the paired timings.
The migrated persistent projections measured 11–18% lower latency in the
three medium/larger cases, with about 1% lower latency in the smallest case.
The two-copy-warp configuration introduces spills (144 store / 324 load bytes
reported by ptxas) despite its lower measured latency; this is a concrete
follow-up tuning target, not a universally free abstraction.

The dated local receipt is `reports/rows-boundaries-2026-09-10/`. It contains
source hashes and snapshots, dependency lockfiles, complete logs, generated
code/resources, paired benchmark samples and the precise baseline definitions.
Boundary timing and alignment costs are reported separately there. Kernel
measurements exclude compilation, allocation, packing/padding and transfers;
clocks were not locked. Use paired results within a run.

Hopper WGMMA and datacenter Blackwell TMEM/attention execution remain pending
H100/H200 and B200/B300, respectively. Current Megakernels warp projections
also need H100 runtime revalidation. GB10 execution does not validate them.

## 2026-09-09 checkpoint (historical)

Validation checkpoint: 2026-09-09. The package remains experimental.

Environment: Julia 1.12.7, CUDACore 6.3.1, CUDA compiler 13.3.73,
CUDA runtime 13.3.0, NVIDIA GB10 (compute capability 12.1).
The 1,423 host checks also pass on Julia 1.11.9. Julia 1.10 is configured in
CI but was not executed locally. There are no detected method ambiguities.

The final GPU run used an isolated PTX checkout at
`32e36c122bc1c7af5f171cf478324b628b06af3a`, matching the assembly CI pin.

### Executed checks

| Check | Result |
|---|---|
| Host layouts, ownership, fragments, address units and copy contracts | 1,423 passed |
| Fragment/TMEM probes through PTX and ptxas | 19 passed |
| Complete GEMM assembly: two dtypes × two layouts × four targets | 160 passed |
| GB10 shared operand loads with nonzero window origins | 8 passed |
| GB10 GEMM correctness and output guards: 41 configurations/shapes | 82 passed |
| GB10 register arithmetic, BF16 conversion and global stores | 3 passed |
| GB10 TMA: both axes, OOB, reuse, GC, cross-stream dependencies | 84 passed |
| TMA/WGMMA pipelines and descriptor-origin assembly | 292 passed |
| Full attention assembly/comparison | 108 passed |
| GEMM/TMA Compute Sanitizer memcheck | 0 errors |
| GEMM/TMA Compute Sanitizer racecheck | 0 errors, 0 warnings |
| GEMM/TMA Compute Sanitizer synccheck | 0 errors |
| Documentation and doctests | Passed |

### Complete GEMM

The GEMM suite executes BF16 and FP16 inputs, plain/swizzled shared layouts,
K tiles of 16/32/64, one/two copy stages, different warp arrangements,
padded runtime leading strides, rectangular grids and a fused scaling/ReLU
epilogue. It checks one iteration, pipeline fill/drain and repeated reuse.
A small tile also exercises copy groups with inactive producers.

A separate test reads operand register words back and checks their logical
coordinates independently of the ownership implementation, including
windows whose origins have a nonzero swizzle phase. Host checks exercise
64-bit global offset arithmetic without allocating multi-gigabyte buffers.

The 16 checked 64×64×32 assembly variants all use **80 registers, zero stack
bytes and zero spill bytes**. Generated entry PTX contains no device calls
or local arrays. Targets are SM80, SM90a, SM100a and SM121a; only GB10 runtime
is claimed. These resource counts apply to the checked configuration, not
all possible tilings or future compiler versions.

The runnable demo reports CUDA-graph median timings for 512×512×512 GEMM.
It excludes compilation, allocation and host input preparation. The example
is not benchmarked against cuBLAS and makes no peak-throughput claim.

### Attention regression

The full attention comparison covers SM100a, SM103a, and SM100f with both
half- and quarter-granular probability publication. **All six Tylo kernels
have byte-identical executable kernel sections to their reference.** This
comparison excludes debug/source metadata and includes the generated spill
instructions. It is now an assertion in the test suite.

Standalone correction, epilogue, fragment, and TMEM round-trip probes have
zero stack/spill bytes. The full attention kernel already has spills under
this compiler; Tylo preserves those resource counts exactly:

| Target | Publication | Registers | Stack bytes | Spill stores / loads (bytes) |
|---|---|---:|---:|---:|
| SM100a / SM100f | Half | 128 | 80 | 380 / 388 |
| SM100a / SM100f | Quarter | 128 | 88 | 60 / 72 |
| SM103a | Half | 128 | 128 | 424 / 444 |
| SM103a | Quarter | 128 | 136 | 116 / 140 |

### Hopper path and Megakernels integration

The 48 standalone Hopper kernels cover BF16/FP16, N=8/16/24/64/128/256,
one/two stages, independent K partials, K=16/32/64 descriptor extents,
nonzero descriptor origins and ordinary shared stores followed by an async
proxy fence. All assemble with **zero stack/spill bytes** and no local-memory
loads/stores. The n8 four-partial pipeline uses 42 registers; the n256 pipeline
uses 154. These are compiler resource counts, not performance measurements.

Megakernels' `HopperProjection` now uses these Tylo primitives for GEMM and
fused gate/up projection. Its persistent kernels use 72 registers, no spills
and a 32-byte stack frame, matching the earlier raw-PTX prototype's resource
counts. Ptxas retains the earlier serialization diagnostic associated with
exception helper calls. The caller still waits each WGMMA group before
releasing its shared stage; no compute-overlap performance claim is made.

Megakernels' full local suite passed 284,001 checks. Both stage counts,
profiling modes and all three executors assemble for SM90a. The standalone
Hopper runtime suite additionally checks changed-input graph replay, K tails,
GC retention and exact arithmetic from nonzero shared descriptor origins.
**WGMMA execution and performance remain unvalidated on H100/H200.**

### Hardware work remaining

TMEM round-trip and full attention execution are explicitly skipped on
GB10, which does not execute these datacenter Blackwell instructions.
B200/B300 correctness, sanitizer and paired timing runs remain prepared.
No datacenter Blackwell attention runtime or timing claim is made.

### Reproduce

From the repository root with a sibling PTX checkout:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=test/gpu -e 'using Pkg; Pkg.instantiate()'
TYLO_EVIDENCE=/tmp/tylo-evidence \
  julia --project=test/gpu test/gpu/runtests.jl --attention
julia --project=test/gpu test/gpu/resources.jl /tmp/tylo-evidence
julia --project=test/gpu examples/gemm/run.jl
```

Run each sanitizer with a version matching the selected CUDA toolchain:

```sh
compute-sanitizer --tool memcheck --error-exitcode 86 \
  julia --project=test/gpu test/gpu/sanitize.jl
compute-sanitizer --tool racecheck --error-exitcode 86 \
  julia --project=test/gpu test/gpu/sanitize.jl
compute-sanitizer --tool synccheck --error-exitcode 86 \
  julia --project=test/gpu test/gpu/sanitize.jl
```

The attention source digest and paired benchmark instructions live in
`examples/flash_attention/README.md`. CI checks out the pinned PTX revision
and saves the generated PTX/cubins. The new CI changes have not run remotely.
Local logs, code and a source-hash receipt are saved under the ignored
`reports/tma-wgmma-2026-09-09/` directory.
