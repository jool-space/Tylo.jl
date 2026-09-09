# Validation

Validation checkpoint: 2026-09-09. The package remains experimental.

Environment: Julia 1.12.7, CUDACore 6.3.1, CUDA compiler 13.3.73,
CUDA runtime 13.3.0, NVIDIA GB10 (compute capability 12.1).
The 1,423 host checks also pass on Julia 1.11.9. Julia 1.10 is configured in
CI but was not executed locally. There are no detected method ambiguities.

The final GPU run used an isolated PTX checkout at
`32e36c122bc1c7af5f171cf478324b628b06af3a`, matching the assembly CI pin.

## Executed checks

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

## Complete GEMM

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

## Attention regression

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

## Hopper path and Megakernels integration

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

## Hardware work remaining

TMEM round-trip and full attention execution are explicitly skipped on
GB10, which does not execute these datacenter Blackwell instructions.
B200/B300 correctness, sanitizer and paired timing runs remain prepared.
No datacenter Blackwell attention runtime or timing claim is made.

## Reproduce

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
