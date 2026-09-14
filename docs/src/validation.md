# Current status and validation

This page describes the current working tree as of 2026-09-13. The detailed
[validation history](validation-history.md) records older revisions, compiler
versions, numerical contracts and measurements. Historical sanitizer or timing
results do not automatically validate later changes.

## Capability and hardware coverage

| Area | Implemented scope | Evidence available | Main limit |
|:--|:--|:--|:--|
| Layout mathematics | Hierarchical affine shapes/strides, static/runtime leaves, composition, XOR swizzles, factorization, windows, two-axis permutation | Host tests; selected device arithmetic on GB10 | No general inverse/complement solver or symbolic simplifier |
| Register arithmetic | Generic `Fragment` values, scalar broadcast, conversions and axis permutation; ready warp-MMA and completed WGMMA values | Host and GB10 tests | Not a full array interface; packed and pending representations have explicit boundaries |
| Reductions/windows | Selected lane-local, warp-striped, warp-MMA, WGMMA and TMEM ownership recipes | Independent coordinate/numerical references, assembly, GB10 | Arbitrary ownership does not imply a supported collective or slice |
| Warp-MMA GEMM | BF16/FP16 `m16n8k16`, repeated atoms, full/bounded copies and stores, one/two copy stages | SM80/90a/100a/121a assembly; GB10 runtime | No claim of a tuned GEMM library; no split-K/autotuning |
| TMA | 2D loads with one canonical B128-swizzled, K=64 BF16/FP16 storage format | GB10 runtime, bounds/reuse/lifetime tests | No arbitrary layouts, stores, multicast or clusters |
| Hopper WGMMA | Shared/shared M=64, N=8:8:256, K=16/32/64, selected partial accumulators | SM90a assembly; H100/H200 tests prepared | Runtime and performance on Hopper remain unvalidated |
| TMEM | FP32 and packed BF16/FP16 `.32x32b` loads/stores, x1–x128, logical windows and transfer partitions | SM100a assembly; B200 round trips; address/register work on GB10 | No tcgen05 MMA in Tylo |
| Streaming attention | Complete single-head BF16 forward kernel, D=64, online statistics, masks and causal tails | GB10 correctness and dated paired measurements | Fixed schedule/geometry; small cases can be slower than the baseline |
| Datacenter attention experiment | TMA loads, correction and epilogue replacements in a raw PTX kernel | Six complete kernel-code comparisons; paired B200 execution and timings | Remaining kernel is the reference; softmax stream and tcgen05 MMA still raw PTX |

For precise fragment-method coverage, see [Register fragments](rows.md). The
online `SoftmaxState(f; dims)` uses reduced fragments on either implemented
logical axis. It does not synthesize collectives for arbitrary ownership.

## Most recent checks

### B200 execution (2026-09-14)

The full suite on a rented NVIDIA B200 (capability 10.0, Julia 1.13) passed
**83,368 checks** with one expected skip (Hopper WGMMA). This is the first
hardware execution of the TMEM round trips and of the datacenter attention
comparison, whose reference kernel is now vendored in
`examples/flash_attention/reference.jl` and whose Tylo variant issues its TMA
loads, correction and epilogue through Tylo:

- All eight paired execution cases (both publication granularities, including
  the rescale-heavy input and the single-CTA repeated-work-item grid) produce
  bit-identical outputs from the reference and Tylo kernels and agree with the
  CPU reference within 5e-2.
- Paired CUDA-graph timings are equal within noise (ratios 0.991–1.000 across
  the eight cases), as expected for byte-identical machine code.

These are modest correctness shapes, not a saturated throughput measurement.
The B300 (capability 10.3) has not been exercised.

### Typed packing and completed MMA values (2026-09-13)

Julia 1.10.12 and 1.13.0 each pass **78,062 host checks** and **3,562
GPU/assembly checks**, with three expected hardware skips on the GB10. The
suite uses CUDACore 6.3.1, CUDA compiler 13.3.73 and PTX revision
`32e36c122bc1c7af5f171cf478324b628b06af3a`.

This batch removes the row-specific compatibility APIs. `Fragment` carries
explicit ownership; `PackedFragment{T}` separates logical BF16/FP16 elements
from register words. It adds bit-preserving packing/unpacking, typed packed
global stores, all `.32x32b.x1` through `.x128` TMEM widths for FP32/BF16/FP16,
and ordinary fragment arithmetic on completed WGMMA values. Streaming softmax
retains an explicit logical reduction axis.

- Host tests exhaust all 65,536 16-bit payloads for both packed element types
  and both logical orientations, including NaN payloads and signed zero.
- GB10 tests run conversion, packed stores, bit round-trips and WGMMA ownership
  arithmetic. The latter uses ordinary register/shuffle instructions, not WGMMA.
- 48 typed TMEM round-trip variants assemble for SM100a. Eight complete
  WGMMA-to-softmax variants assemble for SM90a. Their numerical execution tests
  are gated to server Blackwell and Hopper respectively.
- All six complete FlashAttention machine-code comparisons still pass. Of the
  111 preceding Julia 1.13 saved kernels, 109 retain identical executable
  sections. The two TMEM round-trip probes retain the same SASS instruction
  multiset; independent global-load ordering changed. This is not a measured
  performance claim. There are 74 additional saved kernels in this batch.
- Both Megakernels examples pass, and its 26 TMA/WGMMA projection assembly
  checks pass with the new result type.

The manual builds with doctests. No new sanitizer or rented-hardware execution
is claimed. Local receipts are in the ignored
`reports/expressiveness-2026-09-13/` directory.

### Previous CI compatibility checks

The CI compatibility work was checked locally on GB10 with Julia 1.10.12 and
1.13.0, using fresh test environments, CUDACore 6.3.1, CUDA compiler 13.3.73
and the pinned PTX revision below. Each version passed 36,610 host checks and
3,176 GPU/assembly checks, with three expected hardware skips. All 107 saved
Julia 1.13 kernel binaries retained the baseline's executable sections after
the static-metadata and inlining fixes needed by Julia 1.10.

The offline suite also passed with the GPU hidden; requiring GPU execution in
that environment failed as intended. These are local checks, not a receipt of
hosted x86_64 CI execution. No new sanitizer run is claimed.

### Preceding fragment API checks

The fragment/TMEM API work used Julia 1.13.0, CUDACore 6.3.1, CUDA compiler
13.3.73, and PTX revision `32e36c122bc1c7af5f171cf478324b628b06af3a`.
Host checks also passed on Julia 1.11.9.

- Full host suite: 36,608 passed on each Julia version. Two final negative
  coordinate checks were then added; the focused TMEM suite passed 4,947
  checks on both versions.
- Full GPU suite with `--attention`: 3,176 passed and three explicit skips for
  WGMMA, TMEM and datacenter attention execution on unsupported hardware.
- All six complete datacenter attention variants retain byte-identical executable
  kernel sections to the reference. Debug/source metadata is excluded; the
  reference's existing spill code is included.
- Broadcasted scalar PTX and ordinary/custom Julia functions were also tested
  on GB10. The direct scalar PTX probes have no calls, local arrays or shuffles.
- Replacing explicit macro-call AST construction with interpolated PTX strings
  passed the affected MMA/TMEM tests and all 48 Hopper assembly cases. All 69
  compared kernels retained byte-identical executable sections. Representative
  Hopper cases also compiled with the then-current sibling PTX checkout.

These counts describe those runs, not a maintained assertion about the number
of tests. Subsequent documentation edits are checked by building the manual and
running its doctests. No new full-library sanitizer run is claimed for this API
batch. Receipts for it are in the ignored `reports/fragment-api-2026-09-10/` and
`reports/tmem-api-2026-09-10/` directories; interpolation/probe logs are local
`/tmp/tylo-interpolated-instructions.log` and `/tmp/tylo-scalar-ptx-broadcast.log`.

## What the performance evidence means

Matching machine code establishes the cost of a specific abstraction replacement
under a fixed compiler. It does not prove that the enclosing algorithm or
schedule is optimal. The complete streaming kernel has measured wins at some
larger shapes and losses at small/medium shapes against its materialized cuBLAS
baseline. That baseline is not a tuned FlashAttention implementation, and its
BF16 rounding boundary differs. Read the [dated measurements and numerical
contract](validation-history.md) before interpreting the timings.

Allocation management, general register redistribution, wider collective
coverage and a uniform cooperative-group API remain design work. Actual Hopper
and datacenter Blackwell execution is the main hardware validation gap; it does
not prevent improving and testing the warp path, layout machinery or scalar API
on GB10.

## Reproduce checks

The test project is a workspace member. One instantiate covers the package,
its tests and the manual; `Pkg.test()` and the runner are equivalent:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(; workspace=true)'
julia --project=test test/runtests.jl --jobs=4
julia --project=test test/runtests.jl host        # no CUDA compiler or device needed
```

`host/` tests run everywhere. `gpu/` files need the CUDA compiler; their
assembly checks always run, and each file's `# TEST_TARGET:` banner states
which devices execute its runtime sections, so architecture-specific kernels
that the local GPU cannot run are assembled and then skipped. CI tests Julia
`1.10` and `1` (latest stable) on the GB10 runner with
`TYLO_REQUIRE_GPU_RUNTIME=true`, so a missing GPU fails that job, and on
hosted x86_64 machines without a device. A CI configuration is not a receipt
that a working tree has run remotely.

On a machine without a CUDA driver, explicitly select compiler artifacts before
starting the tests. Without a driver or a version preference, the CUDA compiler
JLL may have no selected artifact, leaving `ptxas` unavailable:

```sh
julia --project=test -e 'using CUDACore; CUDACore.set_runtime_version!(v"13.3"; local_toolkit=false)'
julia --project=test --check-bounds=auto test/runtests.jl gpu/flash_attention
```

The preference applies to this test environment. The second command starts a
fresh Julia process, which loads the selected toolkit. Both CI jobs use normal
bounds semantics and disable GPU coverage instrumentation for the exact code
comparison; host contracts supply coverage separately.

The datacenter attention comparison compiles the reference kernel in
`examples/flash_attention/reference.jl` and its Tylo variant; the example's
README has the rental checklist:

```sh
TYLO_EVIDENCE=/tmp/tylo-evidence julia --project=test examples/flash_attention/run.jl
julia --project=test test/tools/resources.jl /tmp/tylo-evidence
```

`resources.jl` records assembler registers/stack/spills and disassembles the
saved cubins. An evidence directory alone does not indicate which cases ran;
retain the test log and exact source/compiler versions as well.

The manual can be built without a GPU. The host examples are doctests; snippets
marked as kernel excerpts require their surrounding setup and are not standalone
programs:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Open `docs/build/index.html`. For sanitizer and hardware-specific commands, use
`examples/gemm/README.md`, `examples/hopper/README.md`, and
`examples/flash_attention/README.md` with a sanitizer compatible with the selected
CUDA toolchain. The [implementation guide](codebase.md) maps individual tests to
contracts.

### Layout-driven collectives (2026-09-14)

Reductions, broadcast slot maps, register windows and the same-lane C-to-A
conversion are now derived from enumerated ownership tables at generation
time (`src/enumerate.jl`), replacing the per-ownership recipe methods. The
gate for this change is the kernel snapshot in `test/gpu/snapshots/`, taken
on the dirty working tree of 2026-09-14 with Julia 1.13.0, CUDACore 6.4.0,
CUDA compiler 13.4.59 and PTX `ae2362f`.

- Tylo: 3,565 GPU/assembly checks pass with the snapshot active; all 185
  saved kernels are byte-identical or identical up to ptxas register
  numbering. Host checks pass, including 143 new enumeration checks against
  independent ISA figures.
- Megakernels: 238,340 checks pass with one expected Hopper skip; the eight
  projection kernels from `benchmark/tiles.jl` are byte-identical and their
  outputs remain bitwise equal to `7770d93`; the one-stage Hopper kernel is
  identical and the two-stage one has the same instruction multiset, which is
  also true of two fresh compiles of unchanged code.

The derivation admits reductions that had no hand-written recipe before:
lane-axis reductions of lane-local rows, the 2×2-patch ownership and either
axis of a permuted accumulator. No sanitizer run is claimed for this batch.

### Warp MMA atoms as data (2026-09-14)

`MMAAtom{(m,n,k),TA,TB,TC}` describes a warp `mma.sync` instruction by its
shape and element types; its A, B and accumulator ownerships come from one
formula per element width, evaluated while generating. `MMAFragment` and
`MMAAccumulator` are gone: operands are `PackedFragment`s and accumulators are
flat `Fragment`s in the atom's or tiling's ownership. `load_fragment` and the
two-argument `store!` move any static ownership through scalar accesses at its
coordinates, and `test/gpu/atoms.jl` uses them as the oracle: generic loads,
the instruction, and generic stores must reproduce a host matmul, and the
`ldmatrix` loads must agree word for word with the generic loads.

Gate results, same toolchain as the previous checkpoint:

- Tylo: 3,583 GPU/assembly checks pass. Of 185 baseline kernels, 100 are
  byte-identical, the two streaming attention kernels are identical up to the
  renamed entry symbol, and 83 changed deliberately: the accumulator store is
  now the generic aligned store, replacing inline `st.global.f32` in the warp
  path and an unaligned `unsafe_store!` in the WGMMA path. Every one of the 83
  is the same size or smaller; the only added opcodes are whole-word `STG.E`
  with immediate offsets and address arithmetic, and 64 of them lost the
  byte-wise `STG.E.U8` sequences the unaligned store had produced. Examples:
  the 64×64×32 GEMM drops from 552 to 432 SASS instructions and the n64
  two-stage Hopper pipeline from 480 to 288.
- Megakernels: 238,340 checks pass; the eight projection kernels keep their
  instruction multiset and bitwise outputs against `7770d93`; the Hopper
  kernels are identical and multiset-equal.

The snapshot normalizer now ignores symbol names, so type renames do not
register as changes. No sanitizer run is claimed for this batch.

### More warp MMA atoms and the parallel test layout (2026-09-14)

Twenty-four warp atoms now have instruction bindings, each one table entry
in `src/ptx/mma.jl`: 16-bit m16n8k8 and m16n8k16 with FP32 or FP16
accumulation, TF32 m16n8k4 and m16n8k8, FP8 E4M3/E5M2 in every A/B pairing
at k16 and k32, and INT8 signed/unsigned pairings with Int32 accumulation.
`Float8E4M3` and `Float8E5M2` are Microfloats twins with the
`cvt.rn.satfinite` policy, checked exhaustively on the host; packing covers
8-bit words. The atom oracle in `test/gpu/atoms.jl` runs every atom on GB10
with inputs whose products and sums are exact in the atom's arithmetic, and
checks `ldmatrix` loads word for word against the generic loads.

The repository changed shape in the same batch: PTX is a direct dependency
with its bindings in `src/ptx/`, the test project is a workspace member, and
the suite runs in parallel from `test/host/` and `test/gpu/` with per-file
`# TEST_TARGET:` capability banners. Device-only implementations of
host-callable generics are `@device_override` methods in `ext/CUDACoreExt.jl`.

Gate results, Julia 1.13.0, CUDACore 6.4.0, CUDA compiler 13.4.59, PTX from
`main` as pinned by the workspace manifest, four parallel workers:

- Tylo: 83,276 checks pass in 3m18s with three expected hardware skips
  (Hopper, TMEM, datacenter attention). Against the 2026-09-14 snapshot, 100
  kernels are byte-identical, the streaming attention kernels are identical up
  to the renamed entry symbol, the 83 epilogue-store kernels recorded in the
  previous checkpoint remain the only allowed changes, and the 48 oracle
  kernels are new.
- Megakernels: 238,340 checks pass with one expected Hopper skip after
  re-resolving its workspace for Tylo's new dependency; the eight projection
  kernels keep their instruction multiset and bitwise outputs against
  `7770d93`, and the Hopper kernels are identical and multiset-equal.
