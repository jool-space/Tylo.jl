# Current status and validation

This page describes the current working tree as of 2026-09-24. The detailed
[validation history](validation-history.md) records older revisions, compiler
versions, numerical contracts and measurements. Historical sanitizer or timing
results do not automatically validate later changes.

## Capability and hardware coverage

| Area | Implemented scope | Evidence available | Main limit |
|:--|:--|:--|:--|
| Layout mathematics | Hierarchical affine shapes/strides, static/runtime leaves, symbolic composition, complements, right and left inverses, logical/zipped/tiled division, logical/blocked/raked products, slicing, XOR swizzles, windows, two-axis permutation; ownership plans derived from this algebra | Host tests against the enumeration oracle; selected device arithmetic on GB10; machine code of every prior kernel unchanged | Complements and inverses need static strides; windows do not compose symbolically |
| Register arithmetic and memory access | One `Fragment` type of logical values (8- and 16-bit values stored as register words), scalar broadcast, `Fragment{T}` conversion and axis permutation; loads and stores derived from ownership: `ldmatrix`/`stmatrix` blocks, then vectors within a tile's declared alignment, then scalars, through bounded tiles and tiles by index | Host and GB10 tests; Megakernels' MXFP8 body identical to a hand-written reference | Not a full array interface; bounded tiles take the vector and scalar paths only |
| Reductions/windows | Selected lane-local, warp-striped, warp-MMA, WGMMA and TMEM ownership recipes; running reductions from an initial value; cross-warp combination composed through shared memory | Independent coordinate/numerical references, assembly, GB10 | Arbitrary ownership does not imply a supported collective or slice |
| Warp MMA | 24 `mma.sync` atoms (16-bit, TF32, FP8, INT8) as ownership tables; derived `ldmatrix`/`stmatrix` operand copies; `TiledMMA` GEMM with full/bounded copies and stores, one/two copy stages | SM80/89/90a/100a/121a assembly; GB10 runtime oracle for every atom | No claim of a tuned GEMM library; no split-K/autotuning |
| TMA | 2D loads and stores of one-, two- and four-byte elements with 32-, 64- or 128-byte swizzle rows, from matrices or batches of matrices with logical bounds, into canonical storage expressed as `Swizzle{B,M,3}` over packed rows and shared with `ldmatrix`, WGMMA and tcgen05 operands | GB10 runtime for every width, batches, clipping, reuse and lifetime tests | No multicast, clusters, boxes beyond one matrix or unswizzled boxes |
| Hopper WGMMA | Shared/shared M=64, N=8:8:256, K=16/32/64, selected partial accumulators | SM90a assembly; H100/H200 tests prepared | Runtime and performance on Hopper remain unvalidated |
| TMEM and tcgen05 MMA | FP32 and packed BF16/FP16 `.32x32b` loads/stores, x1–x128, logical windows and transfer partitions; `Tcgen05Atom`s (f16/tf32/f8f6f4/i8 kinds, M=128) with shared operands recognized structurally as the 128-byte-swizzled encoding, TMEM accumulators and TMEM-sourced A, commit and TMEM allocation | SM100a assembly; B200 round trips of the transfers; MMA issue compared against the reference attention kernel on GB10 | MMA execution not yet run on B200; M=64 and cta_group::2 not described |
| Streaming attention | Complete multi-head BF16 forward kernel, D=64: TMA-fed stages, register-resident Q, online statistics, optional masks, causal tails, two warp groups alternating on the tensor pipe | GB10 correctness, dated paired measurements against a materialized cuBLAS baseline and the measured `mma.sync` peak | Fixed geometry (128 queries × 64 keys per step); single-head cases below 2048 queries underfill the 48 SMs |
| Datacenter attention experiment | TMA loads, both MMA issues, correction and epilogue replaced in a raw PTX kernel | Six kernel-code comparisons on GB10; paired B200 execution and timings of the pre-MMA revision | Softmax stream, barrier plan and roles remain raw PTX; the MMA revision awaits B200 execution |

For precise fragment-method coverage, see [Register fragments](rows.md). The
online `SoftmaxState(f; dims)` uses reduced fragments on either implemented
logical axis. It does not synthesize collectives for arbitrary ownership.

## Most recent checks

### The MXFP8 projection body, to the instruction (2026-09-24)

Megakernels' MXFP8 streamed projection, its hardest body, was rewritten in
Tylo and held to one test: it must compile to the same machine code as the
same algorithm written by hand with pointers, vector loads, shuffles and
inline PTX (`Megakernels/test/mxfp8_sass/`). Wherever the Tylo version could
not say something, or said it at a cost, the vocabulary changed:

- `bound(tile, extents)` gives a tile its valid region; loads zero what lies
  outside and stores skip it, per vector. `tile(t, shape, index)` is the
  `index`-th tile of a tiling, through hierarchical modes without carries,
  its index split with masks and shifts, its alignment narrowed to what the
  outer strides keep. Vector plans follow the tile's layout, so a 16-byte
  lane load through a permuted K order is one instruction.
- `sum`, `maximum` and `minimum` take a mapped function and an initial value,
  folding a thread's values sequentially from it (`sum(abs2, x; dims=1,
  init=s)` is one fused multiply-add per value). `Fragment(f, ownership)`
  re-owns registers, which makes lanes an axis to reduce; a store of a
  replicated fragment is issued by one replica. With a barrier and a local
  reduction of the loaded rows this composes statistics across warps.
- `relayout(to, f, thread)` selects by the thread's coordinate where each
  thread already holds what it needs; `reinterpret(T, f)` keeps the bits.
- `repeat(ownership, m, n)` tiles an ownership, so one lane's 16 bytes are
  the A registers of two consecutive K blocks.
- Paired FP32→FP8 conversions build register words directly, 8-bit words are
  joined as 16-bit halves (a pair from one `cvt` folds back to itself), a
  slot's value is extracted from its word alone, and a bound check at a zero
  offset compares the coordinate directly.

The hand-written reference and the Tylo body compile to identical SASS for
every kernel of one-operation programs (with and without normalization and
residual), the decoder (fused and unfused), a fork/join program and the full
Llama 3.2 1B program at batch 1 and 8: symbol numbering aside, the same
instructions in the same order. The original raw-PTX body (14,248
instructions in the fork/join program's persistent kernel) is replaced by
one of 11,120.

GB10, Julia 1.13: the full suite passes 72,092 checks (3 broken, as before),
with new host and device tests of tiles by index, bounds, re-owning,
selection, initial values and the bounded chunk pattern end to end. Of 363
kernels saved before and after, 244 are identical, 24 FP8 packing kernels are
smaller by 8 to 24 instructions, and 95 have the same count: 43 differ in
register assignment only and the rest in order, except the 8-bit `16x8x32`
atom oracles, where ptxas scheduled the reordered byte packing with two
uniform adds in place of two alignment NOPs.

### One vocabulary (2026-09-24)

The public surface was consolidated so that each part has one spelling.

- One distribution type: `LocalOwnership`, `StripedOwnership`,
  `TiledMMAOwnership`, `WGMMAOwnership`, `PermutedOwnership` and the
  transfer-as-ownership are gone; each is an `Ownership` built from its
  mapping. The constructor stores the mapping canonically (thread and slot
  modes coalesced), so equal distributions have equal types and transposing
  twice is the identity. `coordinate` evaluates the mapping's projections
  onto the two axes, generated per slot, with shifts and masks for power-of-two
  thread modes, and splits into a thread part and a slot constant: loads and
  stores of affine tiles form one base pointer per thread and add constant
  offsets that fold into the instructions.
- One register type: `PackedFragment`, `pack`, `unpack` and `scale` are gone.
  `f.data` is always logical values; 8- and 16-bit values are stored as the
  register words instructions consume, so loop-carried operands stay in their
  words. `Fragment{T}(f)` converts.
- One instruction description: `abstract type Atom` with `MMAAtom`,
  `ScaledMMAAtom`, `CopyAtom`, `WGMMAAtom` (was `WGMMA64`) and `Tcgen05Atom`
  (was `Tcgen05MMA`); `load_fragment(atom, role, tile, thread)` replaces
  `load_a`, `load_b`, `load_scale_a` and `load_scale_b`; `instruction_atoms()`
  includes the block-scaled atoms. The tiled `pack_operand_a` takes its plan.
- One static convention: shapes, axes, counts and alignments are plain
  constants everywhere (`TiledMMA(atom,(2,2),(2,4),32)`, `window(t,o,(16,16))`,
  `GlobalTile(p,l,16)`, `wait_copies(1)`, `CopyPlan((64,64),128,2)`,
  `TmemTransfer((32,64),2)`, `TMATile(T,(128,64),2)`, `dims=2`).
- `Layouts` is exported with its core names; `TMALoad`, `PreparedTMA`,
  `DeviceTMA` and `BroadcastFragment` are gone. Of the 83 names exported
  before, the kernel-writing vocabulary stays exported (66 names, now
  including `Atom` and the layout core) and inspection, validation and rarer
  operations are declared `public` and used qualified (17 names).

GB10, Julia 1.13: the full suite passes 71,368 checks (3 broken, as before).
Of 365 kernels saved before and after, 117 are smaller, 244 are unchanged and 4
test oracles on sm_80 grew by three address instructions; streaming attention
dropped from 1800 to 1752 instructions (2232 to 2168 with masks). Megakernels'
programs compile to identical PTX except the two Tylo-tile projections, whose
kernels shrank.

### Layout algebra and algebraic derivations (2026-09-16)

`Tylo.Layouts` gained a symbolic algebra (`src/layouts/algebra.jl`): exact
composition of layouts, coalescing, complements, right and left inverses,
logical, zipped and tiled division, logical, blocked and raked products, and
slicing. Structural operations are generated from the layout types, so
static inputs give static results and mixed inputs keep one type; inexact
cases throw instead of rounding. `tile` is now `logical_divide` by static
tiles. The ownership derivations moved from enumerated tables to this
algebra (`src/derive.jl`): every ownership has a `mapping` layout, and
reductions, broadcast slot maps, windows, in-lane relayouts, matrix copy
plans and vector plans are projections, compositions, divisions and
inverses of it. `ownership_table` remains as the oracle.

- Host: `test/host/algebra.jl` checks every operation pointwise against
  enumeration and `@inferred` for static inputs; `test/host/derive.jl`
  checks every ownership's mapping against its coordinate table, distribution
  equality against table equality over all pairs, and the derived plans
  against the previously hand-written recipes.
- GB10 (Julia 1.13.0, CUDACore 6.4, CUDA compiler 13.4): the full suite
  passes **86,313 checks** with three expected hardware skips. All 325 saved
  kernels have byte-identical PTX to the tree before this change; 321 cubins
  are byte-identical and the remaining four (two-stage Hopper WGMMA kernels)
  are identical up to ptxas register numbering.
- Docs: the layout doctests build.

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
time (then `src/enumerate.jl`, now `src/derive.jl`), replacing the per-ownership recipe methods. The
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

### Copy atoms (2026-09-14)

The `ldmatrix` and `stmatrix` instructions are now data, like the warp
MMA instructions. `CopyAtom{:load|:store,Trans}` holds the two ownerships
of one 8×8 matrix copy in `src/copyatoms.jl`: the 16-byte row each lane
addresses and the two 16-bit units each lane holds. `matrix_copy_plan` in
`src/derive.jl` covers any 32-lane ownership with those blocks along the
shared layout's static unit-stride axis, choosing the transposed instruction
when the register pattern requires it, grouping consecutive blocks into
`.x1`/`.x2`/`.x4` instructions and permuting the words into the fragment's
slot order; adjacent 8-bit elements form units, so FP8 and INT8 operands
qualify. `load_fragment` and `store!` use the plan for shared tiles and fall
back to scalar accesses otherwise; `stmatrix` is emitted only when compiling
for sm_90 or later. `load_a`/`load_b` are now that load with the operand's
ownership and serve every atom, and the tiled shared-tile `mma` accepts every
atom with a 32-bit accumulator. The old hand-written m16n8k16 loads and the
`CopyPlan` cp.async copies remain as they were.

Evidence, Julia 1.13.0, CUDACore 6.4.0, CUDA compiler 13.4.59, GB10:

- `test/host/copy.jl` checks the atom tables against the ISA figures and the
  derived plans for all 24 atoms: plain instructions for K-contiguous
  storage, transposed for MN-contiguous 16-bit storage, no plan for 8-bit
  MN-contiguous rows, FP32 values, lane-local rows or multi-warp ownerships.
- `test/gpu/atoms.jl` compiles a shared-memory round trip for every atom and
  asserts that the assembly contains exactly the planned `ldmatrix` widths and
  transpositions, `stmatrix` only at sm_90 or later, and no local memory or
  calls; at runtime every atom's operands survive registers → `stmatrix` →
  raw shared memory → `ldmatrix` → registers on GB10, swizzled and not.
- Machine code: the 16 GEMM kernels, the four operand-load kernels and the
  16-bit oracle kernels compiled before and after the change have identical
  normalized SASS; the PTX differs by one moved `and.b32` and by block labels.
- Tylo: 84,009 checks pass in 2m50s on four workers with the expected
  hardware skips. Megakernels: 238,340 checks pass in 4m47s with one expected
  Hopper skip, including its projection and Hopper kernel comparisons.

### tcgen05 MMA atoms (2026-09-14)

`Tcgen05MMA{(m,n,k),TA,TB,TC}` in `src/tcgen05.jl` describes a
`tcgen05.mma.cta_group::1` instruction by shape and element types, for the
f16, tf32, f8f6f4 and i8 kinds with M=128. Its operands are not thread-owned.
The accumulator is a `TmemTile` in the atom's TMEM layout (`accumulator`),
also accepted as A for TMEM-sourced products. Shared operands are encodings:
`b128_structure` recognizes a static layout type as the canonical
128-byte-swizzled core-matrix form (`Swizzle{3,3,3}` composed with 128-byte
rows, which `TMASharedLayout` already is) and reads the descriptor's
majorness, leading and stride offsets from its strides. `tcgen05_operand`
packs the descriptor with PTX.jl's constant fields, `mma` derives the
instruction descriptor from the operand encodings and steps K through the
tile's layout, by a constant when the K axis is one flat mode, and
`commit_mma` and the TMEM allocation verbs complete the surface.

The vendored FlashAttention kernel is the consumer. `tiles.jl` now also
replaces its QK and PV issue: the Q buffer and the KV ring are described as
stacks of 128-byte rows, both products are one atom each, and origins select
stages, slots and publish groups. The reference's own descriptor prologue was
returned to pyptx's masked form (its port had used PTX.jl's checked builder,
whose runtime checks executed in the B200 run). Evidence, GB10, CUDA
compiler 13.4.59, sm_100a/sm_103a/sm_100f:

- Both kernels issue identical TMEM loads and stores, waits, fences, commits,
  barrier arrivals and global stores, and neither has an exception path. In
  the default split-P configuration both issue 128 `tcgen05.mma`; in the
  quarter-granular fallback the tiled helper's loop unrolls completely while
  LLVM unrolls the reference's only partially (116 of 128 materialized).
- The tiled kernel's machine code is smaller: 3,032 versus 3,672 SASS
  instructions (sm_100a, split-P), because the atom's descriptor arithmetic
  keeps the descriptor's high word constant where the reference materializes
  64-bit adds per K step. The PTX instruction multisets differ only by those
  42 `add.s64`; line order differs where the epilogue's conversions and the
  phase flips were scheduled. Byte-identical machine code is therefore no
  longer the gate; `comparison.jl` asserts the instruction counts and that the
  tiled kernel never grows.
- `test/gpu/tcgen05.jl` compiles a 128×128×64 product with K-major and
  MN-major B through the atom for sm_100a with no local memory or calls, four
  MMAs, one commit and one x128 TMEM load; its execution and the attention
  cases await a B200.
- Tylo: 84,077 checks pass in 3m03s on four workers.

### Aligned tiles, vector accesses and one shared encoding (2026-09-15)

`GlobalTile` and `SharedTile` take a third argument, `Val(align)`, declaring
in bytes that the pointer and every non-unit stride are multiples of the
alignment; windows keep it only at aligned origins, and a swizzle limits the
claim to the bytes it leaves in place. Bounds checks verify the declaration
structurally and are elided with `@inbounds`. The default is the element
size, so every existing kernel compiles unchanged. `vector_plan` in
`src/derive.jl` groups a thread's slots into 16-, 8- or 4-byte vectors
that are contiguous along the tile's unit-stride axis and start aligned on
every thread; `load_fragment` and `store!` use them after the `ldmatrix`
and `stmatrix` paths, as whole words for packed element types, through
`VecElement` tuples so no new instruction wrappers were needed.

`TMASharedLayout` is gone. `shared_layout(plan)` returns the composition
`Swizzle{3,3,3}` over 128-byte rows, the encoding `b128_structure` already
recognized, so TMA storage, `ldmatrix` loads, `wgmma_operand` and
`tcgen05_operand` describe one layout type; `wgmma_operand` accepts any
K-major canonical tile through the same structural recognition.

Evidence, GB10: `test/host/memory.jl` covers the alignment declaration and
the vector plans; `test/gpu/vectors.jl` compiles and executes round trips
for FP32 accumulator pairs (`b64`), lane-local rows (`v2.b64`), BF16 and
FP8 operand words (`b32`) with one access per vector, and the scalar path
with one per element, agreeing exactly. Normalized SASS of the 16 GEMM, 4
operand-load, 48 oracle, 78 copy, 8 WGMMA, 56 Hopper, 6 TMEM, 12 attention
and 2 tcgen05 kernels is identical to the previous checkpoint. Tylo:
84,147 checks pass in 2m56s on four workers.

### TMA tiles as data (2026-09-15)

`TMATile(T, Val(shape), Val(axis), Val(width))` replaces `TMALoad` (kept as
the name of the 128-byte plan): one-, two- and four-byte elements, 32-,
64- or 128-byte swizzle rows, and stores as well as loads through one
binding. `shared_layout(plan)` is `Swizzle{B,M,3}` over packed rows, the
hardware swizzle family in elements, and `swizzled_structure` (formerly
`b128_structure`) recognizes every width, so `wgmma_operand` and
`tcgen05_operand` read the descriptor layout code, stride and alignment
from the storage instead of assuming 128 bytes. `tma_store!`,
`commit_tma_stores`, `wait_tma_reads` and `wait_tma_stores` expose the
bulk-group completion model. Two hardware rules found on GB10 and now
checked: the inner origin starts a 16-byte chunk, and store origins are
non-negative (loads zero-fill negative origins, stores reject them).

Evidence, GB10: `test/gpu/tma.jl` executes loads for four element types,
three row widths, both logical axes, three outer extents and three origins
including zero-filled out-of-bounds ones, and stores for three element
types with clipping past both far edges, comparing every element; the
store path assembles to one tensor store, one commit and both waits with
no calls. `test/host/hopper.jl` checks every plan's layout against the
byte-level hardware rule; `test/host/tcgen05.jl` checks the structures and
descriptor codes of 64- and 32-byte rows. Normalized SASS of every prior
kernel is identical to the previous checkpoint. Tylo: 85,265 checks pass in
3m30s on four workers.

### Streaming attention as a driving consumer (2026-09-15)

The streaming attention example was rewritten around the question of what
Tylo needs to ship one competitive kernel on the GB10. Each CTA now owns
128 queries of one head as two groups of four warps; Q is held in registers
as A operands; K and V tiles of 64 keys arrive by TMA through four shared
stages from rank-3 bindings (one matrix per head, padded V bounded by its
logical key count), so tails need no scalar copies; the two groups alternate
on the tensor pipe through two named barriers so one group's softmax
overlaps the other's MMAs; B operands load in pairs with one `ldmatrix.x4`;
the mask is optional and the masked path has no per-element branches; the
epilogue stores through the aligned vector path.

Two Tylo changes came out of it. `softmax_update`, `softmax_normalize` and
their element operations select instead of branching and evaluate `exp2` of
a scaled argument: a per-element branch around a full-precision `exp` cost
2,878 cycles per 64-key update in isolation, 547 afterwards. TMA bindings
take rank-3 arrays and a `bounds` keyword.

Evidence, GB10 at the 0.9 GHz the SM sustains under this load: the
`mma.sync` micro-benchmark reaches 45 TFLOPS with one warp per
sub-partition; cuBLAS BF16 GEMM reaches 43. The kernel reaches 23.3
TFLOPS on 1024 queries × 1024 keys × 16 heads unmasked (184 µs) and 26.2 on
4096² × 4 heads, against
the original schedule's 3 TFLOPS on the single-head 2048² case and the
materialized cuBLAS baseline's 3.8 on the same 16-head case. Single-head
cases below 2048 queries occupy a fraction of the 48 SMs and gain less. The
runtime test covers masks, tails, empty keys, padded V, heads and graph
replay; the assembly test pins the instruction counts of the loop. Tylo:
85,312 checks pass in 3m22s on four workers.

### FP8 elements on Microfloats' conversion funnel (2026-09-21)

Tylo's two FP8 twin types are gone. The element types are Microfloats'
`Float8_E4M3FN` and `Float8_E5M2`, re-exported, so FP8 arrays made by other
packages feed atoms without a reinterpret. What the twins existed for, the
`cvt.rn.satfinite` policy, is now a property of the conversion:
`convert_element` rounds to nearest even and saturates whatever policy the
type declares, and stores of FP32 accumulators to FP8 tiles use it.

`pack(T, f)` converts all lanes of a fragment in one call of Microfloats 0.2.7's
vector funnel, and `unpack(F, p)` is new and widens packed FP8 to `Float16`,
`BFloat16`, `Float32` or `Float64`. In a kernel compiled for a target with
the instructions each becomes one native `cvt` per two lanes
(`cvt.rn.satfinite.e4m3x2.{f32,f16x2,bf16x2}`, `cvt.rn.{f16x2,bf16x2}.e4m3x2`),
with no calls and no local memory; an sm_80 compile of the same kernel takes
the generic path. `test/gpu/packing.jl` pins both and compares the device
bits with the host for FP32, FP16 and BF16 sources, including infinities,
saturating values and subnormals. The 134 PTX files of the atom and vector
tests are identical to the previous ones apart from the type names.

The work found a defect in Microfloats 0.2.7: its target gate compared
symbols, which CUDACore 6.4 does not fold, so kernels kept the generic path
next to the native instruction behind a runtime branch. Microfloats 0.2.8
fixes it, and Tylo requires that version.

The same batch adds the first block-scaled atom. `ScaledMMAAtom{(16,8,32)}`
binds `mma.sync … kind::mxf8f6f4.block_scale.scale_vec::1X` for the four
E4M3/E5M2 pairings with `Float8_E8M0FNU` scales: one instruction multiplies
one MXFP8 block of 32 along K, so a K loop needs no dequantization. The ISA
describes the scale selectors with figures only, so the ownerships of the two
scale operands were measured on the GB10: a kernel gave every thread and
byte of the scale registers its own exponent and the result named, for each
row and column, the register it had read. Lanes `4q` and `4q+1` (or `4q+2`
and `4q+3`) supply rows `q` and `q+8`, any lane of quad `q` supplies column
`q`, and the byte selector picks one of four scales, which Tylo uses for four
consecutive K blocks. `test/gpu/scaled_atoms.jl` multiplies four and eight
blocks with random operands and scales against an FP64 oracle, exactly, for
every pairing, and pins clean assembly for sm_121a and sm_120f.
Tylo: 87,744 checks pass in 4m07s on four workers.
