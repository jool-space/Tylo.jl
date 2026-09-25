# Status

Tylo is experimental: the vocabulary is settling, and names may still
change. This page says what exists, what evidence supports it, and where it
stops.

## Coverage

| Area | Implemented | Evidence | Limits |
|:--|:--|:--|:--|
| Layouts | Hierarchical shapes and strides with static and runtime leaves; composition, coalescing, complements, inverses, divisions, products, slices, swizzles, windows | Host tests against enumeration, including randomized layouts | Complements and inverses need static strides |
| Ownership | Every distribution as a canonical mapping; `distribute`, `repeat`, `permutedims`, replication | Derived plans checked against enumeration for the atoms' ownerships and random ones | Tiles are two-dimensional |
| Fragments | Broadcasting, conversion, reductions on either axis or both (any type within a thread, 32-bit across lanes), windows, in-lane relayouts, coordinates | Host tests; GB10 runtime | No redistribution across lanes beyond reductions |
| Loads, stores, copies | `ldmatrix`/`stmatrix`, 16/8/4-byte vectors within declared alignments, scalars; `cp.async` with zero-fill; bounded and swizzled tiles | GB10 runtime, including every bound position and alignment; assembly on sm_80 to sm_121 | |
| Warp MMA | 24 `mma.sync` atoms (16-bit, TF32, FP8, INT8), block-scaled MXFP8, `TiledMMA` | GB10 runtime oracle for every atom; complete GEMM, ragged and full | Not a tuned GEMM library |
| Barriers and TMA | mbarriers; 2-D loads and stores of 1-, 2- and 4-byte elements with 32/64/128-byte swizzles and up to eight stripes, batched matrices, logical bounds | GB10 and H200 runtime | No multicast, clusters or unswizzled boxes |
| WGMMA | Shared/shared M=64, N=8:8:256, K=16/32/64, partial accumulators, pipelined issue | H200 runtime for every shape, partials and stages; descriptors checked against the PTX ISA's canonical layouts for every element and K step; SASS checked for ptxas serialization | |
| tcgen05 and TMEM | `.32x32b` TMEM loads and stores (FP32, packed BF16/FP16, 1–128 words); `Tcgen05Atom` (f16, tf32, f8f6f4, i8; M=128) with shared (K- or MN-major, multi-stripe) or TMEM A | Descriptors and instruction descriptors checked against the PTX ISA; sm_100a assembly; transfers executed on a B200 | **tcgen05 MMA not yet run on a B200**; M=64 and `cta_group::2` not described |
| Examples | GEMM, softmax, streaming attention, Hopper GEMMs (a small pipeline, and a warp-specialized one), datacenter FlashAttention pieces | GB10 runtime; the Hopper GEMMs on an H200 and an H100 NVL, where the warp-specialized BF16 GEMM reaches 95% of cuBLAS at 4096³ and 97% at 8192³; the datacenter pieces by assembly | |

## Running the tests

The test project is a workspace member, so one instantiate covers the
package, its tests and its docs:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(; workspace=true)'
julia --project=test test/runtests.jl --jobs=4          # host and GPU tiers
julia --project=test test/runtests.jl host              # host only
julia --project=test test/runtests.jl gpu/gemm gpu/atoms
```

`host/` needs no GPU. `gpu/` files always compile their kernels and check the
assembly; their runtime sections run when the device satisfies the file's
`# TEST_TARGET:` banner. `TYLO_REQUIRE_GPU_RUNTIME=true` fails instead of
skipping without a GPU, `TYLO_EVIDENCE=<dir>` saves every test kernel's PTX
and cubin for comparison, and `TYLO_PROPERTY_SCALE=<n>` multiplies the
randomized cases.

A change that touches code generation is checked by saving evidence before
and after it and comparing the SASS of every kernel. The commit history
records those comparisons.

## On Hopper and Blackwell hardware

The suite has run on a GB10 (sm_121), an H200 and an H100 NVL (sm_90),
where WGMMA, TMA and barriers execute. tcgen05 MMAs have not run on a B200 yet; their
operand descriptors are checked on the host against the PTX ISA: every
descriptor Tylo emits is decoded with the specification's bit layout, and
the canonical layout it describes must read each element where the tile
stores it. On a B200/B300, run the suite with
`TYLO_REQUIRE_GPU_RUNTIME=true`: `gpu/tcgen05`, `gpu/tmem` and
`gpu/flash_attention` then execute their kernels and compare numbers.

Assembly checks compile for fixed targets, but the PTX ISA version depends
on the CUDA toolkit, which follows the driver: a CUDA 12 driver gives PTX
8.8, without the FP8 conversions from and to BF16 pairs, and the checks
expect native instructions only where the emitted ISA has them.
