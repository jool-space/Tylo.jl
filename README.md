# Tylo

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jool-space.github.io/Tylo.jl/dev/)
[![Build Status](https://github.com/jool-space/Tylo.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jool-space/Tylo.jl/actions/workflows/CI.yml)

Composable tile programming for NVIDIA GPUs in Julia.

Tylo represents the data a kernel operates on: register fragments, memory
views, and the operations that move between them. PTX.jl supplies the
instructions. The kernel controls its work assignment and synchronization.

This is an experimental implementation with these worked consumers:

- A complete BF16/FP16 tiled GEMM: shared-memory layouts, asynchronous copies,
  register fragments, warp MMA, bounded copies/stores and composed epilogues,
  executable on CC 8.0+.
- TMA plus Hopper WGMMA: a complete producer/consumer GEMM and
  Megakernels.jl’s GEMM/gate-up projection. TMA runs on GB10; WGMMA has
  SM90a assembly coverage and prepared H100/H200 runtime tests.
- Row reductions and broadcasts with lane-local, warp-striped, and MMA ownership;
  masked softmax examples and Megakernels normalization consumers.
- Correction and epilogue replacements in PTX.jl's datacenter Blackwell
  attention kernel, with typed TMEM views and explicit completion.

Layouts support hierarchical shapes/strides, mixed static/runtime leaves,
composition, XOR swizzles, factorization and windows that preserve swizzle
phase. MMA ownership and packed register representations are separate from
physical memory layout.

See [the complete GEMM](examples/gemm/README.md) for the data path and an
executable demo. See [the design](docs/src/layouts.md) for contracts and limits.

The example's epilogue uses ordinary Julia:

```julia
values = wait_load(load_async(chunk))
part = columns(values, Val(0), Val(32))
packed = pack_bf16(scale(part, inv_sum))
store_row!(destination, packed)
```

The kernel can release its readout barriers after `wait_load`, before the
conversion and global stores. Tylo does not silently insert a CTA barrier.

## Scope

Warp MMA supports `mma.sync.m16n8k16`. Hopper WGMMA supports M=64,
N=8:8:256, K=16/32/64 with BF16/FP16 inputs and FP32 accumulation. TMA
supports one explicit 128-byte-swizzled K=64 storage format. See
[the TMA/WGMMA contracts](docs/src/hopper.md).

The library is not yet a general CuTe or ThunderKittens equivalent.
Tcgen05 MMA, arbitrary redistribution, allocation
management and automatic pipelines remain future work.

`Tylo.Layouts` contains the pure coordinate mathematics. A separate Laythe
package can follow once multiple consumers establish a stable boundary.

Megakernels.jl consumes the TMA/WGMMA path in `HopperProjection`; it owns
task scheduling and producer/consumer buffer reuse.
Standalone kernels can use Tylo without Megakernels. cuTile remains a
separate compiler-driven approach.

## Development

Host tests need Julia 1.10+ and no GPU:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The GPU development environment needs Julia 1.12+ and a sibling PTX checkout:

```sh
julia --project=test/gpu -e 'using Pkg; Pkg.instantiate()'
julia --project=test/gpu test/gpu/runtests.jl
julia --project=test/gpu test/gpu/runtests.jl --attention
```

`using Tylo, PTX, CUDACore` activates the device implementations. Host-only
use does not load CUDA. The GPU suite assembles warp GEMM for SM80, SM90a, SM100a and SM121a,
plus Hopper pipelines for SM90a and TMEM probes for SM100a. Warp GEMM,
TMA and register arithmetic execute on GB10. WGMMA execution requires SM90.
TMEM and full attention execution require B200/B300 and are explicitly
skipped elsewhere. Compilation is not hardware validation.

See [the attention experiment](examples/flash_attention/README.md) for the
reference pin, paired benchmark, and evidence commands.
