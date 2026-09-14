```@meta
CurrentModule = Tylo
DocTestSetup = :(using Tylo)
```

# Tylo

Tylo is an experimental library for writing NVIDIA GPU kernels in Julia. It
provides logical memory views, distributed register values, and specific copy
and matrix instructions that connect them. PTX.jl supplies instruction bindings;
the kernel author controls the launch, work assignment, allocation and pipeline.

The central question is **which values live where, who holds them, and when
may the next operation use them?** Tylo makes these different facts explicit.
The implementation has complete GEMM and streaming-attention examples, but a
much smaller operation set than CuTe or ThunderKittens.

## A reading path

1. [Design and boundaries](design.md): the model, the decisions behind it, and
   the tradeoffs that remain open.
2. [Layouts](layouts.md) and [register fragments](rows.md): follow a logical
   coordinate into memory or into a thread's registers. These pages include
   examples you can run without a GPU.
3. [Walk through GEMM](gemm.md): see the pieces in one complete kernel, including
   who waits and who releases a shared buffer.
4. [Read the implementation](codebase.md): source files, dispatch paths, generated
   code, and the tests that establish their contracts.
5. [Current status](validation.md): distinguish implemented behavior, assembly
   evidence, runtime coverage, and unfinished work.

Then choose a hardware path: [TMA/WGMMA](hopper.md), [TMEM](tmem.md),
[streaming attention](streaming.md), or [boundary tiles](boundaries.md).
[API reference](api.md) collects docstrings separately from the explanation.

## A fragment is one thread's share

The register distribution of a warp matrix-multiply accumulator can be inspected
on the CPU. Here 32 threads would jointly own a 16×8 logical tile, with four
FP32 values per thread:

```jldoctest
julia> using Tylo

julia> atom = MMAAtom((16,8,16),Tylo.BFloat16);

julia> f = Fragment(zero_accumulator(atom));

julia> size(Tylo.Layouts.layout(f)), length(f.data)
((16, 8), 4)

julia> [Tylo.Layouts.coordinate(Tylo.Layouts.layout(f), 0, Val(i)) for i in 0:3]
4-element Vector{Tuple{Int64, Int64}}:
 (0, 0)
 (0, 1)
 (8, 0)
 (8, 1)

julia> (2f0 .* (f .+ 1f0)).data
(2.0f0, 2.0f0, 2.0f0, 2.0f0)
```

The tuple is local payload; its four entries are not a four-element logical
matrix. Elementwise arithmetic works locally. A logical reduction may need
values from other lanes and therefore a GPU collective. A memory layout is a
separate map: these logical coordinates do not say where a global or shared
matrix is stored.

## What is usable now?

The warp-MMA GEMM, fragment arithmetic, TMA copies and complete BF16 streaming
attention have GB10 runtime coverage. Hopper WGMMA and datacenter Blackwell
TMEM paths assemble, with hardware tests prepared but not yet executed on their
required devices. Tylo does not yet provide tcgen05 MMA, arbitrary register
redistribution, automatic allocation or a uniform array interface for every
representation. See [the capability table](validation.md).

## Run and inspect

From this checkout, host tests need no GPU:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The GPU environment expects a sibling `../PTX` checkout and Julia 1.10 or later:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(; workspace=true)'
julia --project=test test/runtests.jl --jobs=4
julia --project=test examples/gemm/run.jl 65 97 73
```

Current local checks use Julia 1.13. Host compatibility is declared from Julia
1.10; recent local host checks used 1.11 and 1.13. For the pinned datacenter
attention comparison, assembly artifacts, and documentation build instructions,
see [Current status and validation](validation.md).
