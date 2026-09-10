```@meta
CurrentModule = Tylo
DocTestSetup = :(using Tylo)
```

# Tylo

Composable tile programming for NVIDIA GPUs in Julia. The implementation includes
shared/global layouts, asynchronous copies, warp MMA, TMA, Hopper WGMMA,
and complete GEMM and streaming-attention pipelines,
alongside row-distributed register fragments and explicit TMEM views/completion.
See [Representation and completion](@ref) for the boundary and current limits.

```jldoctest
julia> f = RowFragment((1f0, 2f0, 3f0, 4f0));

julia> columns(scale(f, 0.5f0), Val(2), Val(2)).data
(1.5f0, 2.0f0)
```

```@autodocs
Modules = [Tylo, Tylo.Layouts]
```
