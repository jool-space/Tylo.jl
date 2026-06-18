export allocate_checkpoints
export allocate_scratchspace
export Duplicated

"""
    Duplicated(primal, shadow)
    Duplicated(primal)

A differentiable tensor paired with its gradient `shadow` — the idea borrowed
from Enzyme's `Duplicated` / Mooncake's `CoDual`, and matching this package's
existing [`grad_shadow`](@ref) vocabulary.

Pass these positionally to the in-place `∇`-kernels (e.g. [`∇rms_norm!`](@ref)):
the backward pass reads each argument's `primal`/`shadow` as it needs and writes
the input gradients into the corresponding `shadow`. Semantics are *overwrite*,
not accumulate — cross-call gradient accumulation is the AD backend's job.

Functions that only consume primals (the forward kernels) accept a bare array or
a `Duplicated` interchangeably via [`primal`](@ref), so the same wrappers can be
threaded through both passes.

`Duplicated(x)` pairs `x` with a fresh, uninitialized `similar(x)` shadow.
"""
struct Duplicated{P,S}
    primal::P
    shadow::S
end

Duplicated(x) = Duplicated(x, similar(x))

@inline primal(x) = x
@inline primal(d::Duplicated) = d.primal

@inline shadow(d::Duplicated) = d.shadow

"""
    allocate_checkpoints(f, args...; kwargs...) -> NamedTuple

Allocate a primitive's **checkpoints** — the saved activations that bridge the
forward and backward passes (e.g. `Rstd` for RMSNorm, `Mean`/`Rstd` for
LayerNorm, `M`/`L` for attention). These are *forward-only* to produce: dispatch
on the forward (`::typeof(rms_norm!)`), the forward writes them, and the backward
reads them. A primitive that saves nothing simply has no method.

Pass the result as the `checkpoints` kwarg. Distinct from
[`allocate_scratchspace`](@ref), which is a single pass's transient workspace.
"""
function allocate_checkpoints end

"""
    allocate_scratchspace(f, args...; kwargs...) -> NamedTuple

Allocate a kernel's **scratch workspace** — transient buffers live only within a
single pass (intra-kernel scratch, or data handed between launches of the same
primitive in one pass; e.g. the backward reduction's `W̄_partial`/`Locks`).
Dispatched *per function* (`::typeof(rms_norm!)` vs `::typeof(∇rms_norm!)`), so a
forward and a backward each declare their own; a pass that needs none has no
method.

Pass the result as the `scratch` kwarg. Forward- and backward-scratch lifetimes
are disjoint, so same-shape ones may be aliased to cut peak memory. Distinct from
[`allocate_checkpoints`](@ref), which bridges forward→backward.
"""
function allocate_scratchspace end
