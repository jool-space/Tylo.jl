```@meta
CurrentModule = Tylo
DocTestSetup = :(using Tylo)
```

# Walk through GEMM

Start with `examples/gemm/kernel.jl`. It computes mathematical
`D = alpha * A * B`, optionally applying ReLU in the epilogue. It is a complete
warp-MMA kernel: the example owns its grid, shared allocation and one- or
two-stage copy pipeline. Tylo supplies the reusable operations inside it.
The source is [here](https://github.com/jool-space/Tylo.jl/blob/main/examples/gemm/kernel.jl).
This walkthrough follows the default full-tile configuration; the example also
has a [bounded path](boundaries.md).

## 1. Choose the work and the storage separately

A `TiledMMA` chooses an instruction atom, a warp arrangement, a grid of atom
repetitions per warp, and the K extent consumed by one call:

```jldoctest
julia> atom = MMA16x8x16(Tylo.BFloat16);

julia> plan = TiledMMA(atom, Val((2, 2)), Val((2, 4)), Val(32));

julia> size(plan), Tylo.threads(plan)
((64, 64, 32), 128)

julia> acc = zero_accumulator(plan);

julia> length(Fragment(acc).data)
32
```

There are four warps. Each warp covers 32×32 output values by repeating a
16×8 atom two times along M and four times along N. Each thread therefore holds
32 accumulator values; all 128 threads together own the 64×64 output tile.
The plan consumes K=32 through two K=16 instruction steps. Register counts come
from the atom's ownership and repetitions, not from a general assumption that
an arbitrary tile can be split uniformly over threads.

In `gemm_config`, A's shared logical shape is `(64,32)` and B's is `(32,64)`.
Both are K-contiguous before optional swizzling. The warp arrangement does not
fix those storage strides. The configuration separately constructs layouts,
copy plans, shared spans and the number of buffering stages.

This configuration runs on the host. Its ordinary input dimensions become
static parameters deliberately, choosing a compiled kernel configuration.
Runtime problem dimensions and leading strides remain kernel arguments.

## 2. Bind runtime global arrays

The kernel binds pointers to logical views:

```julia
# Inside the kernel; pointers, dimensions and leading strides are arguments.
a = GlobalTile(pointer(a_data), @Layout(($m, $k), ($lda, 1)))
b = GlobalTile(pointer(b_data), @Layout(($k, $n), (1, $ldb)))
d = GlobalTile(pointer(out),    @Layout(($m, $n), (1, $ldc)))
```

A is logically `(M,K)` but physically K-contiguous, typically a Julia array
stored as `(K,M)`. B is logically and physically `(K,N)`, while output uses
Julia's usual M-contiguous `(M,N)` storage. This is why the examples prepare
A's storage explicitly. The mathematics remains `A * B`.

The `$` markers preserve the runtime dimensions/strides. Layout coordinates,
block origins and thread indices are zero-based; Julia array indexing and
`threadIdx().x` start at one. The example subtracts one at that boundary.

`GlobalTile` and `SharedTile` borrow storage. Constructing one does not copy
values, allocate a buffer or make it ready. A `window` changes the logical
region while retaining the parent's allocation and mapping.

## 3. Copy one K stage

For each stage, `prefetch_stage!` finds the shared A/B regions, selects the
corresponding global windows, issues their copies, then commits one copy group.
The following is an excerpt of that dataflow, with the surrounding setup omitted:

```julia
copy_async!(config.ac, shared_a, global_a_window, tid)
copy_async!(config.bc, shared_b, global_b_window, tid)
commit_copies()
```

A `CopyPlan` assigns 16-byte vectors to the participating threads. Its vector
axis is logical K for both operands, even though K is axis 2 of A and axis 1 of
B. Host validation checks each layout for vector contiguity/alignment and
unique destination addresses. The caller additionally establishes pointer
alignment and allocation bounds.

The bounded overload accepts a full source view and origin. At edges it combines
asynchronous vector copies with scalar loads/stores and zero-fill. It retains
the same explicit completion and synchronization requirements.

## 4. Consume, then release, the shared stage

The main loop has this shape:

```julia
# Excerpt: the real loop chooses groups_remaining when filling/draining.
wait_copies(Val(groups_remaining))
sync_threads()
acc = mma(config.plan, shared_a, shared_b, acc, tid)
sync_threads()
# It is now safe to prefetch new inputs into this shared stage.
```

The two barriers have different purposes:

| Position | Obligation |
|:--|:--|
| After the copy wait | Every consumer must observe all participating producers' completed writes |
| After MMA has read shared memory | Every reader must finish before producers overwrite the stage |

Copy groups belong to the issuing thread. `wait_copies(Val(1))` allows at most
one committed group to remain pending; it does not identify a particular
buffer or synchronize the CTA. The two-stage example uses this while another
stage is pending, and drains with `Val(0)` at the end. That rule is part of the
example's schedule, not a general rule for every two-buffer pipeline.

Inside `mma(plan, ...)`, the extension selects shared windows for each warp,
loads packed A/B registers with `ldmatrix`, repeats `mma.sync.m16n8k16`, and
returns the updated accumulator. There is no hidden allocation or CTA barrier.
This warp instruction's result is ready for register arithmetic after the call;
the asynchronous WGMMA path has a different completion API.

## 5. Apply the epilogue and store by ownership

The example uses a scalar callback:

```julia
result = map(x -> max(alpha*x, 0f0), acc)
store!(config.plan, output_window, result, tid)
```

The equivalent arithmetic can be written `max.(alpha .* acc, 0f0)`. Each thread
transforms its own values. The store uses the accumulator's ownership to
recover logical output coordinates, then the destination layout to form
addresses. It cannot assume that adjacent local registers are adjacent output
matrix elements. No extra kernel is needed for this epilogue.

This configuration has two warps along N. A full reduction over the output's
N dimension would need communication between those warps; Tylo's current
register reduction recipes deliberately reject it. Changing the epilogue from
pointwise ReLU to softmax introduces a new collective requirement.

## What to inspect when judging the example

Follow `gemm_config` → `prefetch_stage!` → `tiled_gemm_kernel!`, then the
`Tylo.mma(::TiledMMA, ...)` and `store!` methods in `ext/mma.jl`.
`test/gpu/gemm.jl` checks independently decoded operand values, complete
products, storage padding and repeated buffer reuse. `test/gpu/boundaries.jl`
covers partial tiles.

The example establishes that these operations compose into a working kernel.
It does not establish that this schedule or tile choice is best for a given
GPU. Resource counts, occupancy, available blocks and memory traffic still
matter. [Current status](validation.md) points to the measured evidence and
its limits.
