# Design and boundaries

Tylo describes data and operations inside a kernel. Its two main consumers are
standalone kernels and Megakernels operations. A complete kernel still chooses
its grid, assigns work to blocks and warps, allocates storage, and schedules
reuse. Those choices can differ while using the same Tylo primitive.

## Four facts, kept separate

| Fact | Representation | What it does not establish |
|:--|:--|:--|
| Logical domain and storage | `GlobalTile`, `SharedTile`, `TmemTile`, with a layout | Which thread owns a value, or whether the storage is ready |
| Register values and ownership | `Fragment(values, ownership)` | A memory address, or a valid MMA operand representation |
| A supported operation | `CopyPlan`, `TiledMMA`, `TMALoad`, `WGMMA64`, `TmemTransfer` | A complete pipeline or an arbitrary layout conversion |
| Completion | Explicit waits/fences, with pending types for TMEM loads and WGMMA results | Allocation lifetime, converged participation, or exclusive access |

Two maps explain most of the interface:

```text
(thread, local value slot) -- ownership --> logical coordinate
logical coordinate        -- storage   --> offset within an allocation
```

The storage map does not decide register ownership. A load instruction connects
specific storage and ownership arrangements. A general layout can describe a
map even when Tylo has no compatible instruction implementation for it.

For example, `Fragment` can hold a thread's 2×2 patch and broadcast a scalar
function over its values. That alone supplies neither an MMA operand encoding
nor a reduction over a distributed axis. See [Register fragments](rows.md).

## Logical axes and physical organization

A logical row fixes coordinate 1 and varies coordinate 2. Whether those values
are contiguous in memory, scattered across lanes, or replicated is a separate
question. Julia's usual column-major arrays are one possible storage choice;
a kernel can use another layout without redefining its mathematics.

`permutedims` on supported layouts, fragments and TMEM views exchanges logical
axes. The values can remain in the same physical locations and local slots.
The matching operation also changes axes: a reduction over dimension 2 becomes
a reduction over dimension 1. Moving data into a required MMA distribution is
a different operation and can require communication.

TMEM has physical lane and word-column coordinates. Tylo's TMEM storage adapter
encodes those hardware units; logical matrix axes remain independent. The
current transfer recipe is narrow, with explicit compatibility checks. See
[TMEM tiles and transfers](tmem.md).

## Julia arithmetic is a real interface boundary

A ready fragment supports `map` and fused dotted expressions over its local
values. `exp.(f)` uses the same broadcast machinery as a user-defined scalar
function or a scalar PTX callable. The scalar implementation determines the
numerical behavior; Tylo preserves the ownership and reconstructs register
values. It does not keep a special list of tile-level activation functions.

Reductions have more obligations. `maximum(f; dims=2)` must account for every
logical value along that axis and record where the result is replicated.
Only supported ownerships have such a recipe. Broadcasting a reduction back
checks that its distribution matches the destination fragment.

This is not a complete `AbstractArray` implementation. There is no general
indexing, iteration, mutable dotted assignment, or automatic redistribution.
Packed MMA operands and pending values have stricter interfaces. WGMMA's ready
fragment has not yet joined the generic broadcast/reduction interface either.
These boundaries are listed in [the fragment support table](rows.md).

## Why static parameters and generated functions appear

Tile capacities, instruction shapes, and register slot selections usually need
to be known to inference. Julia can specialize on types and `Val` parameters;
`@Layout` makes static shape/stride leaves concise. Runtime matrix dimensions,
leading strides and storage-window offsets can remain ordinary integers.

An immutable tuple gives the compiler individual SSA values. Generated functions
build fixed accesses to those values, avoiding dynamic tuple indexing that could
materialize an array in local memory. Immutability does not imply an allocation
or a fresh physical register for every expression, but it also does not guarantee
low register pressure. Assembly and runtime resource measurements decide that.

The short layout core is a small algebra, not a hidden implementation of all
of CuTe. Much of the GPU compiler and instruction support lives in Julia,
CUDACore and PTX.jl; much of the missing generality remains missing. Review both
the supported maps and the consuming instruction methods before judging scope.

## Completion and reuse stay visible

The GEMM example has two distinct synchronization points: make the producer's
completed copies visible to consumers, then wait for all consumers before
reusing a shared stage. Neither can be replaced by a shape check.

TMEM `wait_load` and WGMMA `wait_mma` also carry compiler dependencies through
returned registers. A memory clobber alone does not force arithmetic on an
already-produced SSA value to remain after an asynchronous wait. These small
PTX adapters are deliberate; their tests inspect instruction ordering.

A pending type makes accidental arithmetic harder, but it is an ordinary Julia
struct, not a linear resource. The caller still establishes allocation lifetime,
correct barrier phases and collective participation. A wait may complete all
prior operations of its hardware scope, rather than just one Julia object.

## Position in the ecosystem

| Layer | Responsibility in this project |
|:--|:--|
| Julia and CUDACore | GPU compilation, launch, device arrays, stream and lifetime integration |
| PTX.jl | Instruction callables, operand contracts, descriptors and low-level utilities |
| Tylo | Layout-bearing views, register ownership, supported tile operations and completion wrappers |
| Kernel author / Megakernels | Work assignment, allocation, pipeline, barriers, scheduling and reuse |

ThunderKittens provides curated register/shared/TMEM tile types and compatible
operations, with a consistent cooperative-group vocabulary. Its core kernel
interface is CUDA C++, and it also has axis-parameterized reductions. Tylo
borrows the emphasis on usable hardware-compatible primitives while exposing
more of the mapping as explicit Julia objects. It currently has much less
operation and hardware coverage. See TK's
[register types](https://github.com/HazyResearch/ThunderKittens/blob/main/include/types/register/rt.cuh)
and [reductions](https://github.com/HazyResearch/ThunderKittens/blob/main/include/ops/group/register/tile/reductions.cuh).

CuTe supplies a richer tensor/layout vocabulary and many established partitions
and instruction mappings. Tylo carries forward the separation of data from its
mapping, without claiming a comparable algebra or operation set. The
[CuTe tensor implementation](https://github.com/NVIDIA/cutlass/blob/main/include/cute/tensor_impl.hpp)
is a useful reference for that distinction.
[cuTile.jl](https://github.com/JuliaGPU/cuTile.jl) has a different division of
responsibility: its compiler controls substantially more of tile lowering.
Tylo exposes concrete ownership, storage and participation to the kernel author.

From the earlier Laythe/Tylo prototypes, the retained ideas are hierarchical
coordinates, mixed static/runtime leaves, memory spaces and operand roles.
The present approach requires each hardware mapping to have an explicit
contract and a worked consumer. An independent Laythe package is a possible
future extraction of shared mathematics, not a prerequisite for this codebase.

## Decisions still worth challenging

- **Plan ergonomics.** `CopyPlan{...}()`, `WGMMA64(..., Val(...))` and
  `partition(TmemTransfer{...}(), tile)` expose related choices in different
  ways. Their contracts are useful; a coherent convenience layer is unfinished.
- **Fragment coverage.** Generic scalar arithmetic is broader than reductions,
  register windows or stores. WGMMA and `SoftmaxState` retain specialized APIs.
  A new ownership should acquire operations through concrete mappings and tests.
- **Collective scope.** Tylo has no single counterpart to TK's `group<N>` yet.
  Whether one improves composition should be tested against the existing warp,
  warpgroup and producer/consumer kernels.
- **Hardware-shaped storage.** TMEM typed-slot strides and canonical TMA layouts
  expose real constraints, but callers should not have to rederive common
  compatible constructions in every kernel.
- **Performance.** Equivalent machine code for a checked replacement establishes
  that replacement's cost on that toolchain. It does not establish efficient
  scheduling, broad performance parity or good register pressure for new shapes.

The next useful improvements should remove friction in a complete consumer or
add a demonstrated missing operation. A larger catalogue of descriptors without
such a consumer would not by itself make the library more composable.
