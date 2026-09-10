# Read the implementation

Read this alongside the [GEMM walkthrough](gemm.md). Paths below are relative
to the Tylo repository root. The files are intentionally small, but their
contracts depend on one another and on the upstream GPU compiler and PTX
bindings.

## Where the concepts live

| Source | Responsibility | Read with |
|:--|:--|:--|
| `src/Tylo.jl` | Public bindings, file ordering and device-operation declarations | `Project.toml` for extension triggers |
| `src/layouts/layouts.jl`, `src/layouts/` | Affine coordinate maps, static notation, composition, swizzles and windows | `test/layouts.jl`, `test/layout_macro.jl` |
| `src/memory.jl` | Typed global/shared pointers plus storage layout; address-unit conversion | `ext/copy.jl`, `test/gpu/gemm.jl` |
| `src/fragments.jl`, `src/arrayops.jl` | Local values plus ownership, generic scalar broadcast, axis views and supported reductions/windows | `test/arrayops.jl`, `test/gpu/arrayops.jl` |
| `src/rows.jl`, `ext/rows.jl` | Particular ownerships and local/warp/MMA reduction recipes; compatibility names | `test/rows.jl`, `test/gpu/rows.jl` |
| `src/copy.jl`, `ext/copy.jl` | Vector copy assignment, structural validation, full and bounded copies | `test/gpu/boundaries.jl` |
| `src/mma.jl`, `ext/mma.jl` | Warp MMA roles, packing, repetitions, shared loads and ownership-based stores | `test/gpu/gemm.jl`, `test/gpu/operand_a.jl` |
| `src/tma.jl`, `ext/tma.jl`, `ext/CUDACoreExt.jl` | Canonical TMA storage, descriptor preparation and launch/lifetime binding | `test/hopper.jl`, `test/gpu/tma.jl` |
| `src/wgmma.jl`, `ext/wgmma.jl` | Warpgroup plan, shared descriptors, partial accumulators and register-dependent completion | `test/gpu/wgmma.jl` |
| `src/tmem.jl`, `ext/PTXExt.jl` | TMEM address mapping, transfer partitions, pending loads and packed stores | `test/tmem.jl`, `test/gpu/tmem_views.jl`, `test/gpu/tmem.jl` |
| `src/online.jl` | Current online softmax state, update, merge and normalization | `test/online.jl`, `test/gpu/online.jl` |

`using Tylo` loads the pure descriptions and host operations. Loading PTX
activates `PTXExt`, which supplies device methods. Loading CUDACore as well
activates `CUDACoreExt` for descriptor preparation and adaptation of owned
resources into device bindings. The `ext/` directory is therefore part of the
implementation, not optional example code.

## Follow a broadcast into registers

For `g = scalar_function.(f .- m)`:

1. Julia builds a fused `Broadcasted` expression. Tylo's `FragmentStyle` selects
   its own materialization path.
2. `_broadcast_anchor` finds a fragment carrying the full result ownership.
   `_check_broadcast` checks compatible ownership and supported expansion of
   reduced axes. Equal logical shapes alone are insufficient.
3. `_materialize_fragment` generates an expression for each static local slot.
   `_broadcast_value` recursively applies ordinary scalar calls to that slot's
   inputs. A reduced input selects the corresponding replicated result slot.
4. `_rebuild_fragment` returns values with the result ownership. Julia and the
   GPU backend compile those scalar expressions; Tylo has no `exp` special case.

Read these functions in `src/arrayops.jl`. The generated code specializes on
local slot count and representation, rather than building a device loop over a
heap array. A custom scalar PTX operation uses this same path.

For `sum(f; dims=2)`, first follow `_fragment_reduce`, `_reduced_layout` and
`_reduce_values`. The last dispatch chooses a known local or shuffle-based
recipe. The mathematical layout description does not synthesize arbitrary
communication. The old `row_*` names still visible in the sources implement
these selected recipes and compatibility APIs.

## Follow a TMEM load into arithmetic

`window(tile, origin, Val(shape))` preserves the allocation-relative storage
map. `partition(transfer, view)` checks the shape, supported affine strides,
word packing and hardware band, then produces the address and transfer
ownership required by the instruction.

`load_async(partition)` issues the PTX load and returns `PendingLoad`. In
`wait_load`, `_wait_words` threads those values through tied assembly operands
before constructing a ready `Fragment`. Subsequent broadcast uses the ordinary
path above. `store_async!` checks ownership against the destination partition;
completion and storage reuse remain explicit.

The types check selected structural facts. They do not establish which warp is
actually executing, that every lane participates, or that the allocation has
sufficient capacity. Read the source docstrings and [TMEM contract](tmem.md)
together.

## What happens when?

| Phase | Examples |
|:--|:--|
| Julia syntax expansion | `@Layout` rewrites leaf expressions; `ptx"..."` builds an instruction-construction expression |
| Host planning and preparation | `gemm_config`, `validate_copy`, descriptor upload in `prepare_tma` |
| Method specialization | `@generated` methods emit fixed register accesses and instruction repetitions from type parameters |
| GPU execution | Runtime address arithmetic, scalar operations, copies, collectives, waits and barriers |

The WGMMA generator now contains:

```julia
# Inside the generator, where T and N are known from argument types:
dtype = T === Tylo.BFloat16 ? "bf16" : "f16"
instruction = ptx"wgmma.mma_async.sync.aligned.m64n$(N)k16.f32.$dtype.$dtype"
```

PTX.jl constructs a concrete callable operation. Interpolating `$instruction`
into the expression returned by the generator embeds that operation. No runtime
instruction-string construction is required in the kernel. Ordinary `@eval`
loops in the MMA and TMEM extensions similarly create method families when the
extension loads. These are different uses of specialization; neither provides
a general-purpose dynamic instruction selector on the device.

`@Layout`'s `$` marker has its own macro meaning: preserve the supplied value's
type. Unmarked values pass through `static`. It does not turn an unknown kernel
argument into a compile-time constant. See [Static layout notation](layouts.md#Static-layout-notation).

## Read a complete consumer

| Consumer | What it establishes | Policy that remains in the consumer |
|:--|:--|:--|
| `examples/gemm/kernel.jl` | Global/shared/register/MMA/store composition | Tile choice, grid, shared allocation and copy pipeline |
| `examples/softmax/kernel.jl` | Generic arithmetic and reductions across selected ownerships | Physical input layout, masks and normalization domain |
| `examples/streaming_attention/kernel.jl` | QK → online statistics → packed operand A → PV → normalization | D=64, chunk sizes, shared staging and traversal |
| `examples/hopper/kernel.jl` | TMA and WGMMA share a storage contract | Producer/consumer roles, barriers and stage reuse |
| `examples/flash_attention/tiles.jl` | TMEM correction/epilogue replace two raw helpers without changing checked machine code | The rest of the kernel remains the pinned PTX reference |

The standalone streaming-attention kernel and the datacenter FlashAttention
replacement experiment are separate consumers. The former is a complete Tylo
kernel that runs on GB10; the latter replaces only two helpers in a much larger
reference kernel and still needs datacenter Blackwell runtime validation.

## How to judge a change

A new operation should make its logical domain, local payload, participating
threads, memory representation and completion rules clear. Look for an
independent coordinate or numerical reference, then assembly checks and a
runtime consumer on applicable hardware. A passing host layout test cannot
establish instruction legality; successful assembly cannot establish barrier
correctness or performance.

The code has a general description layer and a smaller collection of operation
recipes. That separation is useful only if unsupported combinations fail clearly
and supported ones remain straightforward to use. The current gaps are listed
in [Design and boundaries](design.md#Decisions-still-worth-challenging) and
[Current status](validation.md).
