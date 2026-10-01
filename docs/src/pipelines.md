```@meta
CurrentModule = Tylo
```

# Pipelines

A warp-specialized kernel keeps barriers, buffers and scalars in shared
memory and passes stages between roles. Tylo describes the memory and
computes which barrier and which phase each wait needs. The kernel still
issues every wait, arrival and fence itself, so the same pieces serve a
standalone kernel and a megakernel's task bodies.

## Shared memory

A [`SharedLayout`](@ref) names the groups of a kernel's dynamic shared
memory:
- [`barriers`](@ref), with their arrival counts;
- [`scalars`](@ref);
- [`buffers`](@ref), with their alignment.

Declare it once as a function of the kernel's constants. The host calls the
same function for the launch's byte count, and the kernel binds it to its
memory:

```julia
@inline layout(cfg, a, b) = SharedLayout((;
    full  = barriers(cfg.stages; arrivals=1),           # TMA bytes
    empty = barriers(cfg.stages; arrivals=1),           # tcgen05.commit
    tmem  = scalars(UInt32),
    stage = buffers(transfer_bytes(a) + transfer_bytes(b), cfg.stages; align=1024)))

shmem = shared_bytes(layout(cfg, ta, tb))                                 # host
smem = bind(layout(cfg, a, b), pointer(CuDynamicSharedArray(UInt8, 0)))   # kernel
thread == 0 && (initialize!(smem); fence_barrier_init())
```

Groups aligned to at most 16 bytes come first, in order. More strongly
aligned groups follow from the first address past them that has that
alignment, found when the layout is bound, so swizzled stages start on
1024-byte boundaries however the dynamic memory begins. Every offset is a
constant in the machine code.

Each group is a property of the bound layout:
- `smem.full[s]` is a [`Barrier`](@ref);
- `smem.tmem[]` and `smem.sums[i]` read and write scalars at their
  alignment (a [`Scalars`](@ref) view);
- `smem.stage[s]` is a buffer's shared address, for `shared_tile(binding,
  smem.stage[s])`.

## Rings

A [`Ring`](@ref) pairs a group of barriers that complete when a stage is
full with an optional group that completes when it is empty. It holds no
state. `ring[c]` is the `c`-th use of the ring, counted from zero since
`initialize!`:

```julia
kv = Ring(smem.full, smem.empty)
s = kv[c]                         # the producer's c-th fill
acquire(s)                        # free: at once in the first round, then the previous round's release
arrive!(s.full, bytes); tma_load!(shared_tile(b, smem.stage[s.slot]), b, origin, s.full)

s = kv[c]                         # a consumer's c-th use
wait(s)                           # full
mma(...); commit_mma(atom, s.empty)
```

`s.slot` is the stage's slot, `c % stages`, and its phase is the round's
parity, `c ÷ stages`. A producer and its consumers each count their own
uses. Look-ahead is `kv[c+1]`, and releasing an older stage is
`kv[c-1].empty`. A single barrier, or a pair, is a ring of one stage:
`Ring(smem.sfull[i])[k]` waits for phase `k % 2`. `selectdim(smem.summed,
1, i)` turns a row of a two-dimensional group into a ring of its own.

A ring's arithmetic is unsigned, so a power-of-two stage count costs a mask
and a shift, and any other count costs a multiply. An optional timeout,
[`wait_timeout`](@ref), turns a hang into a report and a trap while
debugging.

## Named barriers

[`named_barriers`](@ref) numbers a kernel's hardware barriers from 1 (or
from `first`, past barriers a megakernel reserves). [`wait`](@ref) on a
[`NamedBarrier`](@ref) is `bar.sync`, and [`arrive!`](@ref) is `bar.arrive`.
Groups take runtime indices:

```julia
nb = named_barriers((; handoff = named(8; threads=64), merge = named(; threads=128)))
arrive!(nb.handoff[4i + band]); wait(nb.handoff[4i + band]); wait(nb.merge)
```

A named barrier paces one handoff per phase. Two arrivals in a row from one
side can complete it on their own. A second handoff needs an mbarrier.

## Arrivals

A local [`arrive!`](@ref) returns an [`Arrival`](@ref).
[`after`](@ref)`(arrival, x)` makes work computed from `x` wait for the
arrival. Nothing else orders arithmetic after an arrival, which is a store
as far as the compilers know. It costs a comparison and a selection.

## Configuration

A [`Config`](@ref) carries a kernel's constants as one argument, replacing
trailing `Val`s. `cfg.stages` is a constant in the kernel, integers come out
as `Int32`, and each configuration compiles its own kernel. Inside closures
a configuration keeps its constants, because they live in its type, while a
captured value would not. Pass `cfg` itself into helpers, not
`cfg.emulate`.

## Costs

Each of these compiles to what a hand-written kernel would contain:
- a bound layout is a base address plus constant offsets;
- a ring stage is an index with unsigned division by a constant;
- a named barrier is one instruction;
- an arrival's token is the register the arrival already writes.

The ported kernels' machine code differs from their hand-written versions
in address arithmetic only. The commit history records each comparison
(`test/tools/sassdiff.jl`).
