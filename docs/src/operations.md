```@meta
CurrentModule = Tylo
```

# Hardware operations

Each operation takes the same kinds of argument: ownerships, tiles and
fragments. It derives the instructions from their layouts while the kernel
compiles, and rejects what the hardware cannot do. Nothing here waits or
synchronizes on its own; the table at the end lists the completion verbs.

## Loads and stores

`load_fragment(ownership, tile, thread)` and `store!(tile, fragment,
thread)` pick, in order:

1. `ldmatrix`/`stmatrix` for 8- and 16-bit values in shared memory, when
   the ownership decomposes into 8×8 [`CopyAtom`](@ref) blocks along the
   tile's contiguous axis (`stmatrix` from sm_90);
2. vectors of 16, 8 or 4 bytes where the ownership's slots are contiguous
   and aligned in the tile, within its declared alignment, through
   swizzles, permuted K orders and hierarchical layouts;
3. scalars, which are correct for every ownership.

`load_fragment(atom, role, tile, thread)` loads an atom's register operand.

## Asynchronous copies

`copy_async!(ownership, shared, global, thread)` copies each thread's
values from a global tile to a shared tile of the same shape with
`cp.async`, grouping slots into the widest vectors that are contiguous and
aligned in both tiles. [`distribute`](@ref Layouts.distribute) gives the
usual coalesced distribution:

```julia
copies = distribute((64, 32), 128; vector=8, axis=2)   # BF16, 16-byte vectors along K
copy_async!(copies, sa, tile(bound(a, size(a)), (64, 32), (i, k)), tid)
commit_copies()
wait_copies(0); sync_threads()
```

A bounded source zero-fills what lies outside it through `cp.async`'s
src-size operand, cutting a vector where the bound falls. Replicated
threads issue nothing. Values that form no vector are copied by ordinary
loads and stores, which become visible to other threads only through the
barrier that follows the wait.

## Warp matrix multiply

An [`MMAAtom`](@ref) is one `mma.sync` instruction: 16-bit, TF32, FP8 and
INT8 shapes, 24 in all ([`instruction_atoms`](@ref)). Its operands are
fragments in `operand_layout(atom, role)`:

```julia
atom = MMAAtom((16, 8, 16), BFloat16)
a = load_fragment(atom, OperandA(), window(sa, (r, k), (16, 16)), lane)
b = load_fragment(atom, OperandB(), window(sb, (k, c), (16, 8)), lane)
acc = mma(atom, a, b, acc)
```

A [`ScaledMMAAtom`](@ref) multiplies MXFP8 blocks with one E8M0 scale per
row and column (`ScaleA`, `ScaleB`), and a [`TiledMMA`](@ref) repeats an
atom over warps and a warp tile: `mma(plan, sa, sb, acc, tid)` multiplies
shared tiles, and its accumulator is an ordinary fragment of the whole
block tile, stored with `store!`. [`pack_operand_a`](@ref) turns FP32
accumulator blocks into a BF16/FP16 A operand without moving values
between lanes, as attention does between its two products.

## Barriers

A [`Barrier`](@ref) is an mbarrier in shared memory. TMA loads, tcgen05
MMAs and `cp.async` copies complete on barriers, and threads wait on their
phases:

```julia
full(s) = Barrier(pointer(smem), s)        # one barrier per stage
if thread == 0
    foreach(s -> initialize!(full(s), 1), 0:stages-1)
    fence_barrier_init()                   # visible to TMA
end
sync_threads()
thread == 0 && (arrive!(full(s), transfer_bytes(binding)); tma_load!(tile, binding, origin, full(s)))
wait(full(s), phase)                       # parity 0, 1, 0, ... per reuse of the stage
```

`arrive!(barrier)` counts one arrival, `arrive!(barrier, bytes)` also
announces bytes of asynchronous transactions, `expect_bytes!` announces
them without arriving, and `commit_copies(barrier)` makes a thread's
`cp.async` copies one arrival as they land. `fence_async_shared()` orders
ordinary shared-memory writes before asynchronous reads (a TMA store, an
MMA).

## TMA

A [`TMATile`](@ref) describes a 2-D tile that the tensor memory
accelerator moves between global memory and swizzled shared storage: along
its inner axis, up to eight swizzle rows (stripes) of 32, 64 or 128 bytes.
`prepare_tma(plan, array)` encodes and uploads its descriptor on the host;
in the kernel, `tma_load!(shared_tile(binding, ptr), binding, origin,
barrier)` copies the tile (one box per stripe) and completes on the
barrier, and `tma_store!` writes back in bulk groups. The shared storage,
`shared_layout(plan)`, is the canonical swizzled encoding `ldmatrix`, WGMMA
and tcgen05 read, so a TMA tile is directly an MMA operand.

## WGMMA (Hopper)

A [`WGMMAAtom`](@ref) is a warpgroup's `wgmma.mma_async`. Its A and B
operands are descriptors of shared tiles, `operand(atom, role, tile,
origin)`:

```julia
atom = WGMMAAtom(BFloat16, 64)
c = wait_mma(mma_async(atom, operand(atom, OperandA(), sa), operand(atom, OperandB(), sb)))
c = c .* scale      # ready: a Fragment of the 64×n accumulator, like any other
```

`mma_async(atom, a, b)`, without an accumulator, starts a product from zero
without zeroing registers; `mma_async(atom, a, b, c)` accumulates onto a
ready Fragment or a pending result. A 16-bit A may come from registers, as a
Fragment in `operand_layout(atom, OperandA())`, which is the accumulator's
layout: an FP32 result converted with `Fragment{BFloat16}` is the next
product's A, as attention's probabilities are. Shared operands may be
MN-major (transposed) for 16-bit inputs, as attention's V is. An atom with
several `partials` keeps them apart until `finish_mma` sums them.

A pipeline chains issues on a pending accumulator, which needs no wait, and
`wait_mma(c, 1)` retires every group but the newest, freeing the older
stages' shared memory while the tensor cores work on the latest one:

```julia
c = mma_async(atom, a(0), b(0))
for i in 1:stages-1
    wait(full(i), phase(i))
    c = wait_mma(mma_async(atom, a(i), b(i), c), 1)
    arrive!(empty(i-1))
end
f = wait_mma(c)
```

A warp-specialized kernel gives each warpgroup one role: a producer
issues TMA loads, consumers multiply. [`release_registers`](@ref) and
`acquire_registers` move registers from the producer, which needs few, to
the consumers' accumulators, and [`warp_uniform`](@ref) makes the
warpgroup index a value ptxas knows to be uniform, so that its branches do
not count as divergent. `examples/hopper_gemm` is such a GEMM: one
producer and two consumer warpgroups per 128×N tile, a TMA pipeline of
several stages, and pipelined WGMMA.

The newest stage's buffers stay held while the next is issued, so a
pipeline has one stage fewer to load into. That pays when the tensor cores
are the bottleneck and stages are plentiful; a memory-bound pipeline
should release each stage after `wait_mma(c)` instead (with two stages, a
decode projection on an H200 ran 1.6 times slower pipelined).

ptxas keeps WGMMA asynchronous only when it can prove the pipeline safe.
Any function call in the kernel, even on another warp's path (an
out-of-line error helper, a `StepRange`'s overflow check), makes it wait
after every instruction instead, and says so only as a performance
warning. [`audit`](@ref) reports it, and traces the calls to their lines
(see [Machine code](machine.md)).

## tcgen05 and tensor memory (Blackwell)

A [`Tcgen05Atom`](@ref) is one thread's `tcgen05.mma`: A and B are shared
descriptors from `operand` (K-major or MN-major), or A a TMEM tile, and the
accumulator is a [`TmemTile`](@ref), `accumulator(atom, address)`.
`mma(atom, d, a, b, K, accumulate)` issues the K steps and
`commit_mma(atom, barrier)` arrives when they finish. `tmem_allocate!`,
`tmem_deallocate!` and `tmem_relinquish_permit` manage the columns. The
hardware swizzles by address bits, so swizzled operands start on
1024-byte boundaries; dynamic shared memory that follows static shared
variables is not aligned that far on its own.

An atom with M = 256 is a CTA pair's instruction (`.cta_group::2`): the two
CTAs of a cluster whose ranks differ in the last bit each hold 128 rows of A
and half of B's columns at the same shared addresses, one thread of the pair
issues the MMA, and each CTA's TMEM receives its 128 rows of the product.
The pair's loads complete on one barrier with `tma_load!(...; cta = 0)`, and
`commit_mma(atom, barrier, ctas)` arrives in every CTA of the mask. Every
tcgen05 instruction of a kernel names the same CTA group, which is why the
management functions take the atom. `examples/blackwell_gemm` uses both
forms.

### Tensor memory

A [`TmemTile`](@ref) is tensor memory with a layout over its 128 lanes and
typed columns; `window`, `permutedims` and `reinterpret_tile` view it. A
warp moves a 32-lane band between TMEM and registers with the `.32x32b`
instructions, in the ownership where lane `t` holds line `t`:

```julia
rows = distribute((32, 64), 32; vector=64, axis=2)
values = wait_load(load_async(rows, window(tmem, (32warp, 0), (32, 64))))
store_async!(window(tmem, (32warp, 0), (32, 64)), values .* alpha)
wait_stores()
```

## Completion

Each asynchronous mechanism completes through its own verb, because each
waits on a different scope:

| Issued by | Completed by | Scope |
|:--|:--|:--|
| `copy_async!` | `commit_copies()`, `wait_copies(n)` | the thread's cp.async groups |
| `tma_load!` | `wait(barrier, parity)` | the barrier's arrivals and transaction bytes |
| `tma_store!` | `commit_tma_stores()`, `wait_tma_reads(n)`, `wait_tma_stores(n)` | the thread's bulk groups |
| `mma_async` | `wait_mma(pending[, n])` | the warpgroup's WGMMA groups |
| tcgen05 `mma` | `commit_mma(atom, barrier[, ctas])`, then `wait` | the issuing thread's MMAs |
| `copy_async!` | `commit_copies(barrier)`, then `wait` | the thread's cp.async copies, as one arrival |
| `load_async` | `wait_load(pending)` | the thread's TMEM loads |
| `store_async!` | `wait_stores()` | the thread's TMEM stores |

The pending results of `mma_async` and `load_async` are distinct types, so
their registers cannot be used before the wait, which also carries a
compiler dependency through them. Barriers between threads, fences between
proxies and buffer reuse remain the kernel's.
