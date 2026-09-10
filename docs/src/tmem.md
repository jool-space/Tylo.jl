```@meta
CurrentModule = Tylo
DocTestSetup = :(using Tylo; using Tylo: BFloat16; using Tylo.Layouts: @Layout)
```

# TMEM tiles and transfers

A TMEM tile describes **storage**, a fragment describes **values and ownership**,
and a transfer connects them. Logical rows are not a requirement of any of these
abstractions. The present transfer instruction has a particular hardware mapping,
which Tylo checks when binding it to storage.

## Storage and a transfer

```jldoctest
julia> using Tylo.Layouts: @Layout

julia> tile = TmemTile(Float32, UInt32(0), @Layout((128, 128), (1, 128)));

julia> view = window(tile, (64, 32), Val((32, 64)));

julia> transfer = TmemTransfer{(32, 64), 2}();

julia> access = partition(transfer, view);

julia> access.address == UInt32((64 << 16) + 32)
true

julia> swapped = partition(permutedims(transfer), permutedims(view));

julia> size(swapped), swapped.address == access.address
((64, 32), true)
```

`TmemTransfer{(32,64),2}()` assigns 64 values along logical axis 2 to each
participating thread. The other logical axis identifies the 32 threads. Its
permutation assigns those same values along axis 1. Both partitions issue the
same physical `.32x32b.x64` transfer: the permutation only changes coordinates.

All 32 threads must execute with the same partition. The example selects
physical TMEM lanes 64–95, so the **third warp in its warpgroup** must execute
the transfer. A layout permutation does not change this requirement.

In a kernel, the access is used as follows:

```julia
f = wait_load(load_async(access))
g = f .- maximum(f; dims=2)
store_async!(access, g)
wait_stores()
```

For `swapped`, the reduction uses `dims=1`. Each reduction here is local to a
thread, because all values along the chosen axis belong to that thread. Other
fragment ownerships can require warp collectives. See
[Register fragments and logical axes](@ref).

## Storage units

TMEM addresses encode a physical lane in bits 31:16 and a 32-bit word column
in bits 15:0. Hardware provides 128 lanes and 512 word columns. These are ISA
storage coordinates, not the logical dimensions of a matrix.

Tylo's TMEM layout offsets count typed slots in a virtual column-major grid
with 128 physical lanes. For layout offset `i`:

- The physical lane offset is `i % 128`.
- The typed position along that lane is `i ÷ 128`.
- One FP32 value occupies a word; two BF16 values share a word, low half first.

Thus strides `(1,128)` map logical axis 1 along hardware lanes and axis 2 along
values within a lane. Strides `(128,1)` exchange those logical roles.
The address supplied to `TmemTile` is the allocation-relative base in ISA units,
not a byte pointer. `Tylo.tmem_location(tile, coordinate)` exposes the resulting
word address and bit offset for inspection. Coordinates are zero-based, as in
Tylo's other low-level layout operations.

A tile may describe layouts beyond the available instructions. `partition`
currently accepts affine modes with these contiguous storage positions, a
32-lane aligned band, and 16, 32, or 64 words per thread. Compatible hierarchical
modes are accepted; nonlinear swizzles and incompatible strides are rejected.
It does not silently select another instruction or redistribute values.

## Representation and windows

```jldoctest
julia> tile = TmemTile(Float32, UInt32(0), @Layout((32, 64), (1, 128)));

julia> bf = reinterpret_tile(BFloat16, tile; dims=2);

julia> Tuple(Int.(size(bf)))
(32, 128)

julia> rotated = reinterpret_tile(BFloat16, permutedims(tile); dims=1);

julia> Tuple(Int.(size(rotated)))
(128, 32)
```

Reinterpretation changes the element representation of the same storage. It
requires an explicit logical axis, a flat affine view, and complete aligned
pairs when converting a BF16 view back to FP32. It does not convert values or
make data ready. By contrast, `pack_bf16(f)` numerically converts FP32 registers
and retains their ownership.

A register slice uses `window(f, Val(origin), Val(shape))`. The current
implementation can slice the local-value axis of lane-local and TMEM-transfer
fragments in either orientation; it must retain every participating thread.
Packed windows require complete BF16 pairs. Both origin and shape are static,
so slicing selects registers without dynamic tuple indexing. Storage windows
use the same logical coordinates, with runtime origins permitted.

`store!(destination, packed)` writes this thread's packed payload contiguously
to a global pointer. It requires 16-byte alignment and a multiple of eight BF16
values. The caller computes a distinct destination for each thread; the store
does not infer a global matrix layout from fragment ownership.

## Completion and validation

The caller owns allocation, bounds within that allocation, lifetime, and reuse.
View and hardware bounds are checked; `@inbounds` can omit those checks after
the caller establishes the contract. Julia's types do not prove allocation
capacity, collective participation, or that the executing warp matches the
selected physical band.

`wait_load` completes prior loads and carries the loaded registers through a
compiler dependency. `wait_stores` completes prior stores. Thread synchronization
and `fence_before_thread_sync` / `fence_after_thread_sync` remain explicit.
Permuting or slicing metadata performs none of these operations.

Host tests and GB10 tests cover address mapping, both logical orientations,
nested layouts, runtime storage offsets, register arithmetic, and packed global
stores. SM100 assembly tests cover actual TMEM transfers in both orientations.
Executing those transfers and the complete datacenter FlashAttention kernel
still requires B200/B300; GB10 cannot validate that hardware path.
