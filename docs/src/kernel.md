# A complete kernel

The [GEMM example](https://github.com/jool-space/Tylo.jl/tree/main/examples/gemm)
multiplies BF16 or FP16 matrices with warp MMA, a pipeline of `cp.async`
copies through swizzled shared memory, and an epilogue that handles ragged
edges. It uses every part of the vocabulary once, so it's worth reading
alongside this page.

## On the host: the plan

Everything static is decided before launch and passed to the kernel as
constants in its argument types:

```julia
plan = TiledMMA(MMAAtom((16,8,16), T), warps, (bm÷(16warps[1]), bn÷(8warps[2])), bk)
sa = compose(Swizzle{bits,3,bits}(), @Layout((bm, bk), (bk, 1)))   # shared A, K contiguous
sb = compose(Swizzle{bits,3,bits}(), @Layout((bk, bn), (1, bk)))   # shared B, K contiguous
ac = distribute((bm, bk), Tylo.threads(plan); vector=16÷sizeof(T), axis=2)
bc = distribute((bk, bn), Tylo.threads(plan); vector=16÷sizeof(T), axis=1)
```

The `TiledMMA` says which warp computes which atoms; the swizzled shared
layouts avoid bank conflicts; `ac` and `bc` deal 16-byte vectors along K to
the block's threads, consecutive threads on consecutive vectors, so the
global reads coalesce.

## In the kernel: tiles

The global matrices are tiles with runtime extents and leading dimensions,
static unit strides, and a declared alignment:

```julia
a = GlobalTile(pointer(a_data), @Layout(($m, $k), ($lda, 1)), 16)
```

A full-tile launch promises 16-byte rows (the host checks the leading
dimensions). A ragged launch checks once, in the kernel, and takes either
this path or one with element alignment, where the copies fall back to
scalars. Both paths are straight-line code; neither checks alignment per
vector.

## The pipeline

Each stage copies a block of A and B into shared memory:

```julia
copy_async!(config.ac, sa, window(a, (m, k), (bm, bk)), tid)                      # interior
copy_async!(config.ac, sa, tile(bound(a, size(a)), (bm, bk), (m÷bm, k÷bk)), tid)  # edge
commit_copies()
```

An edge stage copies from a bounded tile: what lies past the matrix arrives
as zeros, a ragged K cuts a vector where it ends, and the multiply that
follows needs no special case. The main loop waits for a stage, publishes it
with a barrier, multiplies, and waits for every warp to finish reading
before the stage is overwritten:

```julia
wait_copies(1)
sync_threads()                       # the copies are visible to every warp
acc = mma(config.plan, sa, sb, acc, tid)
sync_threads()                       # every warp is done with this stage
```

The two barriers have different jobs, and Tylo inserts neither.

## The epilogue

The accumulator is a fragment of the block tile, so arithmetic on it is
broadcasting and storing it is `store!`:

```julia
result = map(x -> max(alpha*x, 0f0), acc)
store!(window(d, (row, col), (bm, bn)), result, tid)                                        # interior
store!(tile(bound(d, size(d)), (bm, bn), (row÷bm, col÷bn)), Fragment{eltype(d)}(result), tid)  # edge
```

The edge store writes only the elements inside the matrix, converted to
the output type.

## Other examples

- [Softmax](https://github.com/jool-space/Tylo.jl/tree/main/examples/softmax):
  lane-local, warp-striped and MMA-accumulator reductions with masking.
- [Streaming attention](https://github.com/jool-space/Tylo.jl/tree/main/examples/streaming_attention):
  a complete D=64 forward kernel, with TMA-fed stages, online softmax
  statistics and two warp groups alternating on the tensor pipe.
- [Hopper GEMM](https://github.com/jool-space/Tylo.jl/tree/main/examples/hopper):
  a TMA and WGMMA producer/consumer pipeline.
- [Datacenter FlashAttention](https://github.com/jool-space/Tylo.jl/tree/main/examples/flash_attention):
  tcgen05 and TMEM operations replacing parts of a raw PTX kernel.
- [Megakernels](https://github.com/jool-space/Megakernels.jl)' MXFP8
  streamed projection, which compiles to the same machine code as the same
  algorithm written by hand.
