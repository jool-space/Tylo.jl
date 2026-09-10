# Row softmax and an MMA epilogue

Run from the Tylo root with the GPU development environment:

```sh
julia --project=test/gpu examples/softmax/run.jl /tmp/tylo-softmax-results
```

The output directory must be new. The benchmark reports warm CUDA-event timings,
registers, and local memory for a simple scalar warp kernel and Tylo's lane-local
and warp-striped kernels. Each graph contains 32 kernel invocations; compilation
and host/device transfers are excluded. Inputs, storage, and masks are shared.

`kernel.jl` also contains `mma_softmax_kernel!`: asynchronous shared copies,
warp MMA, then row softmax directly on its FP32 accumulators, including repeated
N atoms. The GPU tests exercise BF16 and FP16 inputs. It normalizes one complete
output tile and requires a single warp along N; it is not full attention.

Standalone matrices have physical shape `(columns, rows)`. The MMA example
keeps mathematical A(M,K), B(K,N), output(M,N), with A physically stored as (K,M)
to make K contiguous. Masks have the same shape as each output. Valid logits
must be finite; masked entries and fully masked rows produce zero.

See [the row API](../../docs/src/rows.md) for ownership and participation rules.

## Rows wider than a fixed register tile

`streaming.jl` keeps four values per lane regardless of runtime width. Its first
pass updates `SoftmaxState`; its second pass rereads inputs and emits normalized
probabilities. It supports the same masks and FP32/BF16/FP16 storage conversion.

```sh
julia --project=test/gpu examples/softmax/compare_streaming.jl /tmp/tylo-streaming-softmax-results
```

This interleaves the full-row register kernel, fixed-capacity streaming kernel,
and scalar three-pass baseline on the same inputs. Small rows can favor the
full-row kernel; wide rows expose its growing register use and eventual spills.
The streaming kernel's extra reads and per-chunk reductions remain visible costs.
