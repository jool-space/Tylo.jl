# FlashAttention integration experiment

`tiles.jl` replaces the existing kernel's correction and epilogue. Everything
else—the softmax, QK/PV issue order, register budgets, barrier plan, and
persistent work loop—comes from a fixed reference.

The comparison loads two isolated modules from
`PTX/test/gpu/blackwell/flash_attention_defs.jl`. Only the two helper
definitions are replaced in the Tylo module. Neither source checkout is
rewritten and PTX does not acquire a dependency on Tylo.

Reference:
- PTX commit: `32e36c122bc1c7af5f171cf478324b628b06af3a`
- File SHA256: `d4bcc34234bf2a9d85d9fed136f15e035d28dc84123f46d0651958745f132cdc`

The reference update adopts upstream register-dependent TMEM waits in the
softmax, correction and epilogue; the six paired machine-code comparisons
remain required.

The digest is checked before compilation. A mismatch requires reviewing the
reference update. `TYLO_PTX_ROOT` can point to another checkout containing
that exact file; the default is the loaded PTX package's source directory.

## What the comparison checks

The default and quarter-publication variants assemble for SM100a, SM103a,
and SM100f. Checks compare PTX operation counts for TMEM load/store widths,
waits, fences, MMA, barrier arrivals/initialization, and global vector stores.
The kernel entry must not materialize its register tuples as local arrays.
The complete kernel machine-code section must also match byte for byte.
Debug/source metadata outside the executable section may differ.

On B200/B300 the suite also compares both implementations bit for bit and
against the reference's CPU attention calculation. Cases include strong
inputs that exercise correction, plus a single-CTA grid forcing repeated
work items through the same allocation. A separate TMEM round trip checks
FP32 storage, BF16 aliasing, packing, and raw PTX readback.

The `--bench` option adds paired CUDA-graph measurements: 32 launches per
graph, 21 measured pairs after warmup, alternating execution order.
Compilation, allocation, and CPU reference work are outside the timings.
These modest correctness shapes establish a baseline; they are not a
saturated throughput benchmark.

## Run

From the Tylo checkout, after instantiating `test/gpu`:

```sh
TYLO_EVIDENCE=/tmp/tylo-evidence \
  TYLO_PTX_ROOT=/path/to/pinned-PTX julia --project=test test/runtests.jl gpu/flash_attention

# B200/B300 only:
TYLO_EVIDENCE=/tmp/tylo-blackwell \
  TYLO_PTX_ROOT=/path/to/pinned-PTX julia --project=test test/runtests.jl gpu/flash_attention --bench
```

Evidence output contains PTX and cubins for inspection. The resource script
below also assembles the PTX verbosely and records machine-code statistics:

```sh
julia --project=test test/tools/resources.jl /tmp/tylo-evidence
```

For cloud testing, wrap each invocation with a process timeout; a broken
asynchronous kernel can otherwise wait indefinitely. A matching Compute
Sanitizer can run `test/runtests.jl gpu/flash_attention` on B200/B300; the local
GB10 sanitizer run covers register arithmetic and global stores only.
