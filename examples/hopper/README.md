# TMA → WGMMA GEMM

`kernel.jl` contains a complete one-producer-warp / one-consumer-warpgroup
pipeline. Tylo supplies descriptor preparation, shared layout, WGMMA issue,
register-dependent wait and the epilogue fragment. The kernel owns its
barriers and one/two-stage ring. The main loop uses a K=64 WGMMA plan.

From the Tylo checkout, with a sibling PTX checkout at
`32e36c122bc1c7af5f171cf478324b628b06af3a` or a compatible later revision:

```sh
julia --project=test/gpu -e 'using Pkg; Pkg.instantiate()'
julia --project=test/gpu test/gpu/runtests.jl
```

The suite always assembles the Hopper kernels. On H100/H200 it also checks
BF16/FP16 results, N=8/16/24/64/128/256, one/four partial chains, one/two stages,
K=64/128/320/328, forced GC and changed-input graph replay. WGMMA tests
explicitly skip other architectures. TMA alone also runs on GB10.

```sh
compute-sanitizer --tool memcheck --error-exitcode 1 --num-cuda-barriers 16 \
  julia --project=test/gpu test/gpu/sanitize.jl
```

Repeat with `racecheck` and `synccheck` using a sanitizer compatible with the
active compiler and driver. No H100/H200 execution of this new path has been
recorded yet. See [the contracts](../../docs/src/hopper.md).
