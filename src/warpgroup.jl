"""
    release_registers(Val(n))
    acquire_registers(Val(n))

Warpgroup collectives (sm_90a, all 128 threads converged): lower, or
raise, the calling warpgroup's per-thread register limit to `n`, a
multiple of 8 from 24 to 256. Released registers return to the CTA's
pool, from which `acquire_registers` takes them, waiting until enough are
free. A warp-specialized kernel's producer warpgroup, which only issues
copies, releases most of its registers so that the consumers can hold
larger accumulators:

```julia
if warpgroup == 0
    release_registers(Val(40))     # the producer: TMA loads only
    ...
else
    acquire_registers(Val(232))    # the consumers: 64×256 FP32 accumulators
    ...
end
```

With 384 threads, 128 × 40 + 256 × 232 registers fit the 64K register
file. ptxas honors these only when the block size is fixed at compile
time: compile with `@cuda ... minthreads=384` (`.reqntid`).
"""
function release_registers end
@doc (@doc release_registers) function acquire_registers end

"""
    warp_uniform(x)

`x` as the value of lane 0 of the warp, through a shuffle, so that the
compiler knows every lane holds the same value. A branch on a warpgroup
index computed from the thread index is warp-uniform in fact but not to
ptxas, which then treats warpgroup collectives (WGMMA, `setmaxnreg`) in it
as divergent; branch on `warp_uniform(thread ÷ 128)` instead.
"""
function warp_uniform end
