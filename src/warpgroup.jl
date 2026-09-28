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
file. ptxas applies these limits only when both the block size and the
register maximum are fixed at compile time: compile with
`@cuda ... minthreads=384 maxregs=168` (65536 ÷ threads, rounded down to
a multiple of 8). Given the thread count alone, ptxas may ignore the
instructions and say so only in a performance warning (C7508, which
[`audit`](@ref) reports); every warp then keeps the launch's budget, and
the roles meant to receive more registers spill.
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

"""
    elect_one() -> Bool

True in one lane of the warp, the same one each time; every lane of the
warp must call it (`elect.sync`). A role that issues single-thread
instructions runs on the whole warp and issues from `elect_one()`: its loop
counters and operands are then warp-uniform, and ptxas keeps them in
uniform registers. Run on one thread (`thread == k`), the same role is
divergent code, and ptxas moves each operand into uniform registers anew,
in a loop, at every instruction. (tcgen05's [`mma`](@ref) and
[`commit_mma`](@ref) elect their lane themselves: the whole warp calls
them.)
"""
function elect_one end
