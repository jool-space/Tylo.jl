```@meta
CurrentModule = Tylo
```

# Machine code, tuning and timelines

Tylo's operations promise particular instructions: a vector of the widest
width the layouts allow, one `ldmatrix` per block group, WGMMA that runs
asynchronously. Whether a kernel keeps those promises is decided by the
machine code, after LLVM and ptxas, and Julia can inspect every stage of
it from the REPL. [`machine_code`](@ref) compiles a kernel without
launching it, and [`audit`](@ref) checks the result:

```julia
julia> x = CuArray(rand(Float32, 64)); y = similar(x);

julia> @machine_code roots!(y, x, Int32(3))
MachineCode for sm_121a: _Z6roots_13CuDeviceArrayI7Float32Li1ELi1EES1_5Int32
  36 registers, 1040 SASS instructions
  memory: LD.E.U8 ×72, ST.E.U8 ×72, LDS ×20, ATOM.E.CAS.STRONG.GPU ×2, LDG.E ×1, STG.E ×1
  ✗ calls julia_overflow_case
      from roots.jl:2 (roots!) → range.jl:367 (steprange_last)
      hint: A StepRange (a:step:b) with runtime bounds checks its step and computes
            its last element out of line. Iterate a unit range and scale the index,
            or loop with while.
  ✗ calls julia_throw_complex_domainerror
      from roots.jl:3 (roots!) → math.jl:645 (sqrt)
      hint: sqrt, log or ^ of a value not known to be nonnegative throws a
            DomainError. When it is, use Core.Intrinsics.sqrt_llvm.
  ✗ 0 bytes of local memory declared, 5 local loads and stores
      hint: The exception reporting ptxas inlined keeps a local buffer; it goes
            away with the calls.
```

for

```julia
function roots!(y, x, step)
    for i in threadIdx().x:step:length(x)
        @inbounds y[i] = sqrt(x[i])
    end
end
```

`machine_code(f, types; arch)` does the same for device-side argument
types, without a GPU when `arch` names the target, which is how a test
checks code for hardware it does not have:

```julia
code = machine_code(kernel!, Tuple{CuDeviceMatrix{Float32,1}, WGMMAAtom{BFloat16,64,64,1}};
                    arch=CUDACore.SMVersion(9, 0, :arch), threads=128)
@test isempty(audit(code))
```

A kernel from `@cuda launch=false` has its own: `machine_code(kernel)`. The
kernel does not remember the options it was compiled with (`minthreads`,
`maxregs`), and they change the code, so `machine_code` infers them from
the loaded function and checks that it reproduced it, registers and stack,
or is told them. Either way the code audited is the code that runs.

## What the audit checks

- **Calls.** A kernel should call nothing. Every call is out-of-line code,
  and in a kernel with WGMMA, any call makes ptxas serialize all of it.
  Julia brings calls in for exceptions (a bounds check, a `DivideError`
  from `÷` by a runtime value, a `DomainError` from `sqrt`, an
  `InexactError` from `Int32(x)`) and for Base helpers marked
  `@noinline`, such as a `StepRange`'s. The audit traces each call to the
  kernel's line, the innermost frame outside Base, and the Base frame that
  made it, and says how such calls are usually removed.
- **Local memory.** Values that did not become registers live in DRAM:
  tuples indexed at run time, arrays the compiler could not scalarize.
- **Spills.** ptxas ran out of registers.
- **WGMMA serialization.** ptxas keeps WGMMA asynchronous only when it can
  prove the pipeline safe; otherwise it waits after every instruction and
  says so only as a performance warning. The audit counts
  `WARPGROUP.ARRIVE`s in the SASS against `wgmma.fence`s in the PTX, and
  reports the reason ptxas gave.
- **Other ptxas warnings.** ptxas reports lost performance as information
  and compiles on. The costly one for warp-specialized kernels: given a
  block size but no stated register maximum, ptxas may drop the
  `setmaxnreg`s (C7508), so no role receives the registers it was to get,
  and the one that needed them spills. Launch such kernels with
  `maxregs = 65536 ÷ threads` as well as `minthreads`.

A kernel that does not compile at all is often one that Julia had to box
a variable in: a closure inside the kernel assigns a name the kernel also
assigns, and they share one scope. [`machine_code`](@ref) then names those
variables (from the kernel's lowered code, [`boxed_variables`](@ref))
instead of passing on the compiler's report of invalid IR.

## Looking closer

A [`MachineCode`](@ref) holds the PTX, the cubin, its SASS and ptxas's
report. [`registers`](@ref), [`spill_bytes`](@ref),
[`memory_instructions`](@ref) (which widths loads and stores became),
[`wgmma_groups`](@ref) and [`kernel_calls`](@ref) read them; `code.sass`
and `code.ptx` are there to search. Tylo's own tests audit every kernel
they compile, and check the instructions each plan promises this way.

## Tuning

Tile sizes, warp layouts and stage counts are types, so every configuration
of a kernel is its own specialization, compiled on demand. [`tune`](@ref)
searches them: it builds each candidate, rejects those whose constructors
refuse them (an `ArgumentError`) or whose machine code fails the audit,
times the rest on the device, and ranks them. `examples/gemm/tune.jl` tunes
the tiled GEMM:

```julia
tuning = tune(candidates) do c
    config = gemm_config(BFloat16; c...)
    kernel = @cuda launch=false tiled_gemm_kernel!(args..., config)
    kernel, () -> kernel(args..., config; threads, blocks, shmem)
end
```

```
Tuning: 48 of 48 candidates timed
      265.5 μs          124 regs  (block = (64, 128, 32), warps = (2, 2), stages = 2, swizzled = true)
     266.18 μs    1.0×  128 regs  (block = (128, 64, 32), warps = (2, 2), stages = 2, swizzled = true)
     ...
      313.38 μs   1.18×   72 regs  (block = (64, 64, 32), warps = (2, 2), stages = 2, swizzled = true)
     ...
```

On a GB10 at 2048³ the fastest of 48 configurations is 18% faster than the
example's default, `(64, 64, 32)` with 2×2 warps.

## Timelines

The machine code says what a kernel does; a [`Timeline`](@ref) says when.
In a warp-specialized kernel the roles wait on each other through barriers,
and which of them waits, how long and for what decides its speed more than
any single instruction. [`record!`](@ref) stores the SM's clock with an
event number from one thread; the host reads the records back as
[`events`](@ref) and measures between them with [`spans`](@ref):

```julia
function kernel!(..., timeline)
    ...
    lane == 0 && record!(timeline, warp, 2step, 1)      # waiting for S
    wait(sfull, phase)
    lane == 0 && record!(timeline, warp, 2step+1, 2)    # S is ready
    ...
end

timeline = Timeline(CUDACore.zeros(UInt64, 1024, 16, blocks))
@cuda kernel!(..., timeline); synchronize()
median(spans(timeline, 1 => 2))                          # cycles spent waiting for S
```

A record costs a clock read and a global store, so a kernel takes the
timeline as an argument that is `nothing` when it does not record: the
calls fold away, and the machine code is that of the plain kernel.
`examples/blackwell_attention` records every role's steps this way.
