"""
    tune(prepare, candidates; samples=15, audit=true) -> Tuning

Find the fastest of `candidates` on the current device. `prepare(c)` builds
candidate `c` and returns `(kernel, launch)`: the kernel compiled with
`@cuda launch=false`, and a function of no arguments that launches it once
(or only `launch`, which skips the audit).

```julia
tuning = tune([(; block, warps, stages) for block in ((64,64,32), (128,64,32)),
                                          warps in ((2,2), (4,2)), stages in (1,2)]) do c
    config = gemm_config(BFloat16; c...)                 # may throw ArgumentError
    kernel = @cuda launch=false gemm!(out, a, b, config)
    kernel, () -> kernel(out, a, b, config; threads=threads(config), blocks, shmem=shared_bytes(config))
end
best(tuning)
```

A candidate is rejected, never timed, when `prepare` throws an
`ArgumentError` (a configuration its constructors refuse), or when
[`audit`](@ref) finds something in its machine code: spills, local memory,
calls, serialized WGMMA. `audit` may be `false`, or a `NamedTuple` of
[`audit`](@ref)'s keywords, such as `(; local_memory=false)`. The others
are timed with CUDA events, `samples` times each, in rotating order, each
sample repeating the launch long enough to measure (about 0.2 ms), and
ranked by median time. Whatever `launch` computes should be checked
separately: tuning measures, it does not verify.
"""
function tune end

"""
    Trial

One candidate of a [`tune`](@ref): its `status` is `:timed`, `:invalid`
(its construction threw an `ArgumentError`) or `:audited` (its machine
code failed the audit), with the `reason`; timed ones carry the median and
minimum `seconds` per launch and their `registers`.
"""
struct Trial
    candidate::Any
    status::Symbol
    reason::String
    median::Float64
    minimum::Float64
    registers::Int
end

"""
    Tuning

The outcome of [`tune`](@ref): its `trials`, the timed ones fastest first,
then the rejected ones. [`best`](@ref) is the fastest candidate.
"""
struct Tuning
    trials::Vector{Trial}
end

"The fastest candidate of a [`Tuning`](@ref)."
function best(t::Tuning)
    (isempty(t.trials) || first(t.trials).status !== :timed) && throw(ArgumentError("no candidate was timed"))
    first(t.trials).candidate
end

function _ranked(trials)
    timed = sort!(filter(t -> t.status === :timed,trials);by=t -> t.median)
    [timed; filter(t -> t.status !== :timed,trials)]
end

function Base.show(io::IO,::MIME"text/plain",t::Tuning)
    timed = count(x -> x.status === :timed,t.trials)
    print(io,"Tuning: ",timed," of ",length(t.trials)," candidates timed")
    fastest = timed > 0 ? first(t.trials).median : NaN
    for x in t.trials
        print(io,"\n  ")
        if x.status === :timed
            print(io,lpad(string(round(1e6x.median;digits=2)),9)," μs  ",
                  x.median == fastest ? "      " : lpad(string(round(x.median/fastest;digits=2))*"×",6),
                  "  ",lpad(x.registers,3)," regs  ",x.candidate)
        else
            print(io,"   ",rpad(x.status === :invalid ? "invalid" : "audited",9),"  ",x.candidate,": ",x.reason)
        end
    end
end
