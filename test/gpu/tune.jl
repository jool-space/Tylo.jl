# TEST_TARGET: cc>=8.0
# `tune` rejects what its constructors refuse and what fails the audit, and
# ranks the rest by measured time.
function busy_kernel!(y,x,::Val{R}) where R
    i = threadIdx().x + (blockIdx().x-Int32(1))*blockDim().x
    v = @inbounds x[i]
    for _ in 1:R
        v = muladd(v,1.0001f0,0.5f0)
    end
    @inbounds y[i] = v
    nothing
end
function local_kernel!(y,x,j)
    values = ntuple(i -> @inbounds(x[i]),Val(16))
    @inbounds y[1] = values[j]                         # local memory: rejected by the audit
    nothing
end

if runtime_supported(@__FILE__)
@testset "tune" begin
    x = CUDACore.ones(Float32,1<<16); y = similar(x)
    tuning = tune([1,4096,65536,:local,:refused]) do c
        c === :refused && throw(ArgumentError("not a configuration"))
        if c === :local
            kernel = @cuda launch=false local_kernel!(y,x,Int32(3))
            return kernel, () -> kernel(y,x,Int32(3))
        end
        kernel = @cuda launch=false busy_kernel!(y,x,Val(c))
        kernel, () -> kernel(y,x,Val(c);threads=256,blocks=256)
    end
    @test [t.status for t in tuning.trials] == [:timed,:timed,:timed,:audited,:invalid]
    @test Tylo.best(tuning) == 1                         # the least work
    @test last(filter(t -> t.status === :timed,tuning.trials)).candidate == 65536
    @test issorted([t.median for t in tuning.trials[1:3]])
    @test tuning.trials[4].reason == "local_memory"
    @test occursin("not a configuration",tuning.trials[5].reason)
    shown = sprint(show,MIME("text/plain"),tuning)
    @test occursin("3 of 5 candidates timed",shown) && occursin("audited",shown) && occursin("invalid",shown)
    # Without the audit, and with a bare launcher, everything valid is timed.
    @test count(t -> t.status === :timed,tune(c -> () -> nothing,1:3).trials) == 3
    @test_throws ArgumentError Tylo.best(tune(c -> throw(ArgumentError("no")),1:2))
end
end
