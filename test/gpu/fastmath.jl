# TEST_TARGET: cc>=10.0
# Approximate exponentials: MUFU, and pairs emulated on the FMA pipe with
# packed FP32 instructions (sm_100 and later), against exp2.
function exp2_kernel!(out,x,::Val{E}) where E
    t = Int32(threadIdx().x)-Int32(1)
    o = distribute((32,32),32;vector=32,axis=2)
    f = Fragment(ntuple(e -> @inbounds(x[32t+e]),Val(32)),o)
    y = exp2_approx(f,E)
    for e in 1:32
        @inbounds out[32t+e] = y.data[e]
    end
    nothing
end

@testset "Emulated exponentials assembly" begin
    tt = Tuple{CuDeviceVector{Float32,1},CuDeviceVector{Float32,1},Val{(0,3,15)}}
    code = compile_kernel(exp2_kernel!,tt;arch=CUDACore.SMVersion(10,0,:arch),threads=32)
    body = entry_body(code.ptx)
    @test count("ex2.approx.ftz.f32",body) == 32-6
    @test count("fma.rn.ftz.f32x2",body) == 3*3 && count("add.rm.ftz.f32x2",body) == 3
    @test isempty(audit(code))
end

if runtime_supported(@__FILE__)
@testset "Emulated exponentials" begin
    x = Float32.(range(-30,30;length=32*32))
    out = CUDACore.zeros(Float32,length(x))
    @cuda threads=32 exp2_kernel!(out,CuArray(x),Val((0,3,15)))
    y = Array(out)
    @test maximum(abs.(y ./ exp2.(x) .- 1)) < 2e-4
end
end
