# FP32 fragment arithmetic on pairs: packed instructions where the target
# has them (sm_100: FFMA2, FADD2, FMUL2), the scalar ones elsewhere, the same
# values either way.
@testset "FP32 pairs" begin
    p = Tylo.F32x2(1.5f0,-2f0)
    @test Tylo._lanes(p+1f0) == (2.5f0,-1f0) && Tylo._lanes(p*p) == (2.25f0,4f0)
    @test Tylo._lanes(muladd(p,2f0,Tylo.F32x2(1f0,1f0))) == (4f0,-3f0) && Tylo._lanes(-p) == (-1.5f0,2f0)
    rows = distribute((32,32),32;vector=32,axis=2)
    x = Fragment(ntuple(i -> Float32(i)/7,Val(32)),rows)
    y = Tylo._ex2.(muladd.(x,1.25f0,-0.5f0))
    @test all(Tylo._values(y) .≈ exp2.(muladd.(Tylo._values(x),1.25f0,-0.5f0)))
    @test sum(x;dims=2).data[1] ≈ sum(Tylo._values(x))
end

function pair_scale!(out,x,a::Float32,b::Float32)
    rows = distribute((32,32),32;vector=32,axis=2)
    lane = threadIdx().x-1
    v = Fragment(ntuple(i -> @inbounds(x[32lane+i]),Val(32)),rows)
    w = muladd.(v,a,b) .* a
    @inbounds out[lane+1] = sum(w;dims=2).data[1]
    nothing
end
@testset "FP32 pairs in the machine code" begin
    tt = Tuple{CuDeviceVector{Float32,1},CuDeviceVector{Float32,1},Float32,Float32}
    blackwell = compile_kernel(pair_scale!,tt;arch=CUDACore.SMVersion(10,0,:arch),threads=32)
    hopper = compile_kernel(pair_scale!,tt;arch=CUDACore.SMVersion(9,0,:arch),threads=32)
    @test isempty(audit(blackwell)) && isempty(audit(hopper))
    count_(r,code) = count(r,Tylo.kernel_sass(code))
    # All packed on sm_100 but the final sum of the two lanes; LLVM may fuse
    # the multiplications into the reduction's additions.
    @test count_(r"\bF(FMA|MUL|ADD)2\b",blackwell) >= 32 && count_(r"\bF(FMA|MUL)\b",blackwell) == 0 && count_(r"\bFADD\b",blackwell) <= 1
    @test count_(r"\bF(FMA|MUL|ADD)2\b",hopper) == 0 && count_(r"\bFFMA\b",hopper) >= 32
end
