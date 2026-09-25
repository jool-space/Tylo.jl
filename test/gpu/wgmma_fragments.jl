# TEST_TARGET: cc>=8.0
include("wgmma_defs.jl")
# Exercise Hopper's register ownership on GB10 without issuing Hopper MMA.
# The matrix input/output is ordinary column-major Julia storage.
function wgmma_fragment_kernel!(out,input,::Val{N},::Val{A}) where {N,A}
    thread = Int32(threadIdx().x)-Int32(1)
    ownership = operand_layout(WGMMAAtom(BFloat16,N),Accumulator())
    values = Tylo.@rtuple(0:N÷2-1) do e
        i,j = Tylo.Layouts.coordinate(ownership,thread,e)
        @inbounds input[i+Int32(1),j+Int32(1)]
    end
    f = Fragment(values,ownership)
    f = A == 2 ? f : permutedims(f)
    weights = exp.(f .- maximum(f;dims=A))
    result = weights ./ sum(weights;dims=A)
    result = A == 2 ? result : permutedims(result)
    @inbounds store!(GlobalTile(pointer(out),@Layout((64,N),(1,64))),result,thread)
    nothing
end
@testset "WGMMA values use ordinary fragment operations" begin
    for n in (8,24,64,256), axis in (1,2)
        begin # assembly checks
            code = compile_kernel(wgmma_fragment_kernel!,Tuple{CuDeviceMatrix{Float32,1},CuDeviceMatrix{Float32,1},Val{n},Val{axis}};
                                  arch=CUDACore.SMVersion(12,1,:arch))
            save_code("wgmma-fragments-n$n-a$axis",code)
            body = entry_body(code.ptx)
            @test isempty(audit(code))
            @test occursin("shfl.sync.bfly",body)
        end
        if runtime_supported(@__FILE__)
            input = randn(MersenneTwister(1053),Float32,64,n)
            input[1,:] .= -17f0
            output = CuArray{Float32}(undef,64,n)
            @cuda threads=128 wgmma_fragment_kernel!(output,CuArray(input),Val(n),Val(axis))
            reference = exp.(Float64.(input) .- maximum(Float64.(input);dims=2))
            reference ./= sum(reference;dims=2)
            @test Array(output) ≈ reference rtol=2e-6 atol=2e-7
        end
    end
end

@testset "Completed WGMMA and fused softmax epilogue" begin
    for T in (BFloat16,Float16), n in (8,24), k in (16,64)
        p = WGMMAAtom(T,n,k)
        begin # assembly checks
            code = compile_kernel(wgmma_operand_probe!,Tuple{CuDeviceMatrix{Float32,1},typeof(p),Int32,Val{true}};
                                  arch=CUDACore.SMVersion(9,0,:arch))
            save_code("hopper-softmax-$(T)-n$n-k$k",code)
            body = entry_body(code.ptx)
            @test occursin("wgmma.mma_async",body)
            @test occursin("shfl.sync.bfly",body)
            @test first(findfirst("wgmma.wait_group",body)) < first(findfirst("shfl.sync.bfly",body))
            @test isempty(audit(code))
        end
        if capability_is(v"9.0")
            out = CuArray{Float32}(undef,64,n)
            @cuda threads=128 shmem=33792 arch=CUDACore.SMVersion(9,0,:arch) wgmma_operand_probe!(out,p,Int32(0),Val(true))
            a = Float64[(r+3kk)%19-9 for r in 64:127, kk in 0:k-1]
            b = Float64[(2c+kk)%17-8 for kk in 0:k-1,c in 8:8+n-1]
            scores = a*b/128
            ref = exp.(scores .- maximum(scores;dims=2))
            ref ./= sum(ref;dims=2)
            @test Array(out) ≈ ref rtol=3e-6 atol=2e-7
        end
    end
end

# FP32 scores to an FP8 atom's register A (PTX ISA Figure 155), by the quad
# exchange: every A slot holds the score at its coordinate. Two patterns of
# exactly representable values, so that no misplacement can coincide.
function fp8_operand_probe!(out,::Val{K},::Val{Pattern}) where {K,Pattern}
    t = Int32(threadIdx().x)-Int32(1)
    atom = WGMMAAtom(Float8_E4M3FN,64,K)
    scores = zero_accumulator(WGMMAAtom(BFloat16,K,16))
    at = @inbounds coordinates(scores,t)
    value(x) = Pattern == 1 ? Float32((x[1]*Int32(5)+x[2]*Int32(3))%Int32(13)-Int32(6)) :
                              Float32((x[1]+x[2]*Int32(7))%Int32(11)-Int32(5))
    scores = map(value,at)
    a = pack_operand_a(atom,scores,t)
    Tylo.@rtuple(1:K÷2) do e
        @inbounds out[e,t+Int32(1)] = reinterpret(UInt8,a.data[e])
    end
    nothing
end
@testset "FP8 register A from FP32 scores" begin
    for K in (32,64,128), pattern in (1,2)
        code = compile_kernel(fp8_operand_probe!,Tuple{CuDeviceMatrix{UInt8,1},Val{K},Val{pattern}};
                              arch=CUDACore.SMVersion(9,0,:arch))
        @test isempty(audit(code))
        @test count("shfl.sync.idx",entry_body(code.ptx)) == 4*2*(K÷32)
        if runtime_supported(@__FILE__) && capability_at_least(v"8.9")
            out = CuArray{UInt8}(undef,K÷2,128)
            @cuda threads=128 fp8_operand_probe!(out,Val(K),Val(pattern))
            got = Array(out)
            o = operand_layout(WGMMAAtom(Float8_E4M3FN,64,K),OperandA())
            value(r,c) = reinterpret(UInt8,Float8_E4M3FN(pattern == 1 ? Float32((r*5+c*3)%13-6) : Float32((r+c*7)%11-5)))
            @test all(got[e+1,t+1] == value(Tylo.Layouts.coordinate(o,t,e)...) for t in 0:127, e in 0:K÷2-1)
        end
    end
    @test_throws ArgumentError pack_operand_a(WGMMAAtom(BFloat16,64,64),zero_accumulator(WGMMAAtom(BFloat16,64,16)),0)
end
