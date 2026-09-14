# Exercise Hopper's register ownership on GB10 without issuing Hopper MMA.
# The matrix input/output is ordinary column-major Julia storage.
function wgmma_fragment_kernel!(out,input,::Val{N},::Val{A}) where {N,A}
    thread = Int32(threadIdx().x)-Int32(1)
    ownership = Tylo.WGMMAOwnership{N}()
    values = Tylo.@rtuple(0:N÷2-1) do e
        i,j = Tylo.Layouts.coordinate(ownership,thread,Val(e))
        @inbounds input[i+Int32(1),j+Int32(1)]
    end
    f = Fragment(values,ownership)
    f = A == 2 ? f : permutedims(f)
    weights = exp.(f .- maximum(f;dims=A))
    result = weights ./ sum(weights;dims=A)
    result = A == 2 ? result : permutedims(result)
    store!(GlobalTile(pointer(out),@Layout((64,N),(1,64))),result,thread)
    nothing
end
@testset "WGMMA values use ordinary fragment operations" begin
    for n in (8,24,64,256), axis in (1,2)
        if !("--runtime-only" in ARGS)
            code = compile_kernel(wgmma_fragment_kernel!,Tuple{CuDeviceMatrix{Float32,1},CuDeviceMatrix{Float32,1},Val{n},Val{axis}};
                                  arch=CUDACore.SMVersion(12,1,:arch))
            save_code("wgmma-fragments-n$n-a$axis",code)
            body = entry_body(code.ptx)
            @test !occursin(".local .",body)
            @test !occursin(r"\bcall",body)
            @test occursin("shfl.sync.bfly",body)
        end
        if CUDACore.functional()
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
        p = WGMMA64(T,Val(n),Val(k))
        if !("--runtime-only" in ARGS)
            code = compile_kernel(wgmma_operand_probe!,Tuple{CuDeviceMatrix{Float32,1},typeof(p),Int32,Val{true}};
                                  arch=CUDACore.SMVersion(9,0,:arch))
            save_code("hopper-softmax-$(T)-n$n-k$k",code)
            body = entry_body(code.ptx)
            @test occursin("wgmma.mma_async",body)
            @test occursin("shfl.sync.bfly",body)
            @test first(findfirst("wgmma.wait_group",body)) < first(findfirst("shfl.sync.bfly",body))
            @test !occursin(r"\bcall",body)
            @test !occursin(".local .",body)
        end
        if CUDACore.functional() && capability(device()) == v"9.0"
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
