# TEST_TARGET: cc>=8.0
include("../../examples/softmax/kernel.jl")
using .SoftmaxExample: lane_softmax_kernel!, warp_softmax_kernel!, mma_softmax_kernel!

begin # assembly checks
@testset "Softmax and MMA epilogue assembly" begin
    config=SoftmaxExample.gemm_config(BFloat16;block=(32,24,16),warps=(1,1),stages=1)
    for (name,f,tt,threads) in (
        ("softmax-warp",warp_softmax_kernel!,Tuple{CuDeviceMatrix{Float32,1},CuDeviceMatrix{Float32,1},CuDeviceMatrix{Bool,1},Val{3}},128),
        ("softmax-mma",mma_softmax_kernel!,Tuple{CuDeviceMatrix{Float32,1},CuDeviceMatrix{BFloat16,1},CuDeviceMatrix{BFloat16,1},CuDeviceMatrix{Bool,1},typeof(config)},32))
        code=compile_kernel(f,tt;arch=CUDACore.SMVersion(12,1,:arch),threads)
        save_code(name,code); body=entry_body(code.ptx)
        @test !occursin(".local .",body)
        @test !occursin(r"\bcall",body)
        @test occursin("shfl.sync.bfly",body)
        if name=="softmax-mma"
            @test occursin("mma.sync.aligned.m16n8k16",body)
        end
    end
end
end
if runtime_supported(@__FILE__)
@testset "Masked row softmax" begin
    for T in (Float32,BFloat16,Float16),width in (1,3,17,33,97,129)
        rows=19; rng=MersenneTwister(width)
        x=T.(randn(rng,Float32,width,rows) .* 3f0)
        x[:,1] .= T.([width == 1 ? -80f0 : -80f0+160f0*(j-1)/(width-1) for j in 1:width])
        x[:,4] .+= T(256)
        mask=rand(rng,Float32,width,rows) .> 0.25f0
        mask[:,1] .= true; mask[:,2] .= false
        for r in 5:rows
            mask[min(width,r):end,r] .= false
        end
        dx,dm=CuArray(x),CuArray(mask); output=CuArray{Float32}(undef,width,rows)
        expected=softmax_reference(x,mask)
        for (f,n,blocks) in ((lane_softmax_kernel!,width+3,cld(rows,128)),
                             (warp_softmax_kernel!,cld(width,32),cld(rows,4)))
            @cuda threads=128 blocks=blocks f(output,dx,dm,Val(n))
            y=Array(output)
            @test y ≈ expected rtol=3e-5 atol=3e-7
            @test all(iszero,y[.!mask])
            @test all(isfinite,y)
            @test vec(sum(y;dims=1)) ≈ Float32.(vec(any(mask;dims=1))) atol=3e-6
        end
    end
end
@testset "Softmax over actual tiled MMA results" begin
    for T in (BFloat16,Float16),(m,n,k,wm) in ((16,8,32,1),(32,24,16,1),(64,40,64,2))
        cfg=SoftmaxExample.gemm_config(T;block=(m,n,k),warps=(wm,1),stages=1)
        rng=MersenneTwister(m+n+k)
        a=T.(0.4f0 .* randn(rng,Float32,m,k)); b=T.(0.4f0 .* randn(rng,Float32,k,n))
        mask=rand(rng,Float32,m,n) .> 0.3f0; mask[2,:] .= false
        # Irregular logical row lengths span repeated N atoms.
        for row in 5:m
            mask[row,min(n,row):end] .= false
        end
        da,db,dm=CuArray(permutedims(a)),CuArray(b),CuArray(mask)
        output=CuArray{Float32}(undef,m,n)
        @cuda threads=32wm shmem=SoftmaxExample.shared_bytes(cfg,T) mma_softmax_kernel!(output,da,db,dm,cfg)
        y=Array(output)
        expected=permutedims(softmax_reference(permutedims(Float64.(a)*Float64.(b)),permutedims(mask)))
        @test y ≈ expected rtol=4e-5 atol=5e-7
        @test all(iszero,y[.!mask])
        @test all(isfinite,y)
    end
end
end
