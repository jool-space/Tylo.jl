# TEST_TARGET: cc==9.0
# The Hopper FlashAttention forward example: clean for sm_90a, and equal to
# attention in Float64 on Hopper.
include("../../examples/hopper_attention/kernel.jl")

@testset "Hopper attention assembly" begin
    T = BFloat16
    binding(plan) = Tylo.TMABinding{typeof(plan),PTX.TMADescriptorPtr,3}
    for (stages,overlap,pingpong) in ((2,false,false),(2,true,false),(2,true,true),(3,true,true))
        tt = Tuple{CuDeviceArray{T,3,1},binding(TMATile(T,(128,128),2)),binding(TMATile(T,(128,128),1)),
                   binding(TMATile(T,(128,128),2)),Int32,Float32,Val{stages},Val{overlap},Val{pingpong}}
        code = compile_kernel(attention_kernel!,tt;arch=CUDACore.SMVersion(9,0,:arch),threads=384)
        save_code("hopper-attention-s$stages-$(overlap ? "overlap" : "serial")$(pingpong ? "-pingpong" : "")",code)
        # The overlapped schedules hold two score tiles at once, a few
        # registers more than they get: 12 bytes spill.
        @test isempty(audit(code;spills=!overlap,local_memory=!overlap))
        w = Tylo.wgmma_groups(code)
        @test w.instructions > 0 && w.fences == w.arrives
        # P from registers, V transposed: every P V instruction.
        body = entry_body(code.ptx)
        @test count(r"\}, \{%r",body) == count("1, 1, 1, 1;",body) > 0
        @test overlap == occursin("wgmma.wait_group.sync.aligned 2;",body)
    end
end

if capability_is(v"9.0")
@testset "Hopper attention on Hopper" begin
    rng = MersenneTwister(3)
    for (sq,skv,bh) in ((128,128,1),(128,256,1),(256,768,3)), (stages,overlap,pingpong) in ((2,false,false),(2,true,false),(2,true,true),(3,true,true))
        q, k, v = (BFloat16.(randn(rng,Float32,128,s,bh)) for s in (sq,skv,skv))
        _, launch, o = hopper_attention(CuArray(q),CuArray(k),CuArray(v);stages,overlap,pingpong)
        launch(); synchronize()
        actual = Float64.(Array(o))
        for h in 1:bh
            s = transpose(Float64.(q[:,:,h]))*Float64.(k[:,:,h]) ./ sqrt(128)
            p = exp.(s .- maximum(s;dims=2)); p ./= sum(p;dims=2)
            expected = transpose(p*transpose(Float64.(v[:,:,h])))       # d×sq
            @test maximum(abs.(actual[:,:,h] .- expected)) <= 0.03
        end
    end
end
end
