# TEST_TARGET: cc==9.0
# The Hopper FlashAttention forward example: clean for sm_90a, and equal to
# attention in Float64 on Hopper.
include("../../examples/hopper_attention/fp8.jl")   # and kernel.jl

@testset "Hopper attention assembly" begin
    T = BFloat16
    arch = CUDACore.SMVersion(9,0,:arch)
    x = CUDACore.zeros(T,128,256,2)
    for causal in (false,true)
        code = machine_code(hopper_attention_launch(x,x,x;stages=2,persistent=true,causal,sms=132);arch)
        save_code("hopper-attention-persistent$(causal ? "-causal" : "")",code)
        @test isempty(audit(code))
    end
    for (stages,overlap,pingpong) in ((2,false,false),(2,true,false),(2,true,true),(3,true,true))
        code = machine_code(hopper_attention_launch(x,x,x;stages,overlap,pingpong,persistent=false,sms=132);arch)
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
        _, launch, o = hopper_attention(CuArray(q),CuArray(k),CuArray(v);stages,overlap,pingpong,persistent=false)
        launch(); synchronize()
        actual = Float64.(Array(o))
        for h in 1:bh
            s = transpose(Float64.(q[:,:,h]))*Float64.(k[:,:,h]) ./ sqrt(128)
            p = exp.(s .- maximum(s;dims=2)); p ./= sum(p;dims=2)
            expected = transpose(p*transpose(Float64.(v[:,:,h])))       # d×sq
            @test maximum(abs.(actual[:,:,h] .- expected)) <= 0.03
        end
    end
    # Persistent: 2 heads × 3 query blocks, and 9 heads × 16 query blocks,
    # 144 tiles, more than an H100's 132 SMs: CTAs walk two tiles, carrying
    # Q buffers and K/V stages across them.
    for (sq,skv,bh) in ((384,256,2),(384,384,2),(2048,2048,9))
        q, k, v = (BFloat16.(randn(rng,Float32,128,s,bh)) for s in (sq,skv,skv))
        # Two Q buffers and three stages exceed the H100's shared memory.
        @test_throws ArgumentError hopper_attention(CuArray(q),CuArray(k),CuArray(v);stages=3,persistent=true)
        for causal in (false,true)
            causal && sq != skv && continue
            for (dynamic,heads_inner) in (causal ? ((false,false),(true,false),(true,true)) : ((false,false),))
            _, launch, o = hopper_attention(CuArray(q),CuArray(k),CuArray(v);stages=2,persistent=true,causal,dynamic,heads_inner)
            launch(); synchronize()
            actual = Float64.(Array(o))
            for h in 1:bh
                s = transpose(Float64.(q[:,:,h]))*Float64.(k[:,:,h]) ./ sqrt(128)
                causal && (s[[c > r for r in 1:sq, c in 1:skv]] .= -Inf)
                p = exp.(s .- maximum(s;dims=2)); p ./= sum(p;dims=2)
                @test maximum(abs.(actual[:,:,h] .- transpose(p*transpose(Float64.(v[:,:,h]))))) <= 0.03
            end
            end
        end
    end
end
end

if capability_is(v"9.0")
# FP8 inputs, P quantized to FP8 as well: against Float64 attention of the
# FP8-rounded inputs, a few percent relative error (P's 3-bit mantissa).
@testset "FP8 Hopper attention on Hopper" begin
    rng = MersenneTwister(8)
    T = Float8_E4M3FN
    for (s,bh) in ((256,2),(2048,9))
        q, k = (T.(randn(rng,Float32,128,s,bh)) for _ in 1:2)
        v = T.(randn(rng,Float32,128,s,bh))
        vt = permutedims(v,(2,1,3))
        _, launch, o = fp8_attention(CuArray(q),CuArray(k),CuArray(vt))
        launch(); synchronize()
        actual = Float64.(Array(o))
        for h in 1:bh
            sc = transpose(Float64.(Float32.(q[:,:,h])))*Float64.(Float32.(k[:,:,h])) ./ sqrt(128)
            p = exp.(sc .- maximum(sc;dims=2)); p ./= sum(p;dims=2)
            expected = transpose(p*transpose(Float64.(Float32.(v[:,:,h]))))
            @test sqrt(sum(abs2,actual[:,:,h] .- expected)/sum(abs2,expected)) < 0.05
        end
    end
end
end
