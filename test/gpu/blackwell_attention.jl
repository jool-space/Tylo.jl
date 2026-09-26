# TEST_TARGET: cc==10
# The Blackwell FlashAttention forward example, written in Tylo throughout:
# warp-specialized roles with register reallocation, S, P and O in TMEM, P
# published in two parts, in one CTA or CTA pairs. Compiled clean for
# sm_100a, correct on B200/B300.
include("../../examples/blackwell_attention/kernel.jl")

@testset "Blackwell attention assembly" begin
    arch = CUDACore.SMVersion(10,0,:arch)
    for ctas in (1,2)
        tq = Tylo.TMABinding{typeof(TMATile(BFloat16,(128,128),2)),PTX.TMADescriptorPtr,3}
        tk = Tylo.TMABinding{typeof(TMATile(BFloat16,(128,128÷ctas),1)),PTX.TMADescriptorPtr,3}
        tv = Tylo.TMABinding{typeof(TMATile(BFloat16,(128,128÷ctas),2)),PTX.TMADescriptorPtr,3}
        tt = Tuple{CuDeviceArray{BFloat16,3,1},tq,tk,tv,CuDeviceVector{Int32,1},Int32,Int32,Float32,Val{4ctas},Val{ctas},Val{3},Val{false},Val{false},Val{()}}
        code = Tylo.machine_code(blackwell_attention_kernel!,tt;arch,threads=512,maxregs=128)
        save_code("blackwell-attention-ctas$ctas",code)
        # No spills anywhere: the softmax warps hold a row at 192 registers,
        # the MMA warp its operands at 64, because each product derives its
        # descriptors at the issue.
        @test isempty(audit(code))
        @test count("USETMAXREG",Tylo.kernel_sass(code)) == 3
        body = entry_body(code.ptx)
        @test count("tcgen05.ld.sync.aligned.32x32b.x32",body) >= 4
        @test count("tcgen05.st.sync.aligned.32x32b.x16",body) == 4
        # Every tcgen05 instruction in one CTA group.
        @test count("tcgen05.mma.cta_group::$ctas",body) == count("tcgen05.mma",body) > 0
        @test count("tcgen05.commit.cta_group::$ctas",body) == count("tcgen05.commit",body) > 0
        if ctas == 2
            @test count("multicast::cluster",body) == count("tcgen05.commit",body)
            @test occursin("cp.async.bulk.tensor.3d.cta_group::2",body) && occursin("mbarrier.arrive.shared::cluster",body)
        end
    end
    # Causal: the last two blocks masked, the rest unchanged; items handed
    # out dynamically.
    let tq = Tylo.TMABinding{typeof(TMATile(BFloat16,(128,128),2)),PTX.TMADescriptorPtr,3},
        tk = Tylo.TMABinding{typeof(TMATile(BFloat16,(128,128),1)),PTX.TMADescriptorPtr,3}
        code = Tylo.machine_code(blackwell_attention_kernel!,Tuple{CuDeviceArray{BFloat16,3,1},tq,tk,tq,CuDeviceVector{Int32,1},Int32,Int32,Float32,Val{4},Val{1},Val{3},Val{true},Val{true},Val{()}};arch,threads=512,maxregs=128)
        save_code("blackwell-attention-causal",code)
        @test isempty(audit(code))
        @test count("USETMAXREG",Tylo.kernel_sass(code)) == 3
    end
    # Given only its thread count, ptxas drops the register reallocation and
    # says so in a warning: the audit reports it.
    tq = Tylo.TMABinding{typeof(TMATile(BFloat16,(128,128),2)),PTX.TMADescriptorPtr,3}
    tk = Tylo.TMABinding{typeof(TMATile(BFloat16,(128,128),1)),PTX.TMADescriptorPtr,3}
    unstated = Tylo.machine_code(blackwell_attention_kernel!,Tuple{CuDeviceArray{BFloat16,3,1},tq,tk,tq,CuDeviceVector{Int32,1},Int32,Int32,Float32,Val{4},Val{1},Val{3},Val{false},Val{false},Val{()}};arch,threads=512)
    @test :ptxas in getfield.(audit(unstated),:kind)
end

if runtime_supported(@__FILE__)
@testset "Blackwell attention on Blackwell" begin
    function reference(q,k,v;causal=false)                         # d×seq per head, Float64
        o = similar(q,Float64)
        for h in axes(q,3)
            s = transpose(Float64.(q[:,:,h]))*Float64.(k[:,:,h]) ./ sqrt(size(q,1))
            causal && (s[[c > r for r in axes(s,1), c in axes(s,2)]] .= -Inf)
            p = exp.(s .- maximum(s;dims=2)); p ./= sum(p;dims=2)
            o[:,:,h] = transpose(p*transpose(Float64.(v[:,:,h])))
        end
        o
    end
    rng = MersenneTwister(3)
    # One block; several; more tiles than SMs, so that CTAs walk several; and
    # scores whose row maxima move by far more than the rescale threshold.
    for ctas in (1,2), (sq,skv,bh,amp) in ((512,128,1,1f0),(512,1024,2,1f0),(2048,1024,80,1f0),(512,512,3,8f0))
        q = BFloat16.(amp .* randn(rng,Float32,128,sq,bh))
        k = BFloat16.(randn(rng,Float32,128,skv,bh)); v = BFloat16.(randn(rng,Float32,128,skv,bh))
        kernel, launch, o = blackwell_attention(CuArray(q),CuArray(k),CuArray(v);ctas)
        fill!(o,BFloat16(NaN)); launch(); synchronize()
        @test maximum(abs.(Float64.(Array(o)) .- reference(q,k,v))) < 0.03
        @test isempty(audit(machine_code(kernel)))
    end
    # Causal: items of different lengths, longest first, walked by fewer
    # CTAs than there are items.
    for (sq,bh,amp) in ((256,1,1f0),(1024,3,1f0),(2048,80,1f0),(1024,2,8f0)), dynamic in (true,false)
        q = BFloat16.(amp .* randn(rng,Float32,128,sq,bh))
        k = BFloat16.(randn(rng,Float32,128,sq,bh)); v = BFloat16.(randn(rng,Float32,128,sq,bh))
        _, launch, o = blackwell_attention(CuArray(q),CuArray(k),CuArray(v);causal=true,dynamic)
        fill!(o,BFloat16(NaN)); launch(); synchronize()
        @test maximum(abs.(Float64.(Array(o)) .- reference(q,k,v;causal=true))) < 0.03
    end
    @test_throws ArgumentError blackwell_attention(CUDACore.zeros(BFloat16,128,512,1),CUDACore.zeros(BFloat16,128,512,1),CUDACore.zeros(BFloat16,128,512,1);causal=true,ctas=2)
    # A fifth of the middle chunks' exponentials on the FMA pipe.
    let q = BFloat16.(randn(rng,Float32,128,1024,2)), k = BFloat16.(randn(rng,Float32,128,1024,2)), v = BFloat16.(randn(rng,Float32,128,1024,2))
        _, launch, o = blackwell_attention(CuArray(q),CuArray(k),CuArray(v);emulate=(4,5,6,7,8,9))
        fill!(o,BFloat16(NaN)); launch(); synchronize()
        @test maximum(abs.(Float64.(Array(o)) .- reference(q,k,v))) < 0.03
    end
    # P published after one or two of its chunks instead of three.
    for ctas in (1,2), split in (1,2)
        q,k,v = (BFloat16.(randn(rng,Float32,128,1024,2)) for _ in 1:3)
        _, launch, o = blackwell_attention(CuArray(q),CuArray(k),CuArray(v);ctas,split)
        fill!(o,BFloat16(NaN)); launch(); synchronize()
        @test maximum(abs.(Float64.(Array(o)) .- reference(q,k,v))) < 0.03
    end
end
end
