# TEST_TARGET: cc==10
# The Blackwell FlashAttention forward example, written in Tylo throughout:
# warp-specialized roles with register reallocation, S, P and O in TMEM, P
# published in two parts. Compiled clean for sm_100a, correct on B200/B300.
include("../../examples/blackwell_attention/kernel.jl")

@testset "Blackwell attention assembly" begin
    tq = Tylo.TMABinding{typeof(TMATile(BFloat16,(128,128),2)),PTX.TMADescriptorPtr,3}
    tk = Tylo.TMABinding{typeof(TMATile(BFloat16,(128,128),1)),PTX.TMADescriptorPtr,3}
    tt = Tuple{CuDeviceArray{BFloat16,3,1},tq,tk,tq,Int32,Int32,Float32,Val{4}}
    arch = CUDACore.SMVersion(10,0,:arch)
    code = Tylo.machine_code(blackwell_attention_kernel!,tt;arch,threads=512,maxregs=128)
    save_code("blackwell-attention",code)
    # No spills anywhere: the softmax warps hold a row at 192 registers, the
    # MMA warp its operands at 56, because each product derives its
    # descriptors at the issue.
    @test isempty(audit(code))
    @test count("USETMAXREG",Tylo.kernel_sass(code)) == 3
    body = entry_body(code.ptx)
    @test count("tcgen05.ld.sync.aligned.32x32b.x32",body) >= 4
    @test count("tcgen05.st.sync.aligned.32x32b.x16",body) == 4
    # Given only its thread count, ptxas drops the register reallocation and
    # says so in a warning: the audit reports it.
    unstated = Tylo.machine_code(blackwell_attention_kernel!,tt;arch,threads=512)
    @test :ptxas in getfield.(audit(unstated),:kind)
end

if runtime_supported(@__FILE__)
@testset "Blackwell attention on Blackwell" begin
    function reference(q,k,v)                                       # d×seq per head, Float64
        o = similar(q,Float64)
        for h in axes(q,3)
            s = transpose(Float64.(q[:,:,h]))*Float64.(k[:,:,h]) ./ sqrt(size(q,1))
            p = exp.(s .- maximum(s;dims=2)); p ./= sum(p;dims=2)
            o[:,:,h] = transpose(p*transpose(Float64.(v[:,:,h])))
        end
        o
    end
    rng = MersenneTwister(3)
    # One block; several; more tiles than SMs, so that CTAs walk several; and
    # scores whose row maxima move by far more than the rescale threshold.
    for (sq,skv,bh,amp) in ((256,128,1,1f0),(512,1024,2,1f0),(2048,1024,80,1f0),(512,512,3,8f0))
        q = BFloat16.(amp .* randn(rng,Float32,128,sq,bh))
        k = BFloat16.(randn(rng,Float32,128,skv,bh)); v = BFloat16.(randn(rng,Float32,128,skv,bh))
        kernel, launch, o = blackwell_attention(CuArray(q),CuArray(k),CuArray(v))
        fill!(o,BFloat16(NaN)); launch(); synchronize()
        @test maximum(abs.(Float64.(Array(o)) .- reference(q,k,v))) < 0.03
        @test isempty(audit(machine_code(kernel)))
    end
end
end
