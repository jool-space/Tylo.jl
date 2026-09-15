# TEST_TARGET: cc>=8.0
include("../../examples/streaming_attention/kernel.jl")
include("../../examples/streaming_attention/reference.jl")
begin # assembly checks
@testset "Streaming attention assembly" begin
    cfg=StreamingAttention.configuration()
    binding(plan)=Tylo.TMABinding{typeof(plan),PTX.TMADescriptorPtr,3}
    for causal in (false,true), maskt in (CuDeviceMatrix{Bool,1},Nothing)
        tt=Tuple{CuDeviceArray{Float32,3,1},binding(cfg.q),binding(cfg.k),binding(cfg.v),maskt,Int32,Int32,typeof(cfg),Val{causal}}
        code=compile_kernel(StreamingAttention.attention_kernel!,tt;arch=CUDACore.SMVersion(12,1,:arch),threads=256)
        save_code("streaming-attention-causal-$causal-$(maskt===Nothing ? "unmasked" : "masked")",code);body=entry_body(code.ptx)
        @test count("mma.sync.aligned.m16n8k16",body)==32+32+32 # QK and PV of the loop, the last PV block
        @test count("ldmatrix.sync.aligned",body)==4+16+16+16
        @test count("cp.async.bulk.tensor.3d",body)==2+2+2+2 # Q, tile 0, the prologue loop, the main loop's prefetch
        @test count("bar.sync",body)==3 && count("bar.arrive",body)==2
        @test occursin("cvt.rn.bf16x2.f32",body)
        @test !occursin(".local .",body)
        @test !occursin(r"\bcall",body)
    end

end
end
if runtime_supported(@__FILE__)
@testset "Streaming attention masks, tails, empty keys and replay" begin
    for (m,n) in ((1,1),(3,0),(17,29),(64,64),(65,97),(129,257),(200,300)),causal in (false,true)
        rng=MersenneTwister(m+n)
        q=BFloat16.(randn(rng,Float32,64,m));k=BFloat16.(randn(rng,Float32,64,n))
        v=BFloat16.(randn(rng,Float32,n,64))
        mask=rand(rng,Float32,n,m).>0.2f0
        m==64 && (mask.=true)
        m>1 && (mask[:,2].=false)
        n>64 && m>3 && (mask[1:64,3].=false) # initial empty chunks, then valid
        n>32 && m>4 && (mask[33:min(64,n),4].=false)
        # V keeps an eight-aligned leading dimension so the binding outlives replays.
        dq,dk,dm=CuArray(q),CuArray(k),CuArray(mask)
        dv=CUDACore.zeros(BFloat16,cld(n,8)*8,64);dv[1:n,:]=v
        out=CuArray{Float32}(undef,64,m)
        prepared=StreamingAttention.prepare(dq,dk,dv)
        run=()->StreamingAttention.launch!(out,prepared,dm;causal)
        run()
        graph=CUDACore.instantiate(CUDACore.capture(run))
        GC.@preserve out prepared dq dk dv dm graph begin
        for replay in 1:2
            if replay==2
                q .*= BFloat16(3);k .*= BFloat16(3);v .*= BFloat16(-0.5)
                copyto!(dq,q);copyto!(dk,k);dv[1:n,:]=v
                CUDACore.launch(graph)
            end
            y=Array(out)
            reference=attention_reference(q,k,v,mask;causal)
            diagnostic=attention_reference(q,k,v,mask;causal,rounded=true)
            scale=max(maximum(abs,Float64.(v);init=0.0),1.0)
            # BF16 weights introduce <=~0.4% relative coefficient error;
            # absolute tolerance also covers cancellation near zero.
            @test maximum(abs,y.-reference) <= 0.004*scale+3e-5
            @test maximum(abs,y.-diagnostic) <= 0.002*scale+3e-5
            @test all(isfinite,y)
            if m>1; @test all(iszero,y[:,2]);end
            if n==0; @test all(iszero,y);end
        end
        end
    end
    # Padded V uses an aligned leading dimension without changing logical N.
    m,n=65,97;rng=MersenneTwister(800)
    q=BFloat16.(randn(rng,Float32,64,m));k=BFloat16.(randn(rng,Float32,64,n))
    v=BFloat16.(randn(rng,Float32,n,64));padded=fill(BFloat16(NaN),104,64);padded[1:n,:]=v
    mask=trues(n,m);out=CuArray{Float32}(undef,64,m)
    StreamingAttention.launch!(out,CuArray(q),CuArray(k),CuArray(padded),CuArray(mask))
    @test Array(out) ≈ attention_reference(q,k,v,mask) rtol=0.005 atol=0.002
    @test all(isfinite,Array(out))
    # Heads share the mask; without a mask the plain path scales scores only.
    heads=3;m,n=150,203;rng=MersenneTwister(900)
    q=BFloat16.(randn(rng,Float32,64,m,heads));k=BFloat16.(randn(rng,Float32,64,n,heads));v=BFloat16.(randn(rng,Float32,n,64,heads))
    out=CuArray{Float32}(undef,64,m,heads)
    for causal in (false,true)
        StreamingAttention.launch!(out,CuArray(q),CuArray(k),CuArray(v);causal)
        y=Array(out)
        for h in 1:heads
            @test y[:,:,h] ≈ attention_reference(q[:,:,h],k[:,:,h],v[:,:,h],trues(n,m);causal) rtol=0.005 atol=0.002
        end
    end
end
end
