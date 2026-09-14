# TEST_TARGET: cc>=8.0
include("../../examples/streaming_attention/kernel.jl")
include("../../examples/streaming_attention/reference.jl")
begin # assembly checks
@testset "Streaming attention assembly" begin
    cfg=StreamingAttention.configuration()
    for causal in (false,true)
        tt=Tuple{CuDeviceMatrix{Float32,1},CuDeviceMatrix{BFloat16,1},CuDeviceMatrix{BFloat16,1},
            CuDeviceMatrix{BFloat16,1},CuDeviceMatrix{Bool,1},Int32,Int32,typeof(cfg),Val{causal}}
        code=compile_kernel(StreamingAttention.attention_kernel!,tt;arch=CUDACore.SMVersion(12,1,:arch),threads=128)
        save_code("streaming-attention-causal-$causal",code);body=entry_body(code.ptx)
        @test occursin("mma.sync.aligned.m16n8k16",body)
        @test occursin("cvt.rn.bf16x2.f32",body)
        @test occursin("cp.async",body)
        @test !occursin(r"\bcall",body)
    end

end
end
if runtime_supported(@__FILE__)
@testset "Streaming attention masks, tails, empty keys and replay" begin
    for (m,n) in ((1,1),(3,0),(17,29),(64,64),(65,97),(129,257)),causal in (false,true)
        rng=MersenneTwister(m+n)
        q=BFloat16.(randn(rng,Float32,64,m));k=BFloat16.(randn(rng,Float32,64,n))
        v=BFloat16.(randn(rng,Float32,n,64))
        mask=rand(rng,Float32,n,m).>0.2f0
        m==64 && (mask.=true)
        m>1 && (mask[:,2].=false)
        n>64 && m>3 && (mask[1:64,3].=false) # initial empty chunks, then valid
        n>32 && m>4 && (mask[33:min(64,n),4].=false)
        dq,dk,dv,dm=CuArray(q),CuArray(k),CuArray(v),CuArray(mask)
        out=CuArray{Float32}(undef,64,m)
        run=()->StreamingAttention.launch!(out,dq,dk,dv,dm;causal)
        run()
        graph=CUDACore.instantiate(CUDACore.capture(run))
        GC.@preserve out dq dk dv dm graph begin
        for replay in 1:2
            if replay==2
                q .*= BFloat16(3);k .*= BFloat16(3);v .*= BFloat16(-0.5)
                copyto!(dq,q);copyto!(dk,k);copyto!(dv,v)
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
end
end
