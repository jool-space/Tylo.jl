# julia --project=test examples/hopper_attention/run.jl
# Times the Hopper attention forward pass (BF16, d = 128, non-causal) for a
# few shapes, tuning its stage count; compare with reference.py.
using Random
include("kernel.jl")
CUDACore.capability(CUDACore.device()) == v"9.0" || error("requires an H100 or H200")
for (bh,s) in ((64,4096),(32,8192),(128,2048))
    rng = MersenneTwister(1)
    q, k, v = (CuArray(BFloat16.(randn(rng,Float32,128,s,bh))) for _ in 1:3)
    tuning = tune([(; stages=2, persistent=true); [(; stages=s, persistent=false, overlap=o, pingpong=o) for s in (2,3) for o in (false,true)]]) do c
        kernel, launch, _ = hopper_attention(q,k,v;c...)
        kernel, launch
    end
    best = first(tuning.trials)
    flops = 4.0*bh*s*s*128
    println("batch·heads $bh, seq $s: Tylo ",round(flops/best.median/1e12;digits=1)," TFLOP/s (",
            round(1e3best.median;digits=3)," ms, ",best.candidate,")")
    for t in tuning.trials[2:end]
        println("    ",t.candidate,": ",round(flops/t.median/1e12;digits=1)," TFLOP/s")
    end
end
