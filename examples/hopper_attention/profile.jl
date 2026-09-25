# One attention launch after warm-up, for a profiler:
#   ncu -k regex:attention_kernel --launch-skip 2 --launch-count 1 julia --project=test examples/hopper_attention/profile.jl [stages overlap pingpong]
include("kernel.jl")
stages, overlap, pingpong = length(ARGS) >= 3 ? (parse(Int,ARGS[1]),parse(Bool,ARGS[2]),parse(Bool,ARGS[3])) : (2,false,false)
q, k, v = (CuArray(BFloat16.(randn(Float32,128,4096,64))) for _ in 1:3)
_, launch, _ = hopper_attention(q,k,v;stages,overlap,pingpong)
for _ in 1:3
    launch()
end
CUDACore.synchronize()
