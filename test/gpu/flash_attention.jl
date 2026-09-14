# TEST_TARGET: cc==10|cc==11
# Datacenter attention: byte-identical machine code against the reference
# kernel wherever the CUDA compiler is available, and execution on B200/B300.
include(joinpath(@__DIR__,"..","..","examples","flash_attention","comparison.jl"))
