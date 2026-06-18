# Shared bindings loaded into every test worker via runtests.jl's `init_code`.
# Each test file (flex/attention/decode/softmax/norm) is self-contained — it
# defines its own Float64 reference functions and `@testset`s and only relies
# on the `using` block below.
using Tylo
using Adapt: adapt
using CUDA
using LinearAlgebra
using Random
using Test
