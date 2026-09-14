# TEST_TARGET: cc==10|cc==11
# Datacenter attention: byte-identical machine code against the pinned PTX
# reference kernel wherever the CUDA compiler is available, and execution on
# B200/B300. The comparison needs the exact reference file; set TYLO_PTX_ROOT
# to a PTX checkout containing it when the active PTX package has moved on.
using SHA
const FA_TEST_SHA256 = "d4bcc34234bf2a9d85d9fed136f15e035d28dc84123f46d0651958745f132cdc" # keep equal to comparison.jl
reference = joinpath(get(ENV,"TYLO_PTX_ROOT",dirname(dirname(pathof(PTX)))),
                     "test","gpu","blackwell","flash_attention_defs.jl")
if isfile(reference) && bytes2hex(sha256(read(reference))) == FA_TEST_SHA256
    include(joinpath(@__DIR__,"..","..","examples","flash_attention","comparison.jl"))
else
    @testset "Datacenter attention comparison" begin
        @info "pinned PTX reference unavailable; set TYLO_PTX_ROOT to a checkout containing it" reference
        @test_skip false
    end
end
