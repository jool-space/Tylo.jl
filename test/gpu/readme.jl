# TEST_TARGET: cc>=8.0
# The README's kernel, as written there, and the machine code it promises.
const README_ROW = distribute((1, 256), 32; vector=8, axis=2)
@inline readme_rows(a) = @inbounds permutedims(GlobalTile(a, 16))

function readme_softmax_rows!(y, x)
    lane, row = Int32(threadIdx().x) - Int32(1), Int32(blockIdx().x) - Int32(1)
    @inbounds begin
        f = Float32.(load_fragment(README_ROW, tile(readme_rows(x), (1, 256), (row, 0)), lane))
        w = exp.(f .- maximum(f; dims=2))
        store!(tile(readme_rows(y), (1, 256), (row, 0)), Fragment{BFloat16}(w ./ sum(w; dims=2)), lane)
    end
    return
end

@testset "README kernel assembly" begin
    tt = Tuple{CuDeviceMatrix{BFloat16,1},CuDeviceMatrix{BFloat16,1}}
    code = compile_kernel(readme_softmax_rows!, tt; arch=CUDACore.SMVersion(12,1,:arch), threads=32)
    save_code("readme-softmax", code)
    body = entry_body(code.ptx)
    @test count(r"ld\.global\.(v4\.b32|v2\.b64)", body) == 1
    @test count(r"st\.global\.(v4\.b32|v2\.b64)", body) == 1
    @test count("shfl.sync.bfly", body) == 10
    @test count("cvt.rn.bf16x2.f32", body) == 4
    @test !occursin(r"\bcall", body)
    @test !occursin(".local .", body)
end

if runtime_supported(@__FILE__)
    @testset "README kernel" begin
        x = CuArray(BFloat16.(4 .* randn(MersenneTwister(7), Float32, 256, 100)))
        y = similar(x)
        @cuda threads=32 blocks=size(x, 2) readme_softmax_rows!(y, x)
        X = Float32.(Array(x)); E = exp.(X .- maximum(X; dims=1))
        @test maximum(abs.(Float32.(Array(y)) .- E ./ sum(E; dims=1))) < 4f-3   # BF16 output
    end
end
