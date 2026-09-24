# TEST_TARGET: cc>=8.9
using Tylo.Layouts: @Layout, Layout, static

# One thread's 8×8 chunk of a K × 8 matrix, its per-column statistics, and the
# statistics as one row of a warp-wide matrix.
const CHUNK = Ownership((8,8))
const STATS = Ownership((1,8))
const WARP = Ownership((32,8),@Layout((32,8),(1,32)))

# Column maxima of the first `batch` columns (absent columns read as zero)
# and the E4M3 copy of those columns (absent columns untouched).
function chunks_kernel!(maxima,y,x,batch)
    lane = Int32(threadIdx().x)-Int32(1)
    kmax = size(x,1) % Int32
    input = @inbounds bound(GlobalTile(x,16),(nothing,batch))
    output = @inbounds bound(GlobalTile(y,8),(nothing,batch))
    m = zeros(Float32,STATS)
    # The loop keeps every chunk inside the rows.
    @inbounds for chunk in lane:Int32(32):kmax÷Int32(8)-Int32(1)
        v = Float32.(load_fragment(CHUNK,tile(input,(8,8),(chunk,0))))
        m = maximum(abs,v;dims=1,init=m)
        store!(tile(output,(8,8),(chunk,0)),Fragment{Float8_E4M3FN}(v))
    end
    # Every lane holds the reduced row; one of them writes it. The launch
    # is one warp and `maxima` is 1×8.
    @inbounds store!(GlobalTile(maxima),maximum(Fragment(m,WARP);dims=1),lane)
    nothing
end

@testset "Bounded chunks assembly" begin
    tt = Tuple{CuDeviceMatrix{Float32,1},CuDeviceMatrix{Float8_E4M3FN,1},CuDeviceMatrix{BFloat16,1},Int32}
    code = compile_kernel(chunks_kernel!,tt;arch=CUDACore.SMVersion(12,1,:arch),threads=32)
    save_code("tiles-chunks",code)
    body = entry_body(code.ptx)
    @test !occursin(".local .",body)
    @test !occursin(r"\bcall",body)
    @test count("ld.global.v4",body) == 8          # one 16-byte vector per column
    @test count("st.global.v2",body) + count("st.global.b64",body) == 8   # eight E4M3 per column
    @test count("cvt.rn.satfinite.e4m3x2.f32",body) == 32
    @test count("shfl.sync.bfly",body) == 40       # five rounds for each column
    @test count("st.global.f32",body) + count("st.global.b32",body) == 8
end

if runtime_supported(@__FILE__)
    @testset "Bounded chunks" begin
        rng = MersenneTwister(12)
        kmax = 512
        for batch in (1,3,8)
            values = BFloat16.(randn(rng,Float32,kmax,8) .* 40)
            maxima = CUDACore.fill(-1f0,8)
            sentinel = reinterpret(Float8_E4M3FN,0x55)
            y = CUDACore.fill(sentinel,kmax,8)
            @cuda threads=32 chunks_kernel!(reshape(maxima,1,8),y,CuArray(values),Int32(batch))
            expected = [c <= batch ? maximum(abs.(Float32.(values[:,c]))) : 0f0 for c in 1:8]
            @test Array(maxima) == expected
            host = Array(y)
            @test all(host[:,c] == Tylo.convert_element.(Float8_E4M3FN,Float32.(values[:,c])) for c in 1:batch)
            @test all(==(sentinel),host[:,batch+1:end])
        end
    end
end

# A bound along the axis vectors run on must not fall inside a vector:
# checked, a straddling bound is an error rather than a read past it.
const COLUMN = Ownership((8,1))
function straddle_kernel!(y,x,rows)
    t = bound(GlobalTile(x,16),(rows,nothing))
    store!(GlobalTile(y,16),load_fragment(COLUMN,t))
    nothing
end
function straddle_unchecked_kernel!(y,x,rows)
    t = @inbounds bound(GlobalTile(x,16),(rows,nothing))
    @inbounds store!(GlobalTile(y,16),load_fragment(COLUMN,t))
    nothing
end

@testset "Bounds inside a vector" begin
    tt = Tuple{CuDeviceMatrix{BFloat16,1},CuDeviceMatrix{BFloat16,1},Int32}
    checked = entry_body(compile_kernel(straddle_kernel!,tt;arch=CUDACore.SMVersion(12,1,:arch),threads=32).ptx)
    unchecked = entry_body(compile_kernel(straddle_unchecked_kernel!,tt;arch=CUDACore.SMVersion(12,1,:arch),threads=32).ptx)
    wide = r"ld\.global\.(v4\.b32|v2\.b64)"
    @test count(wide,checked) == count(wide,unchecked) == 1
    @test occursin("trap",checked) || occursin(r"\bcall",checked)
    @test !occursin("trap",unchecked) && !occursin(r"\bcall",unchecked)
end
