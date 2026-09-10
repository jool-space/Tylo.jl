using Tylo.Layouts: @Layout

# Construct layouts inside the kernel: runtime sizes/strides must not create
# value-dependent types, while the Val-derived size remains static.
function layout_notation_kernel!(out,n,ld,::Val{K}) where K
    t = Int32(threadIdx().x)-Int32(1)
    l = @Layout ((8, 4), $n) ((1, 8), $ld)
    s = @Layout (K, 1) (1, 0)
    @inbounds out[t+1] = l(((t%Int32(8), t÷Int32(8)), n-Int32(1)))
    @inbounds out[32+t+1] = s((t, Int32(0)))
    nothing
end

if CUDACore.functional()
@testset "Layout notation in device code" begin
    out = CuArray{Int32}(undef,64)
    for (n,ld) in ((Int32(5),Int32(128)),(Int32(7),Int32(256)))
        @cuda threads=32 layout_notation_kernel!(out,n,ld,Val(32))
        expected = vcat(Int32.(0:31) .+ ld*(n-Int32(1)), Int32.(0:31))
        @test Array(out) == expected
    end
end
end
