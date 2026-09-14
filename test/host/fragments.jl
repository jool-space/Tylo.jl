@testset "Row ownership" begin
    for n in (16,32,64,128)
        layout = Tylo.Layouts.LocalOwnership{n,2}()
        @test size(layout) == (32,n)
        owned = [Tylo.Layouts.coordinate(layout,Int32(t),Val(e))
                 for t in 0:31 for e in 0:n-1]
        @test Set(owned) == Set((Int32(t),Int32(e)) for t in 0:31 for e in 0:n-1)
        @test length(unique(owned)) == 32n
    end
    @test_throws ArgumentError Tylo.Layouts.LocalOwnership{0,2}()
    @test_throws BoundsError Tylo.Layouts.coordinate(Tylo.Layouts.LocalOwnership{32,2}(),0,Val(32))
end

@testset "Fragments and static partitions" begin
    for n in (16,32,64,128)
        values = ntuple(i -> Float32(i-7)/3f0, n)
        f = local_fragment(values)
        @test isbitstype(typeof(f))
        @test size(Tylo.Layouts.layout(f)) == (32,n)
        @test scale(f,0.25f0).data == values .* 0.25f0
        @test map(abs,f).data == abs.(values)
        @test map(x -> Float64(x),f) isa Fragment{Float64,n}
        halves = (window(f,Val((0,0)),Val((32,n÷2))),window(f,Val((0,n÷2)),Val((32,n÷2))))
        @test (halves[1].data...,halves[2].data...) == values
        for offset in 0:8:n-8
            @test window(f,Val((0,offset)),Val((32,8))).data == values[offset+1:offset+8]
        end
    end
    p = PackedFragment(BFloat16,ntuple(UInt32,32),Tylo.Layouts.LocalOwnership{64,2}())
    @test size(Tylo.Layouts.layout(p)) == (32,64)
    @test window(p,Val((0,32)),Val((32,32))).data == ntuple(i -> UInt32(i+16),16)
    @test_throws ArgumentError window(local_fragment((1f0,2f0)),Val((0,1)),Val((32,2)))
    @test_throws ArgumentError window(local_fragment((1f0,2f0)),Val((0,-1)),Val((32,1)))
    @test_throws ArgumentError window(p,Val((0,1)),Val((32,2)))
    @test_throws MethodError getindex(local_fragment((1f0,2f0)),1)
    @test_throws ArgumentError local_fragment(())
end
