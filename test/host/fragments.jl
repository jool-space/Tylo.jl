@testset "Row ownership" begin
    for n in (16,32,64,128)
        layout = lane_local(n,2)
        @test size(layout) == (32,n)
        owned = [Tylo.Layouts.coordinate(layout,Int32(t),e)
                 for t in 0:31 for e in 0:n-1]
        @test Set(owned) == Set((Int32(t),Int32(e)) for t in 0:31 for e in 0:n-1)
        @test length(unique(owned)) == 32n
    end
    @test_throws ArgumentError Ownership((32,0),Layout((static(32),static(1)),(static(1),static(32))))
    @test_throws BoundsError Tylo.Layouts.coordinate(lane_local(32,2),0,32)
end

@testset "Fragments and static partitions" begin
    for n in (16,32,64,128)
        values = ntuple(i -> Float32(i-7)/3f0, n)
        f = local_fragment(values)
        @test isbitstype(typeof(f))
        @test size(Tylo.Layouts.layout(f)) == (32,n)
        @test (f .* 0.25f0).data == values .* 0.25f0
        @test map(abs,f).data == abs.(values)
        @test map(x -> Float64(x),f) isa Fragment{Float64,n}
        halves = (window(f,(0,0),(32,n÷2)),window(f,(0,n÷2),(32,n÷2)))
        @test (halves[1].data...,halves[2].data...) == values
        for offset in 0:8:n-8
            @test window(f,(0,offset),(32,8)).data == values[offset+1:offset+8]
        end
    end
    # 16-bit values are logical values like any other; windows need no word alignment.
    p = Fragment(ntuple(i -> BFloat16(i),64),lane_local(64,2))
    @test size(Tylo.Layouts.layout(p)) == (32,64)
    @test window(p,(0,32),(32,32)).data == ntuple(i -> BFloat16(i+32),32)
    @test window(p,(0,1),(32,2)).data == (BFloat16(2),BFloat16(3))
    @test_throws ArgumentError window(local_fragment((1f0,2f0)),(0,1),(32,2))
    @test_throws ArgumentError window(local_fragment((1f0,2f0)),(0,-1),(32,1))
    @test_throws MethodError getindex(local_fragment((1f0,2f0)),1)
    @test_throws ArgumentError local_fragment(())
end
