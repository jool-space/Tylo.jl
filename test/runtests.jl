using Tylo, BFloat16s, Test

@testset "Row ownership" begin
    for n in (16,32,64,128)
        layout = Tylo.Layouts.LaneRows{n}()
        @test size(layout) == (32,n)
        owned = [Tylo.Layouts.coordinate(layout,Int32(t),Val(e))
                 for t in 0:31 for e in 0:n-1]
        @test Set(owned) == Set((Int32(t),Int32(e)) for t in 0:31 for e in 0:n-1)
        @test length(unique(owned)) == 32n
    end
    @test_throws ArgumentError Tylo.Layouts.LaneRows{0}()
    @test_throws BoundsError Tylo.Layouts.coordinate(Tylo.Layouts.LaneRows{32}(),0,Val(32))
end

@testset "Fragments and static partitions" begin
    for n in (16,32,64,128)
        values = ntuple(i -> Float32(i-7)/3f0, n)
        f = RowFragment(values)
        @test isbitstype(typeof(f))
        @test size(Tylo.Layouts.layout(f)) == (32,n)
        @test scale(f,0.25f0).data == values .* 0.25f0
        @test map(abs,f).data == abs.(values)
        @test map(x -> Float64(x),f) isa RowFragment{Float64,n}
        halves = (columns(f,Val(0),Val(n÷2)),columns(f,Val(n÷2),Val(n÷2)))
        @test (halves[1].data...,halves[2].data...) == values
        for offset in 0:8:n-8
            @test columns(f,Val(offset),Val(8)).data == values[offset+1:offset+8]
        end
    end
    p = PackedBF16(ntuple(UInt32,32))
    @test size(Tylo.Layouts.layout(p)) == (32,64)
    @test columns(p,Val(32),Val(32)).data == ntuple(i -> UInt32(i+16),16)
    @test_throws ArgumentError columns(RowFragment((1f0,2f0)),Val(1),Val(2))
    @test_throws ArgumentError columns(RowFragment((1f0,2f0)),Val(-1),Val(1))
    @test_throws ErrorException columns(p,Val(1),Val(2))
    @test_throws MethodError getindex(RowFragment((1f0,2f0)),1)
    @test_throws ArgumentError RowFragment(())
end

include("tmem.jl")

include("layouts.jl")
include("layout_macro.jl")

include("hopper.jl")

include("rows.jl")
include("arrayops.jl")

include("online.jl")

include("operand_a.jl")
