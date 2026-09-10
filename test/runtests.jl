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

@testset "TMEM units, aliases, and warp bands" begin
    for base in (UInt32(0),UInt32(32)), width in (32,64,128)
        tile = TmemTile{Float32,width}(base)
        @test size(tile) == (128,width)
        for warp in UInt32(0):UInt32(3)
            rows = warp_rows(tile,warp)
            @test rows.address == base + UInt32(32warp) * UInt32(65536)
            @test size(rows) == (32,width)
            for offset in 0:16:width-16
                sub = columns(rows,Val(offset),Val(16))
                @test sub.address == rows.address + UInt32(offset)
                @test sub.address >> 16 == rows.address >> 16
            end
            bf = reinterpret_tile(BFloat16,rows)
            @test size(bf) == (32,2width)
            @test reinterpret_tile(Float32,bf).address == rows.address
            for offset in 0:16:2width-16
                @test columns(bf,Val(offset),Val(16)).address ==
                      rows.address + UInt32(offset÷2)
            end
        end
    end
    # S0/S1 become packed P0/P1; O occupies the other half of the allocation.
    all = TmemTile{Float32,512}(UInt32(0))
    for stage in 0:1
        s = columns(all,Val(stage*128),Val(128))
        p = columns(reinterpret_tile(BFloat16,s),Val(0),Val(128))
        o = columns(all,Val(256+stage*128),Val(128))
        @test p.address == s.address
        @test p.address + UInt32(64) <= o.address
        @test columns(p,Val(64),Val(64)).address == s.address + UInt32(32)
    end
    @test_throws ArgumentError TmemTile{BFloat16,3}(UInt32(0))
    @test_throws ArgumentError TmemTile{Float32,513}(UInt32(0))
    @test_throws ArgumentError columns(all,Val(500),Val(32))
    @test_throws ArgumentError columns(reinterpret_tile(BFloat16,all),Val(1),Val(16))
    pending = Tylo.PendingLoad(ntuple(UInt32,32))
    @test_throws MethodError scale(pending,2f0)
    @test_throws MethodError map(abs,pending)
end

@testset "Runtime TMEM windows, static register windows" begin
    tile = TmemTile{Float32,128}(UInt32(16))
    rows = warp_rows(tile,UInt32(2))
    for offset in UInt32(0):UInt32(64)
        @test columns(rows,offset,Val(64)).address == rows.address+offset
    end
    bf = reinterpret_tile(BFloat16,rows)
    for offset in UInt32(0):UInt32(2):UInt32(192)
        @test columns(bf,offset,Val(64)).address == rows.address+offset÷UInt32(2)
    end
    @test_throws BoundsError columns(rows,UInt32(65),Val(64))
    @test_throws ArgumentError columns(bf,UInt32(1),Val(64))
    @test_throws MethodError columns(RowFragment((1f0,2f0)),UInt32(0),Val(1))
end

include("layouts.jl")
include("layout_macro.jl")

include("hopper.jl")

include("rows.jl")

include("online.jl")

include("operand_a.jl")
