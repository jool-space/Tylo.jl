using Tylo.Layouts: @Layout, Layout, static

# Host-side tiles: pointers are plain addresses, so tile arithmetic is
# checked without a device.
tptr(::Type{T},address) where T = reinterpret(Core.LLVMPtr{T,1},UInt64(address))
address(t,c) = UInt64(reinterpret(UInt64,pointer(t,c)))

@testset "Layout splits of runtime indices" begin
    # Non-negative indices split static powers of two with masks and shifts;
    # other extents divide. Both agree with general evaluation.
    for l in (Layout((static(4),static(3),Int32(5)),(static(1),static(4),static(12))),
              Layout((static(8),static(4),static(2),Int32(3)),(static(1),static(16),static(8),static(64))),
              Layout(((static(2),static(3)),Int32(4)),((static(3),static(1)),static(6))))
        n = length(l)
        for c in Int32(0):Int32(n-1)
            @test Tylo.Layouts._offset_nonnegative(l,c) == l(c) == l(Int(c))
            @test typeof(Tylo.Layouts._offset_nonnegative(l,c)) == Int32
        end
    end
end

@testset "Tiles by index" begin
    # A tile addresses the same elements as its parent at `index .* shape`.
    kmax = Int32(256)
    trueK = Layout(((static(8),static(4),static(2),kmax÷Int32(64)),static(8)),
                   ((static(1),static(16),static(8),static(64)),kmax))
    mmaK = Layout(((static(4),static(4),static(2),static(2),kmax÷Int32(64)),Int32(8)),
                  ((static(1),static(16),static(4),static(8),static(64)),kmax))
    rows = Layout((Int32(40),static(8)),(static(1),Int32(40)))
    for (l,shape,align) in ((trueK,(8,8),16),(mmaK,(64,8),16),(rows,(8,8),8),(rows,(4,8),8))
        t = GlobalTile(tptr(UInt8,1024),l,align)
        m,n = size(l)
        for i in 0:m÷shape[1]-1, j in 0:n÷shape[2]-1
            s = tile(t,shape,(Int32(i),Int32(j)))
            @test size(s) == shape
            @test all(address(s,(r,c)) == address(t,(shape[1]i+r,shape[2]j+c))
                      for r in 0:shape[1]-1, c in 0:shape[2]-1)
        end
    end
    # A tile must cover whole inner modes of a hierarchical axis.
    @test_throws ArgumentError tile(GlobalTile(tptr(UInt8,1024),trueK),(12,8),(Int32(0),Int32(0)))
    # Alignment narrows to what the static outer strides keep.
    @test Tylo.alignment(tile(GlobalTile(tptr(UInt8,1024),trueK,16),(8,8),(Int32(1),Int32(0)))) == 8
    @test Tylo.alignment(tile(GlobalTile(tptr(UInt8,1024),mmaK,16),(64,8),(Int32(1),Int32(0)))) == 16
end

@testset "Bounds" begin
    t = GlobalTile(tptr(Float32,1024),@Layout((32,8),(1,32)))
    b = bound(t,(nothing,Int32(5)))
    @test b.bounds == (nothing,Int32(5))
    @test tile(b,(8,8),(Int32(2),Int32(0))).bounds == (nothing,Int32(5))
    @test tile(bound(t,(Int32(20),nothing)),(8,8),(Int32(2),Int32(0))).bounds == (Int32(4),nothing)
    @test window(bound(t,(Int32(20),Int32(5))),(Int32(3),Int32(1)),(8,4)).bounds == (Int32(17),Int32(4))
    @test t.bounds === nothing
end

@testset "Re-owning and reinterpreting fragments" begin
    stats = Ownership((1,8),@Layout((1,8),(0,1)))
    warp = Ownership((32,8),@Layout((32,8),(1,32)))
    f = Fragment(ntuple(Float32,8),stats)
    g = Fragment(f,warp)
    @test g.data == f.data && Tylo.Layouts.layout(g) == warp
    @test_throws DimensionMismatch Fragment(f,lane_local(4,2))
    codes = Fragment(ntuple(i -> UInt8(120+i),8),stats)
    e8 = reinterpret(Float8_E8M0FNU,codes)
    @test eltype(e8) === Float8_E8M0FNU
    @test Tylo._pack_words(e8) === Tylo._pack_words(codes)
    @test reinterpret.(UInt8,e8.data) == codes.data
    @test reinterpret(UInt32,f).data == reinterpret.(UInt32,f.data)
    @test_throws ArgumentError reinterpret(UInt16,codes)
end

@testset "Selection by thread" begin
    # Every lane holds all eight columns; a lane of quad q takes column q.
    columns = Ownership((1,8),@Layout((32,8),(0,1)))
    sb = operand_layout(ScaledMMAAtom((16,8,32),Float8_E4M3FN),ScaleB())
    f = Fragment(ntuple(i -> reinterpret(Float8_E8M0FNU,UInt8(100+i)),8),columns)
    for lane in Int32(0):Int32(31)
        s = relayout(sb,f,lane)
        @test all(==(reinterpret(Float8_E8M0FNU,UInt8(101+lane÷4))),s.data)
    end
    @test_throws ArgumentError relayout(sb,f)
end

@testset "Reductions from an initial value" begin
    stats = Ownership((1,8),@Layout((1,8),(0,1)))
    chunk = Ownership((8,8),@Layout((1,64),(0,1)))
    x = Fragment(ntuple(i -> Float32(i-30)/7,64),chunk)
    s = Fragment(ntuple(i -> Float32(i),8),stats)
    v = reshape(collect(x.data),8,8)
    # Values a thread holds fold sequentially from the initial value.
    # (`muladd` may or may not fuse on the host.)
    @test all(isapprox.(sum(abs2,x;dims=1,init=s).data,Tuple(foldl((a,y) -> a + y*y,v[:,c];init=s.data[c]) for c in 1:8);rtol=1e-6))
    @test maximum(abs,x;dims=1,init=s).data == Tuple(foldl((a,y) -> max(a,abs(y)),v[:,c];init=s.data[c]) for c in 1:8)
    @test_throws DimensionMismatch sum(x;dims=1,init=Fragment(ntuple(Float32,8),lane_local(8,2)))
    # A replicated value is written once, by the replica whose replicating bits are zero.
    @test Tylo.replica_mask(Ownership((1,8),@Layout((32,8),(0,1)))) == 31
    @test Tylo.replica_mask(stats) == 0
end
