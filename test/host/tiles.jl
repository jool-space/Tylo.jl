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

@testset "Named ownerships and filled fragments" begin
    one = Ownership((8,8))
    @test one == Ownership((8,8),@Layout((1,64),(0,1)))
    @test Tylo.threads(one) == 1
    @test [Tylo.Layouts.coordinate(one,0,e) for e in (0,1,8,63)] == [(0,0),(1,0),(0,1),(7,7)]
    replicated = Ownership((1,8);threads=32)
    @test replicated == Ownership((1,8),@Layout((32,8),(0,1)))
    @test Tylo.threads(replicated) == 32
    @test all(Tylo.Layouts.coordinate(replicated,t,3) == (0,3) for t in 0:31)
    @test fill(0x7f,Ownership((1,8))).data == ntuple(_ -> 0x7f,8)
    @test zeros(Float32,one).data == ntuple(_ -> 0f0,64)
    @test Tylo._pack_words(fill(0x7f,Ownership((1,8)))) == (0x7f7f7f7f,0x7f7f7f7f)
end

@testset "Tiles from arrays and pointers" begin
    # Host arrays stand in for device arrays: the layout is the array's own.
    A = zeros(Float32,6,5)
    p = reinterpret(Core.LLVMPtr{Float32,1},UInt64(1024))
    t = GlobalTile(p,Tylo._column_major(A))
    @test size(t) == (6,5)
    @test all(address(t,(r,c)) == 1024 + 4*(r + 6c) for r in 0:5, c in 0:4)
    @test typeof(size(t.layout)[1]) == Int32
    @test size(GlobalTile(p,Tylo._column_major(zeros(Float32,7)))) == (7,1)
    # A byte pointer, such as a workspace, typed at construction.
    bytes = reinterpret(Core.LLVMPtr{UInt8,1},UInt64(2048))
    f = GlobalTile(Float32,bytes,@Layout((8,8),(1,8)),16)
    @test eltype(f) === Float32 && Tylo.alignment(f) == 16 && address(f,(1,1)) == 2048 + 36
    # Transposing a tile keeps its memory and exchanges its bounds.
    s = permutedims(bound(t,(Int32(4),nothing)))
    @test size(s) == (5,6) && s.bounds == (nothing,Int32(4))
    @test all(address(s,(c,r)) == address(t,(r,c)) for r in 0:5, c in 0:4)
end

@testset "Thread-free access of one thread's tile" begin
    # One thread's ownership resolves to thread zero at generation time; several threads must pass one.
    @test Tylo._only_thread(Ownership((8,8))) === Int32(0)
    @test_throws ArgumentError Tylo._only_thread(lane_local(4,2))
end

@testset "Windows and tiles are checked against the tile and its bounds" begin
    t = GlobalTile(tptr(BFloat16,4096),Layout((Int32(40),static(24)),(static(1),Int32(40))),16)
    @test_throws BoundsError window(t,(Int32(32),Int32(0)),(16,8))     # past the rows
    @test_throws BoundsError tile(t,(16,8),(Int32(2),Int32(0)))
    @test_throws ArgumentError window(t,(Int32(5),Int32(0)),(16,8))    # breaks the 16-byte declaration
    @test size(window(t,(Int32(24),Int32(16)),(16,8))) == (16,8)
    # A bounded axis may be exceeded: its bound says what exists.
    b = bound(t,(Int32(37),nothing))
    @test window(b,(Int32(32),Int32(0)),(16,8)).bounds == (Int32(5),nothing)
    @test tile(b,(16,8),(Int32(2),Int32(1))).bounds == (Int32(5),nothing)
    @test_throws BoundsError tile(b,(16,8),(Int32(0),Int32(3)))           # unbounded columns
    @test_throws BoundsError window(b,(Int32(-8),Int32(0)),(16,8))
end
