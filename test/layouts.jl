using Tylo.Layouts: Layout, Swizzle, compose, cosize, shape, static
const TL = Tylo.Layouts

@testset "Hierarchical memory layouts" begin
    l = Layout((static(16),static(32)),(static(32),static(1)))
    @test size(l) == (16,32)
    @test cosize(l) == 512
    @test l((Int32(3),Int32(7))) === Int32(103)
    @test l((static(3),static(7))) === static(103)
    mixed = Layout((static(16),Int32(32)),(Int32(40),static(1)))
    @test mixed((Int32(3),Int32(7))) === Int32(127)
    @test cosize(mixed) == 632
    factored = TL.tile(l,Val((8,16)))
    @test shape(factored) == ((8,2),(16,2))
    @test factored(((3,1),(7,1))) == l((11,23))
    @test all(factored(i) == l(i) for i in 0:511)
    coalesced = TL.coalesce(Layout(((static(4),static(2)),static(1),static(3)),
                                  ((static(1),static(4)),static(0),static(8))))
    @test shape(coalesced) == (24,)
    @test all(coalesced(i) == i for i in 0:23)
    @test_throws ArgumentError Layout((16,0),(1,16))
    @test_throws ArgumentError Layout((16,8),(1,-16))
    @test_throws ArgumentError Layout(((4,4),8),(1,16))
    @test_throws ArgumentError TL.tile(l,Val((3,16)))
end

@testset "Swizzles and parent-relative windows" begin
    for sw in (Swizzle{1,3,1}(),Swizzle{2,3,2}(),Swizzle{3,3,3}(),Swizzle{2,3,-2}())
        @test all(sw(sw(Int32(i))) === Int32(i) for i in 0:2047)
        @test all(sw(i) % 8 == i % 8 for i in 0:2047)
    end
    @test_throws ArgumentError Swizzle{2,3,1}() # overlapping fields are not a permutation
    @test_throws ArgumentError Swizzle{1,-1,1}()
    l = compose(Swizzle{2,3,2}(),Layout((static(32),static(32)),(static(32),static(1))))
    @test Set(l(i) for i in 0:1023) == Set(0:1023)
    @test cosize(l) == 1024
    @test l((Int32(1),Int32(0))) === Int32(40)
    # Nonzero origins deliberately include nonzero swizzle phase.
    for r in (0,1,3,8),c in (0,8,16)
        w = TL.window(l,(Int32(r),Int32(c)),Val((16,16)))
        @test all(w((i,j)) == l((i+r,j+c)) for i in 0:15,j in 0:15)
        nested = TL.window(w,(Int32(2),Int32(3)),Val((8,8)))
        @test all(nested((i,j)) == l((i+r+2,j+c+3)) for i in 0:7,j in 0:7)
        @test w((Int32(1),Int32(2))) isa Int32
        @test_throws BoundsError TL.window(w,(12,0),Val((8,8)))
    end
    @test_throws BoundsError TL.window(l,(-1,0),Val((16,16)))
    @test_throws BoundsError TL.window(l,(17,0),Val((16,16)))
    @test_throws ArgumentError TL.window(l,(0,0),Val((0,16)))
end

@testset "Copy vector contracts" begin
    for k in (16,32,64),axis in (1,2)
        s = axis == 1 ? (k,64) : (64,k)
        d = axis == 1 ? (static(1),static(k)) : (static(k),static(1))
        plain = Layout(map(static,s),d)
        swiz = compose(Swizzle{trailing_zeros(k)-3,3,trailing_zeros(k)-3}(),plain)
        src = Layout(s,axis == 1 ? (1,k+8) : (k+8,1))
        plan = CopyPlan{s,128,axis}()
        @test validate_copy(plan,BFloat16,swiz,src) === nothing
        @test validate_copy(plan,Float16,plain,src) === nothing
        @test validate_copy(plan,BFloat16,plain,swiz) === nothing
    end
    plan = CopyPlan{(16,16),32,2}()
    l = Layout((16,16),(16,1))
    @test_throws ArgumentError validate_copy(plan,BFloat16,l,Layout((16,16),(17,1)))
    @test_throws ArgumentError validate_copy(plan,BFloat16,Layout((16,16),(0,1)),l)
    @test_throws ArgumentError validate_copy(plan,BFloat16,l,Layout((16,16),(1,16)))
    @test_throws DimensionMismatch validate_copy(plan,BFloat16,l,Layout((32,16),(16,1)))
    @test_throws ArgumentError CopyPlan{(16,16),7,2}()
    @test_throws ArgumentError CopyPlan{(16,16),32,1.0}()
end

@testset "ISA ownership and typed MMA registers" begin
    atom = MMA16x8x16(BFloat16)
    for role in (OperandA(),OperandB(),Accumulator())
        o = operand_layout(atom,role)
        count = role isa OperandA ? 8 : 4
        expected = [(role isa OperandA ?
            (t÷4 + 8*((e÷2)%2),2*(t%4)+(e%2)+8*(e÷4)) : role isa OperandB ?
            (2*(t%4)+(e%2)+8*(e÷2),t÷4) :
            (t÷4+8*(e÷2),2*(t%4)+(e%2))) for t in 0:31 for e in 0:count-1]
        actual = [TL.coordinate(o,Int32(t),Val(e)) for t in 0:31 for e in 0:count-1]
        @test actual == expected
        @test all(c -> c isa Tuple{Int32,Int32},actual)
        @test Set(actual) == Set((i,j) for i in 0:size(o)[1]-1 for j in 0:size(o)[2]-1)
    end
    a = zero_accumulator(atom)
    @test scale(a,2f0).data == (0f0,0f0,0f0,0f0)
    @test map(x -> x+1f0,a).data == (1f0,1f0,1f0,1f0)
    @test_throws ArgumentError Tylo.MMAFragment(BFloat16,OperandA(),(UInt32(0),UInt32(0)))
    @test_throws ArgumentError Tylo.MMAFragment(Float32,OperandA(),ntuple(_ -> UInt32(0),4))
    @test_throws ArgumentError Tylo.MMAFragment(BFloat16,Accumulator(),(0f0,0f0,0f0,0f0))
    @test_throws ArgumentError MMA16x8x16(Float32)
    p = TiledMMA(atom,Val((2,2)),Val((2,4)),Val(32))
    @test size(p) == (64,64,32)
    @test Tylo.threads(p) == 128
    @test length(zero_accumulator(p).data) == 8
    @test all(f -> f.data == (1f0,1f0,1f0,1f0),map(x -> x+1f0,zero_accumulator(p)).data)
    @test_throws ArgumentError TiledMMA(atom,Val((2,2)),Val((2,4)),Val(24))
end

@testset "Address-space units" begin
    l = Layout((Int32(65536),Int32(65536)),(Int32(65536),static(1)))
    @test Tylo._byte_offset(Val(1),BFloat16,l,(Int32(65535),Int32(65535))) == 2*(Int64(65536)^2-1)
    small = Layout((static(16),static(32)),(static(32),static(1)))
    @test Tylo._byte_offset(Val(3),BFloat16,small,(Int32(2),Int32(3))) === Int32(134)
    @test Tylo._byte_offset(Val(3),BFloat16,small,(static(2),static(3))) == 134
end
