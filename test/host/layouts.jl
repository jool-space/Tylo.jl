using Tylo.Layouts: @Layout, Layout, Swizzle, compose, cosize, shape, static
const TL = Tylo.Layouts

@testset "Hierarchical memory layouts" begin
    l = @Layout (16, 32) (32, 1)
    @test size(l) == (16,32)
    @test cosize(l) == 512
    @test l((Int32(3),Int32(7))) === Int32(103)
    @test l((static(3),static(7))) === static(103)
    mixed = @Layout (16, $(Int32(32))) ($(Int32(40)), 1)
    @test mixed((Int32(3),Int32(7))) === Int32(127)
    @test cosize(mixed) == 632
    factored = TL.tile(l,(8,16))
    @test shape(factored) == ((8,2),(16,2))
    @test factored(((3,1),(7,1))) == l((11,23))
    @test all(factored(i) == l(i) for i in 0:511)
    coalesced = TL.coalesce(@Layout(((4, 2), 1, 3), ((1, 4), 0, 8)))
    @test shape(coalesced) == (24,)
    @test all(coalesced(i) == i for i in 0:23)
    @test_throws ArgumentError Layout((16,0),(1,16))
    @test_throws ArgumentError Layout((16,8),(1,-16))
    @test_throws ArgumentError Layout(((4,4),8),(1,16))
    @test_throws ArgumentError TL.tile(l,(3,16))
end

@testset "Swizzles and parent-relative windows" begin
    for sw in (Swizzle{1,3,1}(),Swizzle{2,3,2}(),Swizzle{3,3,3}(),Swizzle{2,3,-2}())
        @test all(sw(sw(Int32(i))) === Int32(i) for i in 0:2047)
        @test all(sw(i) % 8 == i % 8 for i in 0:2047)
    end
    @test_throws ArgumentError Swizzle{2,3,1}() # overlapping fields are not a permutation
    @test_throws ArgumentError Swizzle{1,-1,1}()
    l = compose(Swizzle{2,3,2}(),@Layout((32, 32), (32, 1)))
    @test Set(l(i) for i in 0:1023) == Set(0:1023)
    @test cosize(l) == 1024
    @test l((Int32(1),Int32(0))) === Int32(40)
    # Nonzero origins deliberately include nonzero swizzle phase.
    for r in (0,1,3,8),c in (0,8,16)
        w = TL.window(l,(Int32(r),Int32(c)),(16,16))
        @test all(w((i,j)) == l((i+r,j+c)) for i in 0:15,j in 0:15)
        nested = TL.window(w,(Int32(2),Int32(3)),(8,8))
        @test all(nested((i,j)) == l((i+r+2,j+c+3)) for i in 0:7,j in 0:7)
        @test w((Int32(1),Int32(2))) isa Int32
        @test_throws BoundsError TL.window(w,(12,0),(8,8))
    end
    @test_throws BoundsError TL.window(l,(-1,0),(16,16))
    @test_throws BoundsError TL.window(l,(17,0),(16,16))
    @test_throws ArgumentError TL.window(l,(0,0),(0,16))
end

@testset "Copy vectors" begin
    type(l) = typeof(l)
    for k in (16,32,64),axis in (1,2)
        s = axis == 1 ? (k,64) : (64,k)
        d = axis == 1 ? (1,k) : (k,1)
        plain = @Layout s d
        swiz = compose(Swizzle{trailing_zeros(k)-3,3,trailing_zeros(k)-3}(),plain)
        # Runtime leading dimension, static unit stride: vectors are provable.
        ld = k+8
        src = axis == 1 ? @Layout($s,(1,$ld)) : @Layout($s,($ld,1))
        o = distribute(s,128;vector=8,axis)
        # One 16-byte cp.async per vector, through the swizzle and runtime strides.
        for (dst,T) in ((swiz,BFloat16),(plain,Float16))
            plan = Tylo.copy_plan(o,T,type(dst),16,type(src),16)
            @test plan.width == 16 && length(plan.groups) == Tylo._register_count(o) ÷ 8
        end
        # A source declared element-aligned has no vectors; scalars remain.
        @test Tylo.copy_plan(o,BFloat16,type(swiz),16,type(src),2) === nothing
        @test Tylo.copy_plan(o,BFloat16,type(swiz),16,type(src),8).width == 8
    end
    # Contiguous along different axes: no common vector.
    o = distribute((16,16),32;vector=8,axis=2)
    @test Tylo.copy_plan(o,BFloat16,type(@Layout((16,16),(16,1))),16,type(@Layout((16,16),(1,16))),16) === nothing
    # 32 vectors for 7 threads: one thread holds them all, six replicate.
    @test Tylo.replica_mask(distribute((16,16),8;vector=8,axis=2)) == 0
    @test Tylo._register_count(distribute((16,24),32;vector=8,axis=1)) == 24   # 48 vectors: 16 threads hold 3
    @test Tylo.replica_mask(distribute((16,24),32;vector=8,axis=1)) == 16
    @test_throws ArgumentError distribute((16,16),32;vector=3,axis=2)
    @test_throws ArgumentError distribute((16,16),32;vector=8,axis=3)
    # More threads than vectors: the rest replicate, and store once.
    @test Tylo.replica_mask(distribute((8,64),128;vector=8,axis=2)) == 64
end

@testset "ISA ownership and typed MMA registers" begin
    atom = MMAAtom((16,8,16),BFloat16)
    for role in (OperandA(),OperandB(),Accumulator())
        o = operand_layout(atom,role)
        count = role isa OperandA ? 8 : 4
        expected = [(role isa OperandA ?
            (t÷4 + 8*((e÷2)%2),2*(t%4)+(e%2)+8*(e÷4)) : role isa OperandB ?
            (2*(t%4)+(e%2)+8*(e÷2),t÷4) :
            (t÷4+8*(e÷2),2*(t%4)+(e%2))) for t in 0:31 for e in 0:count-1]
        actual = [TL.coordinate(o,Int32(t),e) for t in 0:31 for e in 0:count-1]
        @test actual == expected
        @test all(c -> c isa Tuple{Int32,Int32},actual)
        @test Set(actual) == Set((i,j) for i in 0:size(o)[1]-1 for j in 0:size(o)[2]-1)
    end
    a = zero_accumulator(atom)
    @test (a .* 2f0).data == (0f0,0f0,0f0,0f0)
    @test map(x -> x+1f0,a).data == (1f0,1f0,1f0,1f0)
    @test Tylo._words(atom,OperandA()) == 4 && Tylo._words(atom,OperandB()) == 2
    @test_throws DimensionMismatch Fragment((0f0,0f0),operand_layout(atom,Accumulator()))
    @test_throws ArgumentError MMAAtom((16,8,32),BFloat16)
    @test_throws ArgumentError MMAAtom((16,8,16),Float32)
    @test_throws ArgumentError MMAAtom((16,8,16),BFloat16,Float16)
    @test MMAAtom((16,8,8),Float32) isa MMAAtom
    p = TiledMMA(atom,(2,2),(2,4),32)
    @test size(p) == (64,64,32)
    @test Tylo.threads(p) == 128
    @test length(zero_accumulator(p).data) == 32
    @test all(==(1f0),map(x -> x+1f0,zero_accumulator(p)).data)
    @test_throws ArgumentError TiledMMA(atom,(2,2),(2,4),24)
end

@testset "Address-space units" begin
    l = @Layout ($(Int32(65536)), $(Int32(65536))) ($(Int32(65536)), 1)
    @test Tylo._byte_offset(Val(1),BFloat16,l,(Int32(65535),Int32(65535))) == 2*(Int64(65536)^2-1)
    small = @Layout (16, 32) (32, 1)
    @test Tylo._byte_offset(Val(3),BFloat16,small,(Int32(2),Int32(3))) === Int32(134)
    @test Tylo._byte_offset(Val(3),BFloat16,small,(static(2),static(3))) == 134
end

@testset "Display in constructor syntax" begin
    @test repr(@Layout((4, 8), (8, 1))) == "@Layout (4, 8) (8, 1)"
    @test repr(Layout((Int32(10), static(8)), (static(1), Int32(16)))) == "@Layout (\$10, 8) (1, \$16)"
    @test repr(Ownership((2, 4))) == "Ownership((2, 4), @Layout (1, 8) (0, 1))"
    @test repr(Fragment((1f0, 2f0), Ownership((2, 1)))) == "Fragment((1.0f0, 2.0f0), Ownership((2, 1), @Layout (1, 2) (0, 1)))"
    @test repr(compose(Swizzle{3,3,3}(), @Layout((8, 64), (64, 1)))) == "compose(Swizzle{3, 3, 3}(), @Layout (8, 64) (64, 1))"
end
