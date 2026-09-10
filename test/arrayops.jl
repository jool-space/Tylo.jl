using Tylo.Layouts: @Layout, Layout, shape
arrayop_select(f) = ifelse.(f .> 0f0,f,-1f0)
arrayop_roundtrip(f) = Float32.(Float16.(f))
arrayop_rows(f,r) = ifelse.(r .> 1f0,f .+ r,-1f0)
arrayop_polynomial(x) = @. (x + 2f0)^2 / 3f0
flat_registers(x) = x.data
flat_registers(x::Tylo.PermutedFragment) = flat_registers(parent(x))
flat_registers(x::Tylo.MMAAccumulator) = Tuple(v for atom in x.data for v in atom.data)

@testset "Julia fragment arithmetic" begin
    atom = MMA16x8x16(BFloat16)
    plan = TiledMMA(atom,Val((2,1)),Val((2,3)),Val(16))
    fragments = (RowFragment((-2f0,0f0,3f0)),WarpRowFragment((-2f0,0f0,3f0)),
        Tylo.MMAFragment(Float32,Accumulator(),(-2f0,0f0,3f0,4f0)),
        Tylo.MMAAccumulator(plan,ntuple(i -> Tylo.MMAFragment(Float32,Accumulator(),
            (Float32(i),Float32(-i),0f0,0.5f0)),6)),
        RowValues(Tylo.MMARowOwnership{2,2}(),(-2f0,0f0,3f0,4f0)))
    for f in fragments
        data = flat_registers(f)
        @test eltype(f) === Float32
        @test flat_registers(@inferred broadcast(exp,f)) == exp.(data)
        @test flat_registers(@inferred f .+ 2f0) == data .+ 2f0
        @test flat_registers(@inferred 2f0 .- f) == 2f0 .- data
        @test flat_registers(@inferred arrayop_polynomial(f)) == arrayop_polynomial(data)
        @test flat_registers(@inferred map(+,f,f)) == data .+ data
        @test flat_registers(@inferred map(muladd,f,f,f)) == muladd.(data,data,data)
        mask = @inferred f .> 0f0
        @test eltype(mask) === Bool
        @test flat_registers(mask) == (data .> 0f0)
        @test flat_registers(@inferred broadcast(ifelse,mask,f,-1f0)) == ifelse.(data .> 0f0,data,-1f0)
        @test flat_registers(@inferred arrayop_select(f)) == flat_registers(ifelse.(mask,f,-1f0))
        wide = @inferred f .+ 1.0
        @test eltype(wide) === Float64 # ordinary Julia promotion, no implicit narrowing
        @test flat_registers(wide) == data .+ 1.0
        @test eltype(@inferred broadcast(Float16,f)) === Float16
        @test typeof(@inferred arrayop_roundtrip(f)) === typeof(f)
        @test flat_registers(@inferred map(Float64,f)) == Float64.(data)
        @test flat_registers(@inferred broadcast((x,r) -> x+r[1],f,Ref((2f0,)))) == data .+ 2f0
        @test flat_registers(@inferred broadcast(clamp,f,-1f0,1f0)) == clamp.(data,-1f0,1f0)
    end
end

@testset "Reduction axes and row broadcasting" begin
    f = RowFragment((-2f0,0f0,3f0))
    @test (@inferred sum(f;dims=2)) === row_sum(f)
    @test (@inferred maximum(f;dims=(2,))) === row_max(f)
    @test only(@inferred minimum(f;dims=2)) === -2f0
    @test (f .- maximum(f;dims=2)).data == (-5f0,-3f0,0f0)
    for dims in (:,1,0,3,(),(1,2),(2,2),2.0,(2.0,))
        @test_throws ArgumentError sum(f;dims)
    end
    @test_throws ArgumentError sum(f)
    @test_throws ArgumentError sum(Float64.(f);dims=2)
    @test isequal(only(maximum(RowFragment((1f0,NaN32));dims=2)),NaN32)
    @test only(maximum(RowFragment((-Inf32,-Inf32));dims=2)) === -Inf32

    atom = MMA16x8x16(BFloat16)
    for wm in (1,2),rm in (1,2),rn in (1,3)
        p = TiledMMA(atom,Val((wm,1)),Val((rm,rn)),Val(16))
        f = zero_accumulator(p)
        r = RowValues(row_ownership(f),ntuple(i -> Float32(i),2rm))
        @test flat_registers(@inferred f .+ r) == flat_registers(row_map(+,f,r))
        @test flat_registers(@inferred r .- f) == flat_registers(row_map((x,y) -> y-x,f,r))
        @test flat_registers(@inferred arrayop_rows(f,r)) ==
            flat_registers(row_map((x,y) -> y>1f0 ? x+y : -1f0,f,r))
        @test typeof(@inferred (f .+ r) .* 2f0) === typeof(f)
        @test flat_registers(@inferred broadcast(max,r,2f0)) == max.(r.data,2f0)
        @test_throws DimensionMismatch map(+,f,r)
    end
    r = RowValues(Tylo.WarpRowOwnership(),(2f0,))
    @test (WarpRowFragment((1f0,3f0)) ./ r).data == (0.5f0,1.5f0)
    @test_throws DimensionMismatch f .+ r
    @test_throws DimensionMismatch RowFragment((1f0,2f0)) .+ RowFragment((1f0,))
    @test_throws DimensionMismatch RowFragment((1f0,2f0)) .+ WarpRowFragment((1f0,2f0))
    @test_throws DimensionMismatch RowValues(Tylo.LaneRowOwnership(),(1f0,)) .+ r

    # Same logical shape, different lane ownership: no automatic redistribution.
    a = zero_accumulator(TiledMMA(atom,Val((2,1)),Val((1,2)),Val(16)))
    b = zero_accumulator(TiledMMA(atom,Val((1,2)),Val((2,1)),Val(16)))
    @test size(Tylo.Layouts.layout(a)) == size(Tylo.Layouts.layout(b))
    @test_throws DimensionMismatch a .+ b
    @test_throws ArgumentError maximum(b;dims=2)
    @test_throws ArgumentError b .+ RowValues(Tylo.MMARowOwnership{1,2}(),(1f0,2f0,3f0,4f0))
    @test flat_registers(b .+ 1f0) == ntuple(_ -> 1f0,8)
    c = zero_accumulator(TiledMMA(atom,Val((2,1)),Val((1,2)),Val(32)))
    @test flat_registers(a .+ c) == ntuple(_ -> 0f0,8)
end

@testset "Broadcast fusion and immutable results" begin
    f = RowFragment((1f0,2f0,3f0))
    calls = Ref(0)
    counted(x) = (calls[] += 1; x+1f0)
    result = counted.(f) .* counted.(f)
    @test calls[] == 6
    @test result.data == (4f0,9f0,16f0)
    @test f.data == (1f0,2f0,3f0)
end

@testset "Logical axis permutations" begin
    f = RowFragment((-2f0,0f0,3f0))
    g = @inferred permutedims(f)
    @test parent(g) === f
    @test (@inferred permutedims(g)) === f
    @test permutedims(f,(1,2)) === f
    @test permutedims(g,(1,2)) === g
    @test sizeof(g) == sizeof(f)
    @test eltype(g) === Float32
    @test size(Tylo.Layouts.layout(g)) == (3,32)
    r = @inferred maximum(g;dims=1)
    @test r === permutedims(maximum(f;dims=2))
    @test size(Tylo.Layouts.layout(r)) == (1,32)
    @test (@inferred sum(g;dims=(1,))) === permutedims(sum(f;dims=2))
    @test only(minimum(g;dims=1)) === -2f0
    @test (@inferred g .- r) === permutedims(f .- maximum(f;dims=2))
    @test (@inferred r .- g) === permutedims(maximum(f;dims=2) .- f)
    @test (@inferred map(+,g,g)) === permutedims(map(+,f,f))
    @test (@inferred map(exp,g)) === permutedims(map(exp,f))
    @test (@inferred arrayop_select(g)) === permutedims(arrayop_select(f))
    @test (@inferred broadcast(exp,g .- r)) === permutedims(exp.(f .- maximum(f;dims=2)))
    @test_throws ArgumentError sum(g;dims=2)
    @test_throws ArgumentError sum(g)
    @test_throws DimensionMismatch f .+ g
    @test_throws DimensionMismatch g .+ maximum(f;dims=2)
    @test_throws DimensionMismatch f .+ r
    for perm in ((1,1),(0,2),(2,3),(1,),(),(2.0,1.0))
        @test_throws ArgumentError permutedims(f,perm)
    end

    atom = MMA16x8x16(BFloat16)
    p = TiledMMA(atom,Val((2,1)),Val((2,3)),Val(16))
    for f in (f,WarpRowFragment((1f0,2f0,3f0)),zero_accumulator(atom),zero_accumulator(p))
        l = Tylo.Layouts.layout(f)
        g = permutedims(f)
        gl = Tylo.Layouts.layout(g)
        @test size(gl) == reverse(size(l))
        threads = f isa Tylo.MMAAccumulator ? 64 : 32
        for t in 0:threads-1,e in 0:Tylo._local_count(typeof(f))-1
            @test Tylo.Layouts.coordinate(gl,t,Val(e)) == reverse(Tylo.Layouts.coordinate(l,t,Val(e)))
        end
        rv = RowValues(row_ownership(f),ntuple(Float32,Tylo._row_count(row_ownership(f))))
        @test parent(g .+ permutedims(rv)) === Fragment(f .+ rv)
    end

    # Reverse only the top-level modes, including when a mode is hierarchical.
    for l in (@Layout((3,5),(5,1)), @Layout(((2,3),5),((1,2),6)))
        q = Layout(reverse(shape(l)),reverse(strides(l)))
        coords = first(shape(l)) isa Tuple ? [((i,j),k) for i in 0:1 for j in 0:2 for k in 0:4] :
            [(i,j) for i in 0:2 for j in 0:4]
        for c in coords
            @test l(c) == q(reverse(c))
        end
    end
end

@testset "Fragments with general ownership" begin
    # Every lane holds a 2×2 patch; neither a whole row nor adjacent columns.
    o = Tylo.Layouts.Ownership(Val((16,8)),@Layout(((8,4),(2,2)),((2,32),(1,16))))
    f = @inferred Fragment((1f0,-2f0,3f0,-4f0),o)
    @test f isa Fragment
    @test Tylo.Layouts.layout(f) === o
    @test [Tylo.Layouts.coordinate(o,0,Val(e)) for e in 0:3] == [(0,0),(1,0),(0,1),(1,1)]
    @test (f .+ 2f0).data == (3f0,0f0,5f0,-2f0)
    @test map(abs,f).data == (1f0,2f0,3f0,4f0)
    @test Tylo.Layouts.layout(ifelse.(f .> 0f0,f,0f0)) === o
    @test permutedims(permutedims(f)) === f
    @test_throws ArgumentError sum(f;dims=2)
    @test_throws ArgumentError sum(permutedims(f);dims=1)
    @test_throws DimensionMismatch Fragment((1f0,),o)
    @test RowFragment((1f0,2f0)) === Fragment((1f0,2f0),Tylo.Layouts.LaneRows{2}())
    @test WarpRowFragment((1f0,2f0)) isa Fragment
    @test sum(RowFragment((1f0,2f0));dims=2) isa Fragment

    # Runtime stride values matter even when ownership types are identical.
    a = Tylo.Layouts.Ownership(Val((16,8)),Layout((32,4),(4,1)))
    b = Tylo.Layouts.Ownership(Val((16,8)),Layout((32,4),(1,32)))
    @test typeof(a) === typeof(b)
    fa,fb = Fragment(f.data,a),Fragment(f.data,b)
    @test_throws DimensionMismatch fa .+ fb
    @test_throws DimensionMismatch map(+,fa,fb)
    @test (fa .+ Fragment(f.data,a)).data == 2f0 .* f.data
    @test isempty(Test.detect_ambiguities(Tylo;recursive=true))
end
