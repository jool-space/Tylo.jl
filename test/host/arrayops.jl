using Tylo.Layouts: @Layout, Layout, shape
arrayop_select(f) = ifelse.(f .> 0f0,f,-1f0)
arrayop_roundtrip(f) = Float32.(Float16.(f))
arrayop_rows(f,r) = ifelse.(r .> 1f0,f .+ r,-1f0)
arrayop_polynomial(x) = @. (x + 2f0)^2 / 3f0
flat_registers(x) = x.data
flat_registers(x::Tylo.PermutedFragment) = flat_registers(parent(x))

@testset "Julia fragment arithmetic" begin
    atom = MMAAtom((16,8,16),BFloat16)
    plan = TiledMMA(atom,Val((2,1)),Val((2,3)),Val(16))
    fragments = (local_fragment((-2f0,0f0,3f0)),striped_fragment((-2f0,0f0,3f0)),
        Fragment((-2f0,0f0,3f0,4f0),operand_layout(atom,Accumulator())),
        Fragment(Tuple(v for i in 1:6 for v in (Float32(i),Float32(-i),0f0,0.5f0)),Tylo.Layouts.layout(zero_accumulator(plan))),
        Fragment((-2f0,0f0,3f0,4f0),Tylo._reduced_ownership(Tylo.Layouts.layout(zero_accumulator(plan)),Val(2))))
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
    f = local_fragment((-2f0,0f0,3f0))
    # The result type depends on the axis; Val states it for inference.
    @test only(@inferred sum(f;dims=Val(2))) === 1f0
    @test only(maximum(f;dims=(2,))) === 3f0
    @test only(@inferred minimum(f;dims=Val(2))) === -2f0
    @test only(sum(f;dims=2)) === 1f0
    @test (f .- maximum(f;dims=2)).data == (-5f0,-3f0,0f0)
    for dims in (:,0,3,(),(1,2),(2,2),2.0,(2.0,))
        @test_throws ArgumentError sum(f;dims)
    end
    # Reducing the lane axis is a warp collective; its plan exists, and GPU tests run it.
    @test Tylo.reduction_plan(Tylo.Layouts.layout(f),1).bits == [0,1,2,3,4]
    @test_throws ArgumentError sum(f)
    @test_throws ArgumentError sum(Float64.(f);dims=2)
    @test isequal(only(maximum(local_fragment((1f0,NaN32));dims=2)),NaN32)
    @test only(maximum(local_fragment((-Inf32,-Inf32));dims=2)) === -Inf32

    atom = MMAAtom((16,8,16),BFloat16)
    for wm in (1,2),rm in (1,2),rn in (1,3)
        p = TiledMMA(atom,Val((wm,1)),Val((rm,rn)),Val(16))
        f = zero_accumulator(p)
        r = Fragment(ntuple(i -> Float32(i),2rm),Tylo._reduced_ownership(Tylo.Layouts.layout(f),Val(2)))
        @test flat_registers(@inferred f .+ r) == Tuple(Float32(2i-1+h) for j in 1:rn for i in 1:rm for h in (0,0,1,1))
        @test flat_registers(@inferred r .- f) == Tuple(Float32(2i-1+h) for j in 1:rn for i in 1:rm for h in (0,0,1,1))
        @test flat_registers(@inferred arrayop_rows(f,r)) ==
            Tuple(2i-1+h > 1 ? Float32(2i-1+h) : -1f0 for j in 1:rn for i in 1:rm for h in (0,0,1,1))
        @test typeof(@inferred (f .+ r) .* 2f0) === typeof(f)
        @test flat_registers(@inferred broadcast(max,r,2f0)) == max.(r.data,2f0)
        @test_throws DimensionMismatch map(+,f,r)
    end
    r = Fragment((2f0,),Tylo._reduced_ownership(Tylo.Layouts.StripedOwnership{2,2}(),Val(2)))
    @test (striped_fragment((1f0,3f0)) ./ r).data == (0.5f0,1.5f0)
    # A warp-replicated scalar broadcasts into a warp; f spans two warps here.
    @test_throws DimensionMismatch f .+ r
    @test flat_registers(zero_accumulator(atom) .+ r) == ntuple(_ -> 2f0,4)
    @test (local_fragment((1f0,2f0)) .+ local_fragment((1f0,))).data == (2f0,3f0)
    @test_throws DimensionMismatch local_fragment((1f0,2f0)) .+ striped_fragment((1f0,2f0))
    @test (Fragment((1f0,),Tylo._reduced_ownership(Tylo.Layouts.LocalOwnership{2,2}(),Val(2))) .+ r).data == (3f0,)
    @test_throws DimensionMismatch striped_fragment((1f0,2f0)) .+ Fragment((1f0,),Tylo._reduced_ownership(Tylo.Layouts.LocalOwnership{2,2}(),Val(2)))

    # Same logical shape, different lane ownership: no automatic redistribution.
    a = zero_accumulator(TiledMMA(atom,Val((2,1)),Val((1,2)),Val(16)))
    b = zero_accumulator(TiledMMA(atom,Val((1,2)),Val((2,1)),Val(16)))
    @test size(Tylo.Layouts.layout(a)) == size(Tylo.Layouts.layout(b))
    @test_throws DimensionMismatch a .+ b
    @test_throws ArgumentError maximum(b;dims=2)
    other = Tylo._reduced_ownership(Tylo.Layouts.layout(zero_accumulator(TiledMMA(atom,Val((2,1)),Val((1,1)),Val(16)))),Val(2))
    @test_throws DimensionMismatch b .+ Fragment((1f0,2f0),other)
    @test flat_registers(b .+ 1f0) == ntuple(_ -> 1f0,8)
    c = zero_accumulator(TiledMMA(atom,Val((2,1)),Val((1,2)),Val(32)))
    @test flat_registers(a .+ c) == ntuple(_ -> 0f0,8)
end

@testset "Broadcast fusion and immutable results" begin
    f = local_fragment((1f0,2f0,3f0))
    calls = Ref(0)
    counted(x) = (calls[] += 1; x+1f0)
    result = counted.(f) .* counted.(f)
    @test calls[] == 6
    @test result.data == (4f0,9f0,16f0)
    @test f.data == (1f0,2f0,3f0)
end

@testset "Logical axis permutations" begin
    f = local_fragment((-2f0,0f0,3f0))
    g = @inferred permutedims(f)
    @test parent(g) === f
    @test (@inferred permutedims(g)) === f
    @test permutedims(f,(1,2)) === f
    @test permutedims(g,(1,2)) === g
    @test sizeof(g) == sizeof(f)
    @test eltype(g) === Float32
    @test size(Tylo.Layouts.layout(g)) == (3,32)
    r = @inferred maximum(g;dims=Val(1))
    @test r === permutedims(maximum(f;dims=2))
    @test size(Tylo.Layouts.layout(r)) == (1,32)
    @test sum(g;dims=(1,)) === permutedims(sum(f;dims=2))
    @test only(minimum(g;dims=1)) === -2f0
    @test (@inferred g .- r) === permutedims(f .- maximum(f;dims=2))
    @test (@inferred r .- g) === permutedims(maximum(f;dims=2) .- f)
    @test (@inferred map(+,g,g)) === permutedims(map(+,f,f))
    @test (@inferred map(exp,g)) === permutedims(map(exp,f))
    @test (@inferred arrayop_select(g)) === permutedims(arrayop_select(f))
    @test (@inferred broadcast(exp,g .- r)) === permutedims(exp.(f .- maximum(f;dims=2)))
    @test Tylo.reduction_plan(Tylo.Layouts.layout(g),2).bits == [0,1,2,3,4]
    @test_throws ArgumentError sum(g)
    @test_throws DimensionMismatch f .+ g
    @test_throws DimensionMismatch g .+ maximum(f;dims=2)
    @test_throws DimensionMismatch f .+ r
    for perm in ((1,1),(0,2),(2,3),(1,),(),(2.0,1.0))
        @test_throws ArgumentError permutedims(f,perm)
    end

    atom = MMAAtom((16,8,16),BFloat16)
    p = TiledMMA(atom,Val((2,1)),Val((2,3)),Val(16))
    for f in (f,striped_fragment((1f0,2f0,3f0)),zero_accumulator(atom),zero_accumulator(p))
        l = Tylo.Layouts.layout(f)
        g = permutedims(f)
        gl = Tylo.Layouts.layout(g)
        @test size(gl) == reverse(size(l))
        threads = Tylo._thread_count(l)
        for t in 0:threads-1,e in 0:Tylo._local_count(typeof(f))-1
            @test Tylo.Layouts.coordinate(gl,t,Val(e)) == reverse(Tylo.Layouts.coordinate(l,t,Val(e)))
        end
        reduced = Tylo._reduced_ownership(Tylo.Layouts.layout(f),Val(2))
        rv = Fragment(ntuple(Float32,Tylo._register_count(reduced)),reduced)
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
    # Each row is shared by the four lanes differing in bits 3 and 4: a derived
    # recipe exists (two xor shuffles), even though no hand-written one did.
    @test Tylo.reduction_plan(o,2).bits == [3,4]
    @test Tylo.reduction_plan(o,2).groups == [[1,3],[2,4]]
    @test Tylo.reduction_plan(Tylo.PermutedOwnership(o),1).bits == [3,4]
    @test_throws DimensionMismatch Fragment((1f0,),o)
    @test local_fragment((1f0,2f0)) === Fragment((1f0,2f0),Tylo.Layouts.LocalOwnership{2,2}())
    @test striped_fragment((1f0,2f0)) isa Fragment
    @test sum(local_fragment((1f0,2f0));dims=2) isa Fragment

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

@testset "Explicit ownership axes and canonical broadcast matching" begin
    for axis in (1,2)
        f = Fragment((1f0,2f0,3f0,4f0),Tylo.Layouts.LocalOwnership{4,axis}())
        g = permutedims(f)
        direct = Fragment(f.data,Tylo.Layouts.LocalOwnership{4,3-axis}())
        @test (g .+ direct).data === 2f0 .* f.data
        @test only(sum(g;dims=3-axis)) === 10f0
        @test (g .- maximum(g;dims=3-axis)).data === (-3f0,-2f0,-1f0,0f0)
        for T in (BFloat16,Float16)
            packed = permutedims(pack(T,f))
            shape = axis == 2 ? (4,32) : (32,4)
            strides = axis == 2 ? (128,1) : (1,128)
            tile = TmemTile(T,UInt32(0),Tylo.Layouts.Layout(shape,strides))
            access = partition(TmemTransfer{shape,3-axis}(),tile)
            @test Tylo._check_tmem_store(access,packed) === nothing
        end
    end
end
