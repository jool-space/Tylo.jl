@testset "Row ownership and local arithmetic" begin
    for n in (1,3,17,64)
        data = ntuple(i -> Float32(i-9),n)
        f = RowFragment(data)
        @test only(row_sum(f)) == sum(data)
        @test only(row_max(f)) == maximum(data)
        @test row_map(-,f,row_max(f)).data == data .- maximum(data)
        for t in 0:31
            @test row_coordinate(row_sum(f),t,Val(0)) == t
        end
        w = WarpRowFragment(data)
        @test size(Tylo.Layouts.layout(w)) == (1,32n)
        @test [Tylo.Layouts.coordinate(Tylo.Layouts.layout(w),t,Val(e))
               for t in 0:31,e in 0:n-1] == [(0,t+32e) for t in 0:31,e in 0:n-1]
        r = RowValues(row_ownership(w),(3f0,))
        @test row_map(*,w,r).data == data .* 3f0
        @test row_coordinate(r,Int32(67),Val(0)) === Int32(2)
        @test_throws MethodError row_map(+,f,r)
    end
    atom = MMA16x8x16(BFloat16)
    for wm in (1,2),rm in (1,2),rn in (1,3)
        plan = TiledMMA(atom,Val((wm,1)),Val((rm,rn)),Val(16))
        f = zero_accumulator(plan)
        r = RowValues(row_ownership(f),ntuple(i -> Float32(i),2rm))
        actual = row_map(+,f,r)
        # Each atom has two rows/lane and two values/row. Column repeats
        # reuse a row result, and M repeats select distinct results.
        @test all(actual.data[i+(j-1)*rm].data == (Float32(2i-1),Float32(2i-1),Float32(2i),Float32(2i))
                  for i in 1:rm,j in 1:rn)
        @test [row_coordinate(r,t,Val(e)) for t in 0:32wm-1,e in 0:2rm-1] ==
              [16rm*(t÷32)+(t%32)÷4+8*(e%2)+16*(e÷2) for t in 0:32wm-1,e in 0:2rm-1]
        counts = Dict{Int,Int}()
        for t in 0:32wm-1,e in 0:2rm-1
            row = row_coordinate(r,t,Val(e)); counts[row] = get(counts,row,0)+1
        end
        @test sort!(collect(keys(counts))) == collect(0:16wm*rm-1)
        @test all(==(4),values(counts))
    end
    @test_throws ArgumentError row_ownership(TiledMMA(atom,Val((1,2)),Val((1,1)),Val(16)))
    @test_throws DimensionMismatch RowValues(Tylo.LaneRowOwnership(),(1f0,2f0))
    @test_throws ArgumentError WarpRowFragment(())
    @test_throws BoundsError row_coordinate(row_sum(RowFragment((1f0,))),0,Val(1))
end

@testset "Tiled accumulator coordinates" begin
    for wm in (1,2),wn in (1,2),rm in (1,2),rn in (1,3)
        p=TiledMMA(MMA16x8x16(BFloat16),Val((wm,wn)),Val((rm,rn)),Val(16))
        layout=Tylo.Layouts.layout(zero_accumulator(p))
        m,n=16wm*rm,8wn*rn
        coords=[Tylo.Layouts.coordinate(layout,Int32(t),Val(e)) for t in 0:32wm*wn-1,e in 0:4rm*rn-1]
        oracle=[(16rm*((t÷32)%wm)+16*((e÷4)%rm)+(t%32)÷4+8*((e%4)÷2),
                 8rn*((t÷32)÷wm)+8*((e÷4)÷rm)+2*(t%4)+(e%2))
                for t in 0:32wm*wn-1,e in 0:4rm*rn-1]
        @test coords == oracle
        @test Set(coords) == Set((r,c) for r in 0:m-1,c in 0:n-1)
        @test length(unique(coords)) == m*n
        @test size(layout) == (m,n)
        @test eltype(coords) == Tuple{Int32,Int32}
    end
end
