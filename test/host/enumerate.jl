using Tylo.Layouts: @Layout, coordinate, LocalOwnership, StripedOwnership, Ownership

# Independent ISA figures for the m16n8k16 warp MMA and the 64×N WGMMA
# accumulator. These do not go through Tylo's coordinate methods.
isa_a(t,e) = (t÷4 + 8*((e÷2)%2), 2*(t%4) + e%2 + 8*(e÷4))
isa_b(t,e) = (2*(t%4) + e%2 + 8*(e÷2), t÷4)
isa_c(t,e) = (t÷4 + 8*(e÷2), 2*(t%4) + e%2)
isa_wgmma_c(t,e) = (16*(t÷32) + (t%32)÷4 + 8*((e%4)÷2), 2*(t%4) + e%2 + 8*(e÷4))

@testset "Ownership tables" begin
    atom = MMAAtom((16,8,16),BFloat16)
    @test Tylo.ownership_table(operand_layout(atom,OperandA())) == [isa_a(t,e) for t in 0:31, e in 0:7]
    @test Tylo.ownership_table(operand_layout(atom,OperandB())) == [isa_b(t,e) for t in 0:31, e in 0:3]
    @test Tylo.ownership_table(operand_layout(atom,Accumulator())) == [isa_c(t,e) for t in 0:31, e in 0:3]
    for n in (8,16,64)
        @test Tylo.ownership_table(Tylo.WGMMAOwnership{n}()) == [isa_wgmma_c(t,e) for t in 0:127, e in 0:n÷2-1]
    end
    @test Tylo.ownership_table(LocalOwnership{3,2}()) == [(t,e) for t in 0:31, e in 0:2]
    @test Tylo.ownership_table(StripedOwnership{2,1}()) == [(t+32e,0) for t in 0:31, e in 0:1]
    @test Tylo.ownership_table(Tylo.TmemTransfer{(32,4),2}()) == [(t,e) for t in 0:31, e in 0:3]
    @test Tylo.ownership_table(Tylo.PermutedOwnership(LocalOwnership{3,2}())) == [(e,t) for t in 0:31, e in 0:2]
    for o in (operand_layout(atom,OperandA()),operand_layout(atom,Accumulator()),
              LocalOwnership{4,2}(),StripedOwnership{2,2}(),Tylo.WGMMAOwnership{16}(),
              Tylo.Layouts.layout(zero_accumulator(TiledMMA(atom,Val((2,2)),Val((2,3)),Val(16)))))
        @test Tylo.is_complete(o)
        @test Tylo.is_injective(o)
        @test Tylo.same_distribution(o,o)
    end
    @test !Tylo.is_injective(Tylo.reduction_plan(StripedOwnership{2,2}(),2).result)
    @test Tylo.same_distribution(LocalOwnership{4,2}(),Tylo.TmemTransfer{(32,4),2}())
    @test !Tylo.same_distribution(LocalOwnership{4,2}(),LocalOwnership{4,1}())
    @test Tylo.same_distribution(Tylo.PermutedOwnership(LocalOwnership{4,2}()),LocalOwnership{4,1}())
end

@testset "Affine ownership fits" begin
    atom = MMAAtom((16,8,16),BFloat16)
    for o in (operand_layout(atom,OperandA()),operand_layout(atom,OperandB()),
              operand_layout(atom,Accumulator()),LocalOwnership{4,2}(),LocalOwnership{3,1}(),
              StripedOwnership{2,2}(),Tylo.WGMMAOwnership{24}(),
              Tylo.Layouts.layout(zero_accumulator(TiledMMA(atom,Val((2,2)),Val((2,2)),Val(16)))))
        fitted = Tylo.fit_ownership(Tylo.ownership_table(o),map(Int,size(o)))
        @test fitted !== nothing
        @test fitted isa Ownership
        @test isbitstype(typeof(fitted))
        @test Tylo.same_distribution(fitted,o)
    end
    # A table that is not affine in the thread bits has no fit.
    table = [(t == 5 ? 6 : t, e) for t in 0:31, e in 0:1]
    @test Tylo.fit_ownership(table,(32,2)) === nothing
end

@testset "Derived reduction plans match the implemented recipes" begin
    atom = MMAAtom((16,8,16),BFloat16)
    plan = Tylo.reduction_plan(operand_layout(atom,Accumulator()),2)
    @test plan.groups == [[1,2],[3,4]]
    @test plan.bits == [0,1]
    @test Tylo.ownership_table(plan.result) == [(t÷4+8e,0) for t in 0:31, e in 0:1]

    local_plan = Tylo.reduction_plan(LocalOwnership{5,2}(),2)
    @test local_plan.groups == [collect(1:5)]
    @test local_plan.bits == Int[]
    @test Tylo.ownership_table(local_plan.result) == [(t,0) for t in 0:31, e in 0:0]
    # Reducing the lane axis of lane-local rows is a full-warp shuffle per slot.
    columns = Tylo.reduction_plan(LocalOwnership{5,2}(),1)
    @test columns.groups == [[e] for e in 1:5]
    @test columns.bits == [0,1,2,3,4]
    @test Tylo.same_distribution(columns.result,StripedOwnership{5,2}()) == false
    @test Tylo.ownership_table(columns.result) == [(0,e) for t in 0:31, e in 0:4]

    striped = Tylo.reduction_plan(StripedOwnership{3,2}(),2)
    @test striped.groups == [[1,2,3]]
    @test striped.bits == [0,1,2,3,4]
    @test Tylo.ownership_table(striped.result) == [(0,0) for t in 0:31, e in 0:0]

    for wm in (1,2), rm in (1,2), rn in (1,3)
        tiled = Tylo.Layouts.layout(zero_accumulator(TiledMMA(atom,Val((wm,1)),Val((rm,rn)),Val(16))))
        p = Tylo.reduction_plan(tiled,2)
        @test p !== nothing
        @test length(p.groups) == 2rm
        @test all(==(2rn),length.(p.groups))
        @test p.bits == [0,1]
        @test Tylo.ownership_table(p.result) ==
              [(16rm*(t÷32)+(t%32)÷4+8*(e%2)+16*(e÷2),0) for t in 0:32wm-1, e in 0:2rm-1]
    end
    # Two N warps hold disjoint columns; no shuffle recipe reaches them.
    @test Tylo.reduction_plan(Tylo.Layouts.layout(zero_accumulator(TiledMMA(atom,Val((1,2)),Val((1,1)),Val(16)))),2) === nothing

    for n in (8,16,32)
        w = Tylo.reduction_plan(Tylo.WGMMAOwnership{n}(),2)
        @test w.groups == [[e for e in 1:n÷2 if ((e-1)%4)÷2 == h] for h in 0:1]
        @test w.bits == [0,1]
        @test Tylo.ownership_table(w.result) == [(16*(t÷32)+(t%32)÷4+8e,0) for t in 0:127, e in 0:1]
    end
    # Permutation exchanges the reducible axis.
    permuted = Tylo.reduction_plan(Tylo.PermutedOwnership(operand_layout(atom,Accumulator())),1)
    @test permuted.groups == plan.groups && permuted.bits == plan.bits
    # The permuted accumulator's axis 2 is the atom's row axis: an eight-lane group.
    permuted_columns = Tylo.reduction_plan(Tylo.PermutedOwnership(operand_layout(atom,Accumulator())),2)
    @test permuted_columns.groups == [[1,3],[2,4]] && permuted_columns.bits == [2,3,4]
    # Column reduction of the atom needs the eight-row group: bits 2,3,4 and both row slots.
    column = Tylo.reduction_plan(operand_layout(atom,Accumulator()),1)
    @test column.groups == [[1,3],[2,4]]
    @test column.bits == [2,3,4]
    @test_throws ArgumentError Tylo.reduction_plan(LocalOwnership{2,2}(),3)
end

@testset "In-lane relayout and windows" begin
    atom = MMAAtom((16,8,16),BFloat16)
    pair = Tylo.Layouts.layout(zero_accumulator(TiledMMA(atom,Val((1,1)),Val((1,2)),Val(16))))
    permutation = Tylo.relayout_permutation(pair,operand_layout(atom,OperandA()))
    # pack_operand_a pairs (c1,c2),(c3,c4) of the left atom then the right atom.
    @test permutation == [1,2,3,4,5,6,7,8]
    @test Tylo.relayout_permutation(operand_layout(atom,OperandA()),operand_layout(atom,OperandB())) === nothing
    @test Tylo.relayout_permutation(LocalOwnership{4,2}(),Tylo.TmemTransfer{(32,4),2}()) == [1,2,3,4]
    @test Tylo.relayout_permutation(LocalOwnership{4,2}(),LocalOwnership{2,2}()) === nothing

    w = Tylo.window_plan(LocalOwnership{8,2}(),(0,2),(32,4))
    @test w.slots == [3,4,5,6]
    @test Tylo.same_distribution(w.ownership,LocalOwnership{4,2}())
    @test Tylo.window_plan(LocalOwnership{8,2}(),(1,0),(31,8)) === nothing
    t = Tylo.window_plan(Tylo.TmemTransfer{(8,32),1}(),(2,0),(4,32))
    @test t.slots == [3,4,5,6]
    @test Tylo.same_distribution(t.ownership,Tylo.TmemTransfer{(4,32),1}())
    c = Tylo.window_plan(operand_layout(atom,Accumulator()),(8,0),(8,8))
    @test c.slots == [3,4]
end
