# Derivations from ownership layouts, checked against the enumeration
# oracle: every ownership's `mapping` must reproduce the device `coordinate`
# table, and every derived plan must match the independently known recipe.
using Tylo.Layouts: @Layout, Layout, coordinate, Ownership, static
wgmma_c(n) = operand_layout(WGMMAAtom(BFloat16,n),Accumulator())

# Independent ISA figures for the m16n8k16 warp MMA and the 64×N WGMMA
# accumulator. These do not go through Tylo's coordinate methods.
isa_a(t,e) = (t÷4 + 8*((e÷2)%2), 2*(t%4) + e%2 + 8*(e÷4))
isa_b(t,e) = (2*(t%4) + e%2 + 8*(e÷2), t÷4)
isa_c(t,e) = (t÷4 + 8*(e÷2), 2*(t%4) + e%2)
isa_wgmma_c(t,e) = (16*(t÷32) + (t%32)÷4 + 8*((e%4)÷2), 2*(t%4) + e%2 + 8*(e÷4))

# The table an ownership's mapping describes, for comparison with the
# table its coordinate method produces.
function mapped_table(o)
    m = Tylo.mapping(o)
    M = Int(size(o)[1])
    [(Int(m((t,e))) % M, Int(m((t,e))) ÷ M) for t in 0:Tylo._thread_count(o)-1, e in 0:Tylo._register_count(o)-1]
end
atom = MMAAtom((16,8,16),BFloat16)
tiled(w,r) = Tylo.Layouts.layout(zero_accumulator(TiledMMA(atom,w,r,16)))
ownerships = Any[operand_layout(atom,OperandA()),operand_layout(atom,OperandB()),operand_layout(atom,Accumulator()),
    operand_layout(MMAAtom((16,8,32),Float8_E4M3FN),OperandA()),operand_layout(MMAAtom((16,8,8),Float32),OperandB()),
    lane_local(3,2),lane_local(4,1),striped(2,1),striped(5,2),
    Tylo.Layouts.layout(Tylo.TmemTransfer((32,4),2)),Tylo.Layouts.layout(Tylo.TmemTransfer((8,32),1)),
    wgmma_c(8),wgmma_c(24),wgmma_c(64),
    tiled((1,1),(1,2)),tiled((2,2),(2,3)),tiled((1,2),(1,1)),tiled((2,1),(3,1)),
    permutedims(lane_local(3,2)),permutedims(operand_layout(atom,Accumulator())),
    permutedims(tiled((2,2),(2,2))),
    [operand_layout(a,r) for a in copy_atoms() for r in (Registers(),Addresses())]...]

@testset "Mappings reproduce the coordinate tables" begin
    @test Tylo.ownership_table(operand_layout(atom,OperandA())) == [isa_a(t,e) for t in 0:31, e in 0:7]
    @test Tylo.ownership_table(operand_layout(atom,OperandB())) == [isa_b(t,e) for t in 0:31, e in 0:3]
    @test Tylo.ownership_table(operand_layout(atom,Accumulator())) == [isa_c(t,e) for t in 0:31, e in 0:3]
    for n in (8,16,64)
        @test Tylo.ownership_table(wgmma_c(n)) == [isa_wgmma_c(t,e) for t in 0:127, e in 0:n÷2-1]
    end
    @test Tylo.ownership_table(lane_local(3,2)) == [(t,e) for t in 0:31, e in 0:2]
    @test Tylo.ownership_table(striped(2,1)) == [(t+32e,0) for t in 0:31, e in 0:1]
    @test Tylo.ownership_table(Tylo.Layouts.layout(Tylo.TmemTransfer((32,4),2))) == [(t,e) for t in 0:31, e in 0:3]
    @test Tylo.ownership_table(permutedims(lane_local(3,2))) == [(e,t) for t in 0:31, e in 0:2]
    for o in ownerships
        @test mapped_table(o) == Tylo.ownership_table(o)
        @test Tylo.Layouts._isstatic(Tylo.mapping(o))
        @test Tylo.same_distribution(o,o)
    end
    for o in (operand_layout(atom,OperandA()),operand_layout(atom,Accumulator()),
              lane_local(4,2),striped(2,2),wgmma_c(16),tiled((2,2),(2,3)))
        @test Tylo.is_complete(o)
        @test Tylo.is_injective(o)
    end
    @test !Tylo.is_injective(Tylo.reduction_plan(striped(2,2),2).result)
end

@testset "Distribution equality agrees with the oracle" begin
    for a in ownerships, b in ownerships
        expected = size(a) == size(b) && Tylo._thread_count(a) == Tylo._thread_count(b) &&
            Tylo._register_count(a) == Tylo._register_count(b) && Tylo.ownership_table(a) == Tylo.ownership_table(b)
        @test Tylo.same_distribution(a,b) == expected
    end
    @test Tylo.same_distribution(lane_local(4,2),Tylo.Layouts.layout(Tylo.TmemTransfer((32,4),2)))
    @test !Tylo.same_distribution(lane_local(4,2),lane_local(4,1))
    @test Tylo.same_distribution(permutedims(lane_local(4,2)),lane_local(4,1))
    # Canonical forms: one type per distribution.
    @test permutedims(lane_local(4,2)) === lane_local(4,1)
    @test Tylo.Layouts.layout(Tylo.TmemTransfer((32,4),2)) === lane_local(4,2)
    @test permutedims(permutedims(operand_layout(atom,Accumulator()))) === operand_layout(atom,Accumulator())
end

@testset "Derived reduction plans match the implemented recipes" begin
    plan = Tylo.reduction_plan(operand_layout(atom,Accumulator()),2)
    @test plan.groups == [[1,2],[3,4]]
    @test plan.bits == [0,1]
    @test Tylo.ownership_table(plan.result) == [(t÷4+8e,0) for t in 0:31, e in 0:1]
    @test plan.result isa Ownership && isbitstype(typeof(plan.result))

    local_plan = Tylo.reduction_plan(lane_local(5,2),2)
    @test local_plan.groups == [collect(1:5)]
    @test local_plan.bits == Int[]
    @test Tylo.ownership_table(local_plan.result) == [(t,0) for t in 0:31, e in 0:0]
    # Reducing the lane axis of lane-local rows is a full-warp shuffle per slot.
    columns = Tylo.reduction_plan(lane_local(5,2),1)
    @test columns.groups == [[e] for e in 1:5]
    @test columns.bits == [0,1,2,3,4]
    @test Tylo.same_distribution(columns.result,striped(5,2)) == false
    @test Tylo.ownership_table(columns.result) == [(0,e) for t in 0:31, e in 0:4]

    stripes = Tylo.reduction_plan(striped(3,2),2)
    @test stripes.groups == [[1,2,3]]
    @test stripes.bits == [0,1,2,3,4]
    @test Tylo.ownership_table(stripes.result) == [(0,0) for t in 0:31, e in 0:0]

    for wm in (1,2), rm in (1,2), rn in (1,3)
        p = Tylo.reduction_plan(tiled((wm,1),(rm,rn)),2)
        @test p !== nothing
        @test length(p.groups) == 2rm
        @test all(==(2rn),length.(p.groups))
        @test p.bits == [0,1]
        @test Tylo.ownership_table(p.result) ==
              [(16rm*(t÷32)+(t%32)÷4+8*(e%2)+16*(e÷2),0) for t in 0:32wm-1, e in 0:2rm-1]
    end
    # Two N warps hold disjoint columns; no shuffle recipe reaches them.
    @test Tylo.reduction_plan(tiled((1,2),(1,1)),2) === nothing

    for n in (8,16,32)
        w = Tylo.reduction_plan(wgmma_c(n),2)
        @test w.groups == [[e for e in 1:n÷2 if ((e-1)%4)÷2 == h] for h in 0:1]
        @test w.bits == [0,1]
        @test Tylo.ownership_table(w.result) == [(16*(t÷32)+(t%32)÷4+8e,0) for t in 0:127, e in 0:1]
    end
    # Permutation exchanges the reducible axis.
    permuted = Tylo.reduction_plan(permutedims(operand_layout(atom,Accumulator())),1)
    @test permuted.groups == plan.groups && permuted.bits == plan.bits
    # The permuted accumulator's axis 2 is the atom's row axis: an eight-lane group.
    permuted_columns = Tylo.reduction_plan(permutedims(operand_layout(atom,Accumulator())),2)
    @test permuted_columns.groups == [[1,3],[2,4]] && permuted_columns.bits == [2,3,4]
    # Column reduction of the atom needs the eight-row group: bits 2,3,4 and both row slots.
    column = Tylo.reduction_plan(operand_layout(atom,Accumulator()),1)
    @test column.groups == [[1,3],[2,4]]
    @test column.bits == [2,3,4]
    @test_throws ArgumentError Tylo.reduction_plan(lane_local(2,2),3)
end

@testset "Broadcast slot maps" begin
    acc = operand_layout(atom,Accumulator())
    rows = Tylo.reduction_plan(acc,2).result
    @test Tylo.broadcast_slots(rows,acc) == [1,1,2,2]
    cols = Tylo.reduction_plan(acc,1).result
    @test Tylo.broadcast_slots(cols,acc) == [1,2,1,2]
    @test Tylo.broadcast_slots(acc,acc) == [1,2,3,4]
    @test Tylo.broadcast_slots(rows,lane_local(8,2)) === nothing
    local_rows = Tylo.reduction_plan(lane_local(5,2),2).result
    @test Tylo.broadcast_slots(local_rows,lane_local(5,2)) == [1,1,1,1,1]
    @test Tylo.broadcast_slots(Tylo.reduction_plan(wgmma_c(16),2).result,wgmma_c(16)) ==
          [1+((e-1)%4)÷2 for e in 1:8]
end

@testset "In-lane relayout and windows" begin
    pair = tiled((1,1),(1,2))
    permutation = Tylo.relayout_permutation(pair,operand_layout(atom,OperandA()))
    # pack_operand_a pairs (c1,c2),(c3,c4) of the left atom then the right atom.
    @test permutation == [1,2,3,4,5,6,7,8]
    @test Tylo.relayout_permutation(operand_layout(atom,OperandA()),operand_layout(atom,OperandB())) === nothing
    @test Tylo.relayout_permutation(lane_local(4,2),Tylo.Layouts.layout(Tylo.TmemTransfer((32,4),2))) == [1,2,3,4]
    @test Tylo.relayout_permutation(lane_local(4,2),lane_local(2,2)) === nothing

    w = Tylo.window_plan(lane_local(8,2),(0,2),(32,4))
    @test w.slots == [3,4,5,6]
    @test Tylo.same_distribution(w.ownership,lane_local(4,2))
    @test Tylo.window_plan(lane_local(8,2),(1,0),(31,8)) === nothing
    t = Tylo.window_plan(Tylo.Layouts.layout(Tylo.TmemTransfer((8,32),1)),(2,0),(4,32))
    @test t.slots == [3,4,5,6]
    @test Tylo.same_distribution(t.ownership,Tylo.Layouts.layout(Tylo.TmemTransfer((4,32),1)))
    c = Tylo.window_plan(operand_layout(atom,Accumulator()),(8,0),(8,8))
    @test c.slots == [3,4]
    @test Tylo.ownership_table(c.ownership) == [isa_c(t,e) for t in 0:31, e in 0:1]
    # A window of a tiled accumulator selects atoms; a window cutting an atom has no plan.
    big = tiled((2,2),(2,2))
    quarter = Tylo.window_plan(big,(32,16),(32,16))
    @test quarter === nothing   # the second warp row and column: other threads, not other slots
    atoms = Tylo.window_plan(big,(0,0),(32,32))
    @test atoms === nothing     # 32 rows span two warps in M
    right = Tylo.window_plan(tiled((1,1),(2,2)),(0,8),(32,8))
    @test right.slots == [9,10,11,12,13,14,15,16]
    @test Tylo.ownership_table(right.ownership) == Tylo.ownership_table(tiled((1,1),(2,1)))
    @test Tylo.window_plan(tiled((1,1),(2,2)),(4,0),(8,16)) === nothing
end
