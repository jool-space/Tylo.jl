using Tylo.Layouts: @Layout, Layout, Swizzle, compose, LocalOwnership, StripedOwnership

# Independent ISA figures for ldmatrix/stmatrix: lane t holds units
# (t÷4, 2(t%4)+e) of an 8×8 matrix, transposed with .trans; lanes 8i:8i+7
# address the rows of matrix i.
isa_matrix(trans,t,e) = trans ? (2(t%4)+e,t÷4) : (t÷4,2(t%4)+e)

@testset "Copy atoms as data" begin
    @test length(copy_atoms()) == 4
    for atom in copy_atoms()
        trans = typeof(atom).parameters[2]
        @test size(atom) == (8,8)
        @test Tylo.threads(atom) == 32
        registers = operand_layout(atom,Registers())
        @test Tylo.ownership_table(registers) == [isa_matrix(trans,t,e) for t in 0:31, e in 0:1]
        @test Tylo.is_complete(registers) && Tylo.is_injective(registers)
        @test Tylo.ownership_table(operand_layout(atom,Addresses())) == [(t%8,e) for t in 0:31, e in 0:7]
    end
    @test CopyAtom(:load) === CopyAtom{:load,false}()
    @test CopyAtom(:store,true) === CopyAtom{:store,true}()
    @test_throws ArgumentError CopyAtom{:move,false}()
    @test_throws ArgumentError CopyAtom{:load,1}()
end

@testset "Derived matrix copies for every warp MMA operand" begin
    for atom in instruction_atoms()
        TA,TB = eltype(atom,OperandA()),eltype(atom,OperandB())
        m,n,k = size(atom)
        bits = Tylo._element_bits(TA)
        a,b = operand_layout(atom,OperandA()),operand_layout(atom,OperandB())
        if bits == 32
            @test Tylo.matrix_copy_plan(a,TA,2) === nothing
            @test Tylo.matrix_copy_plan(b,TB,1) === nothing
            continue
        end
        per = 16 ÷ bits
        # K contiguous: the plain instruction, blocks in word order.
        pa = Tylo.matrix_copy_plan(a,TA,2)
        pb = Tylo.matrix_copy_plan(b,TB,1)
        @test !pa.trans && pa.per_unit == per && pa.grid == (m÷8,k÷8per) && pa.words == 1:length(pa.words)
        @test !pb.trans && pb.per_unit == per && pb.grid == (n÷8,k÷8per) && pb.words == 1:length(pb.words)
        # M or N contiguous: the transposed instruction for 16-bit elements.
        # 8-bit lanes hold K-adjacent pairs, which MN-contiguous rows cannot supply.
        qa,qb = Tylo.matrix_copy_plan(a,TA,1),Tylo.matrix_copy_plan(b,TB,2)
        @test bits == 16 ? (qa.trans && qb.trans) : (qa === nothing && qb === nothing)
    end
end

@testset "Matrix copy planning rejects what it cannot express" begin
    atom = MMAAtom((16,8,16),BFloat16)
    a = operand_layout(atom,OperandA())
    @test Tylo.matrix_copy_plan(operand_layout(atom,Accumulator()),Float32,2) === nothing
    # A packed 16-bit accumulator has the ldmatrix register pattern along N.
    @test Tylo.matrix_copy_plan(operand_layout(atom,Accumulator()),Float16,2).grid == (2,1)
    @test Tylo.matrix_copy_plan(operand_layout(atom,Accumulator()),Float16,1).trans
    # Lane-local rows hold different slots in each block; striped rows are not words.
    @test Tylo.matrix_copy_plan(LocalOwnership{8,2}(),BFloat16,2) === nothing
    @test Tylo.matrix_copy_plan(StripedOwnership{2,2}(),BFloat16,2) === nothing
    # Multi-warp ownerships are outside the warp-collective instruction.
    tiled = Tylo.Layouts.layout(zero_accumulator(TiledMMA(atom,Val((2,1)),Val((1,1)),Val(16))))
    @test Tylo.matrix_copy_plan(tiled,Float16,2) === nothing
    # Logical transposition swaps which storage axis needs .trans.
    @test !Tylo.matrix_copy_plan(Tylo.PermutedOwnership(a),BFloat16,1).trans
    @test Tylo.matrix_copy_plan(Tylo.PermutedOwnership(a),BFloat16,2).trans
    @test_throws ArgumentError Tylo.matrix_copy_plan(a,BFloat16,3)
    @test Tylo._matrix_groups(4) == [1:4]
    @test Tylo._matrix_groups(2) == [1:2]
    @test Tylo._matrix_groups(1) == [1:1]
    @test Tylo._matrix_groups(3) == [1:2,3:3]
    @test Tylo._matrix_groups(6) == [1:4,5:6]
    @test Tylo._matrix_groups(7) == [1:4,5:6,7:7]
end

@testset "Matrix copy validation against shared layouts" begin
    atom = MMAAtom((16,8,16),BFloat16)
    a,b = operand_layout(atom,OperandA()),operand_layout(atom,OperandB())
    l = @Layout (16,16) (16,1)
    @test validate_copy(a,BFloat16,l) === nothing
    @test validate_copy(a,BFloat16,compose(Swizzle{1,3,1}(),l)) === nothing
    @test validate_copy(a,BFloat16,@Layout (16,16) (1,16)) === nothing
    @test validate_copy(b,BFloat16,@Layout (16,8) (1,16)) === nothing
    @test validate_copy(b,BFloat16,Tylo.Layouts.window(@Layout((64,32),(1,64)),(16,8),Val((16,8)))) === nothing
    fp8 = MMAAtom((16,8,32),Float8E4M3)
    @test validate_copy(operand_layout(fp8,OperandA()),Float8E4M3,@Layout (16,32) (32,1)) === nothing
    @test validate_copy(operand_layout(fp8,OperandA()),Float8E4M3,compose(Swizzle{1,4,1}(),@Layout (16,32) (32,1))) === nothing
    @test_throws ArgumentError validate_copy(a,BFloat16,@Layout (16,16) (17,1))   # rows not 16-byte aligned
    @test_throws ArgumentError validate_copy(a,BFloat16,compose(Swizzle{1,2,1}(),l)) # swizzle breaks 8-element rows
    @test_throws ArgumentError validate_copy(a,BFloat16,@Layout (16,16) (2,32))   # no unit stride
    @test_throws ArgumentError validate_copy(a,BFloat16,Layout((16,16),(16,1)))   # runtime strides
    @test_throws DimensionMismatch validate_copy(a,BFloat16,@Layout (16,8) (8,1))
    @test_throws ArgumentError validate_copy(LocalOwnership{8,2}(),BFloat16,@Layout (32,8) (8,1))
    @test Tylo._contiguous_axis(typeof(l)) == 2
    @test Tylo._contiguous_axis(typeof(@Layout (16,16) (1,16))) == 1
    @test Tylo._contiguous_axis(typeof(shared_layout(TMALoad(BFloat16,Val((64,64)),Val(2))))) == 2
    @test Tylo._contiguous_axis(typeof(@Layout (16,16) (1,1))) === nothing
end
