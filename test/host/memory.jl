using Tylo.Layouts: @Layout, Layout, Swizzle, compose, LocalOwnership, StripedOwnership

# Host-side tiles: LLVM pointers are plain addresses here, so alignment
# checks run without a device.
gptr(::Type{T},address) where T = reinterpret(Core.LLVMPtr{T,1},UInt64(address))

@testset "Tile alignment declarations" begin
    l = @Layout (16,16) (16,1)
    @test Tylo.alignment(GlobalTile(gptr(Float32,256),l)) == 4
    @test Tylo.alignment(GlobalTile(gptr(Float32,256),l,Val(16))) == 16
    @test Tylo.alignment(typeof(GlobalTile(gptr(BFloat16,32),l,Val(8)))) == 8
    @test_throws ArgumentError GlobalTile(gptr(Float32,256),l,Val(2))          # below the element size
    @test_throws ArgumentError GlobalTile(gptr(Float32,256),l,Val(12))         # not a power of two
    @test_throws ArgumentError GlobalTile(gptr(Float32,264),l,Val(16))         # misaligned pointer
    @test_throws ArgumentError GlobalTile(gptr(Float32,256),@Layout((16,16),(17,1)),Val(16))  # stride not a multiple
    @test GlobalTile(gptr(Float32,256),@Layout((16,16),(17,1)),Val(4)) isa GlobalTile
    @test GlobalTile(gptr(Float32,256),Layout((16,16),(20,1)),Val(16)) isa GlobalTile  # runtime strides checked at construction
    @test_throws ArgumentError GlobalTile(gptr(Float32,256),Layout((16,16),(18,1)),Val(16))
    # Swizzles keep only their unchanged low bits aligned.
    swz = compose(Swizzle{3,3,3}(),@Layout((16,64),(64,1)))
    @test GlobalTile(gptr(BFloat16,1024),swz,Val(16)) isa GlobalTile
    @test_throws ArgumentError GlobalTile(gptr(BFloat16,1024),swz,Val(32))
    # Windows keep the declaration only at aligned origins along the unit stride.
    t = GlobalTile(gptr(Float32,256),l,Val(16))
    @test Tylo.alignment(window(t,(3,4),Val((8,8)))) == 16
    @test_throws ArgumentError window(t,(3,2),Val((8,8)))
    unchecked(t) = @inbounds window(t,(3,2),Val((8,8)))
    @test Tylo.alignment(unchecked(t)) == 16
    @test window(GlobalTile(gptr(Float32,256),l),(3,2),Val((8,8))) isa GlobalTile
end

@testset "Vector plans from ownership" begin
    atom = MMAAtom((16,8,16),BFloat16)
    acc = operand_layout(atom,Accumulator())
    # Accumulator pairs run along N: two FP32 per lane when rows are N-contiguous.
    @test Tylo.vector_plan(acc,Float32,2,16) == (;width=8,groups=[[1,2],[3,4]])
    @test Tylo.vector_plan(acc,Float32,2,8) == (;width=8,groups=[[1,2],[3,4]])
    @test Tylo.vector_plan(acc,Float32,2,4) === nothing
    @test Tylo.vector_plan(acc,Float32,1,16) === nothing
    # 16-bit operand pairs along K are single words; FP8 quads are words too.
    @test Tylo.vector_plan(operand_layout(atom,OperandA()),BFloat16,2,16) == (;width=4,groups=[[1,2],[3,4],[5,6],[7,8]])
    @test Tylo.vector_plan(operand_layout(atom,OperandB()),BFloat16,1,4) == (;width=4,groups=[[1,2],[3,4]])
    @test Tylo.vector_plan(operand_layout(atom,OperandA()),BFloat16,2,2) === nothing
    fp8 = MMAAtom((16,8,32),Float8E4M3)
    @test Tylo.vector_plan(operand_layout(fp8,OperandA()),Float8E4M3,2,16).width == 4
    # Lane-local rows vectorize to the declared width; striped rows never do.
    @test Tylo.vector_plan(LocalOwnership{4,2}(),Float32,2,16) == (;width=16,groups=[[1,2,3,4]])
    @test Tylo.vector_plan(LocalOwnership{4,2}(),Float32,2,8) == (;width=8,groups=[[1,2],[3,4]])
    @test Tylo.vector_plan(LocalOwnership{6,2}(),Float32,2,16) == (;width=8,groups=[[1,2],[3,4],[5,6]])
    @test Tylo.vector_plan(StripedOwnership{4,2}(),Float32,2,16) === nothing
    @test Tylo.vector_plan(LocalOwnership{4,1}(),Float32,2,16) === nothing
    @test_throws ArgumentError Tylo.vector_plan(acc,Float32,3,16)
end
