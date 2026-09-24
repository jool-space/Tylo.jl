using Tylo.Layouts: @Layout, Layout, Swizzle, compose

# Host-side tiles: LLVM pointers are plain addresses here, so alignment
# checks run without a device.
gptr(::Type{T},address) where T = reinterpret(Core.LLVMPtr{T,1},UInt64(address))

@testset "Tile alignment declarations" begin
    l = @Layout (16,16) (16,1)
    @test Tylo.alignment(GlobalTile(gptr(Float32,256),l)) == 4
    @test Tylo.alignment(GlobalTile(gptr(Float32,256),l,16)) == 16
    @test Tylo.alignment(typeof(GlobalTile(gptr(BFloat16,32),l,8))) == 8
    @test_throws ArgumentError GlobalTile(gptr(Float32,256),l,2)          # below the element size
    @test_throws ArgumentError GlobalTile(gptr(Float32,256),l,12)         # not a power of two
    @test_throws ArgumentError GlobalTile(gptr(Float32,264),l,16)         # misaligned pointer
    # Static strides are known exactly; only the base and runtime strides are declared.
    @test GlobalTile(gptr(Float32,256),@Layout((16,16),(17,1)),16) isa GlobalTile
    @test GlobalTile(gptr(Float32,256),Layout((16,16),(20,1)),16) isa GlobalTile
    @test_throws ArgumentError GlobalTile(gptr(Float32,256),Layout((16,16),(18,1)),16)
    # Swizzles keep only their unchanged low bits aligned.
    swz = compose(Swizzle{3,3,3}(),@Layout((16,64),(64,1)))
    @test GlobalTile(gptr(BFloat16,1024),swz,16) isa GlobalTile
    @test_throws ArgumentError GlobalTile(gptr(BFloat16,1024),swz,32)
    # A window's base is its parent's address of the origin.
    t = GlobalTile(gptr(Float32,256),l,16)
    @test Tylo.alignment(window(t,(3,4),(8,8))) == 16
    @test_throws ArgumentError window(t,(3,2),(8,8))
    unchecked(t) = @inbounds window(t,(3,2),(8,8))
    @test Tylo.alignment(unchecked(t)) == 16
    @test window(GlobalTile(gptr(Float32,256),l),(3,2),(8,8)) isa GlobalTile
end

rowmajor(m,n) = typeof(@Layout(($m,$n),($n,1)))
colmajor(m,n) = typeof(@Layout(($m,$n),(1,$m)))
@testset "Vector plans from ownership" begin
    atom = MMAAtom((16,8,16),BFloat16)
    acc = operand_layout(atom,Accumulator())
    # Accumulator pairs run along N: two FP32 per lane when rows are N-contiguous.
    @test Tylo.vector_plan(acc,Float32,rowmajor(16,8),16) == (;width=8,groups=[[1,2],[3,4]])
    @test Tylo.vector_plan(acc,Float32,rowmajor(16,8),8) == (;width=8,groups=[[1,2],[3,4]])
    @test Tylo.vector_plan(acc,Float32,rowmajor(16,8),4) === nothing
    @test Tylo.vector_plan(acc,Float32,colmajor(16,8),16) === nothing
    # 16-bit operand pairs along K are single words; FP8 quads are words too.
    @test Tylo.vector_plan(operand_layout(atom,OperandA()),BFloat16,rowmajor(16,16),16) == (;width=4,groups=[[1,2],[3,4],[5,6],[7,8]])
    @test Tylo.vector_plan(operand_layout(atom,OperandB()),BFloat16,colmajor(16,8),4) == (;width=4,groups=[[1,2],[3,4]])
    @test Tylo.vector_plan(operand_layout(atom,OperandA()),BFloat16,rowmajor(16,16),2) === nothing
    fp8 = MMAAtom((16,8,32),Float8_E4M3FN)
    @test Tylo.vector_plan(operand_layout(fp8,OperandA()),Float8_E4M3FN,rowmajor(16,32),16).width == 4
    # Lane-local rows vectorize to the declared width; striped rows never do.
    @test Tylo.vector_plan(lane_local(4,2),Float32,rowmajor(32,4),16) == (;width=16,groups=[[1,2,3,4]])
    @test Tylo.vector_plan(lane_local(4,2),Float32,rowmajor(32,4),8) == (;width=8,groups=[[1,2],[3,4]])
    @test Tylo.vector_plan(lane_local(6,2),Float32,rowmajor(32,6),16) == (;width=8,groups=[[1,2],[3,4],[5,6]])
    @test Tylo.vector_plan(striped(4,2),Float32,rowmajor(1,128),16) === nothing
    @test Tylo.vector_plan(lane_local(4,1),Float32,rowmajor(4,32),16) === nothing
    # Runtime strides separate vectors; the static ones inside a row still group.
    @test Tylo.vector_plan(lane_local(4,2),Float32,typeof(Layout((static(32),static(4)),(Int32(8),static(1)))),16) == (;width=16,groups=[[1,2,3,4]])
end
