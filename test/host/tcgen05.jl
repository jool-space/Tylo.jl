using Tylo.Layouts: @Layout, Layout, Swizzle, compose, window, static

@testset "tcgen05 atoms as data" begin
    a = Tcgen05MMA((128,128,16),BFloat16)
    @test size(a) == (128,128,16)
    @test Tylo.threads(a) == 1
    @test eltype(a,OperandA()) === BFloat16 && eltype(a,Accumulator()) === Float32
    @test operand_layout(a,Accumulator()) == @Layout((128,128),(1,128))
    @test operand_layout(Tcgen05MMA((128,64,16),Float16),Accumulator()) == @Layout((128,64),(1,128))
    @test Tcgen05MMA((128,64,8),Float32) isa Tcgen05MMA
    @test Tcgen05MMA((128,256,32),Float8E4M3,Float8E5M2,Float32) isa Tcgen05MMA
    @test Tcgen05MMA((128,32,32),Int8,UInt8,Int32) isa Tcgen05MMA
    @test_throws ArgumentError Tcgen05MMA((128,128,8),BFloat16)      # K is 256 bits
    @test_throws ArgumentError Tcgen05MMA((64,128,16),BFloat16)      # M=64 layout not described
    @test_throws ArgumentError Tcgen05MMA((128,8,16),BFloat16)       # N multiple of 16
    @test_throws ArgumentError Tcgen05MMA((128,128,16),BFloat16,Float16)
    @test_throws ArgumentError Tcgen05MMA((128,128,16),BFloat16,Float16,Float32)
    @test_throws ArgumentError Tcgen05MMA((128,128,32),Int8,Int8,Float32)
end

@testset "Instruction descriptors are pure functions of the atom" begin
    a = Tcgen05MMA((128,128,16),BFloat16)
    # PTX ISA Table 45: D=f32 at bits 4-5, A/B format bf16=1 at bits 7-9 and
    # 10-12, majorness at bits 15-16, N>>3 at bits 17-22, M>>4 at bits 24-28.
    expected(am,bm) = UInt32(1<<4 | 1<<7 | 1<<10 | am<<15 | bm<<16 | (128>>3)<<17 | (128>>4)<<24)
    @test Tylo.instruction_descriptor(a) == expected(0,0)
    @test Tylo.instruction_descriptor(a;b_major=:MN) == expected(0,1)
    @test Tylo.instruction_descriptor(a;a_major=:MN,b_major=:MN) == expected(1,1)
    @test Tylo.instruction_descriptor(a;b_major=:MN) ==
        PTX.tcgen05_instr_desc_f16bf16_f32(m=128,n=128,ab_dtype=:bf16,a_major=:K,b_major=:MN)
    @test Tylo.instruction_descriptor(Tcgen05MMA((128,64,8),Float32)) ==
        PTX.tcgen05_instr_desc_f16bf16_f32(m=128,n=64,ab_dtype=:tf32)
    @test Tylo.instruction_descriptor(Tcgen05MMA((128,32,32),Int8,UInt8,Int32)) ==
        PTX.tcgen05_instr_desc_i8(m=128,n=32,a_dtype=:s8,b_dtype=:u8)
    @test Tylo.instruction_descriptor(Tcgen05MMA((128,256,32),Float8E4M3,Float8E5M2,Float32)) ==
        PTX.tcgen05_instr_desc_f8f6f4(m=128,n=256,a_dtype=:e4m3,b_dtype=:e5m2)
end

@testset "Canonical swizzled encodings" begin
    structure = Tylo.swizzled_structure
    # Tylo's TMA storage is the canonical K-major encoding of either operand.
    tma_a = typeof(shared_layout(TMALoad(BFloat16,Val((128,64)),Val(2))))
    tma_b = typeof(shared_layout(TMALoad(BFloat16,Val((64,128)),Val(1))))
    sa = structure(tma_a,BFloat16,2)
    @test sa.major === :K && sa.swizzle_bytes == 128 && sa.leading_bytes == 16 && sa.stride_bytes == 1024 && sa.groups == 1
    sb = structure(tma_b,BFloat16,1)
    @test sb.major === :K && sb.stride_bytes == 1024
    # The same storage read along the other axis is MN-major.
    @test structure(tma_a,BFloat16,1).major === :MN
    @test structure(tma_a,Float32,2) === nothing
    # The attention kernel's buffers: a two-stage Q buffer, K slots viewed
    # transposed, V slots MN-major with a 16 KiB leading offset.
    q = compose(Swizzle{3,3,3}(),@Layout(((128,2),(64,2)),((64,16384),(1,8192))))
    k = compose(Swizzle{3,3,3}(),@Layout(((64,2),(128,2)),((1,8192),(64,32768))))
    v = compose(Swizzle{3,3,3}(),@Layout(((128,2),(64,2)),((64,32768),(1,8192))))
    @test structure(typeof(q),BFloat16,2) == (;major=:K,swizzle_bytes=128,leading_bytes=16,stride_bytes=1024,row_elements=64,groups=2)
    @test structure(typeof(k),BFloat16,1) == (;major=:K,swizzle_bytes=128,leading_bytes=16,stride_bytes=1024,row_elements=64,groups=2)
    @test structure(typeof(v),BFloat16,1) == (;major=:MN,swizzle_bytes=128,leading_bytes=16384,stride_bytes=1024,row_elements=64,groups=2)
    @test structure(typeof(window(q,(Int32(128),Int32(0)),Val((128,128)))),BFloat16,2).major === :K
    # Element widths change the swizzle's unchanged low bits and the row extent.
    @test structure(typeof(compose(Swizzle{3,4,3}(),@Layout((128,128),(128,1)))),Float8E4M3,2).major === :K
    @test structure(typeof(compose(Swizzle{3,2,3}(),@Layout((128,32),(32,1)))),Float32,2).major === :K
    @test structure(typeof(compose(Swizzle{3,3,3}(),@Layout((128,128),(128,1)))),Float8E4M3,2) === nothing
    # Narrower rows: 64- and 32-byte swizzles with eight-row groups packed
    # (stride 8W) or spread, and MN-major groups one row wide.
    @test structure(typeof(compose(Swizzle{2,3,3}(),@Layout((128,32),(32,1)))),BFloat16,2) ==
        (;major=:K,swizzle_bytes=64,leading_bytes=16,stride_bytes=512,row_elements=32,groups=1)
    @test structure(typeof(compose(Swizzle{1,3,3}(),@Layout((128,16),(16,1)))),BFloat16,2) ==
        (;major=:K,swizzle_bytes=32,leading_bytes=16,stride_bytes=256,row_elements=16,groups=1)
    @test structure(typeof(compose(Swizzle{2,2,3}(),@Layout((128,(16,4)),(16,(1,2048))))),Float32,2) ==
        (;major=:K,swizzle_bytes=64,leading_bytes=16,stride_bytes=512,row_elements=16,groups=4)
    @test structure(typeof(compose(Swizzle{2,3,3}(),@Layout(((32,4),64),((1,2048),32)))),BFloat16,2) ==
        (;major=:MN,swizzle_bytes=64,leading_bytes=4096,stride_bytes=512,row_elements=32,groups=4)
    @test structure(typeof(compose(Swizzle{2,3,3}(),@Layout(((8,16),32),((32,512),1)))),BFloat16,2).stride_bytes == 1024
    # Rejections: no swizzle, the wrong swizzle, rows of the wrong width,
    # a row count that is not a multiple of eight, strides breaking the cycle.
    @test structure(typeof(@Layout((128,64),(64,1))),BFloat16,2) === nothing
    @test structure(typeof(compose(Swizzle{2,3,2}(),@Layout((128,64),(64,1)))),BFloat16,2) === nothing
    @test structure(typeof(compose(Swizzle{3,3,3}(),@Layout((128,32),(32,1)))),BFloat16,2) === nothing
    @test structure(typeof(compose(Swizzle{2,3,3}(),@Layout((128,64),(64,1)))),BFloat16,2) === nothing
    @test structure(typeof(compose(Swizzle{4,3,4}(),@Layout((128,256),(256,1)))),BFloat16,2) === nothing
    @test structure(typeof(compose(Swizzle{3,3,3}(),@Layout((12,64),(64,1)))),BFloat16,2) === nothing
    @test structure(typeof(compose(Swizzle{3,3,3}(),@Layout(((128,2),64),((64,8200),1)))),BFloat16,2) === nothing
    @test structure(typeof(compose(Swizzle{3,3,3}(),Layout((128,64),(64,1)))),BFloat16,2) === nothing
    # Descriptor codes follow the row width.
    for (W,tc,wc) in ((128,PTX.BlackwellLayout.B128,PTX.WgmmaSwizzle.B128),(64,PTX.BlackwellLayout.B64,PTX.WgmmaSwizzle.B64),(32,PTX.BlackwellLayout.B32,PTX.WgmmaSwizzle.B32))
        @test Tylo._tcgen05_swizzle(W) == tc && Tylo._wgmma_swizzle(W) == wc
        @test Tylo._descriptor_fields(Val(16),Val(8W),Val(W)) ==
            PTX.tcgen05_descriptor(UInt32(0);leading_bytes=16,stride_bytes=8W,swizzle=tc)
    end
end
