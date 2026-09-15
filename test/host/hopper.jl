@testset "Canonical TMA storage" begin
    # The hardware rule in bytes: 16-byte chunk bits XORed with bits 7..9 of
    # the byte address, masked to the row width.
    swizzle(a,W) = a ⊻ (((a >> 7) & (W ÷ 16 - 1)) << 4)
    for T in (UInt8,BFloat16,Float16,Float32), W in (32,64,128), axis in (1,2), rows in (8,16,64,128,256)
        row = W ÷ sizeof(T)
        shape = axis == 2 ? (rows,row) : (row,rows)
        p = TMATile(T,Val(shape),Val(axis),Val(W)); l = shared_layout(p)
        @test size(p) == shape && swizzle_bytes(p) == W && transfer_bytes(p) == W*rows
        offsets = [Int(l(axis == 2 ? (Int32(r),Int32(k)) : (Int32(k),Int32(r)))) for r in 0:rows-1,k in 0:row-1]
        expected = [swizzle(sizeof(T)*(row*r+k),W) ÷ sizeof(T) for r in 0:rows-1,k in 0:row-1]
        @test offsets == expected
        @test sort(vec(offsets)) == collect(0:row*rows-1)
        @test Tylo.Layouts.cosize(l)*sizeof(T) == transfer_bytes(p)
        @test l((Int32(0),Int32(0))) isa Int32
        s = Tylo.swizzled_structure(typeof(l),T,axis)
        @test s.major === :K && s.swizzle_bytes == W && s.row_elements == row && s.stride_bytes == 8W && s.groups == 1
    end
    @test TMALoad(BFloat16,Val((128,64)),Val(2)) === TMATile(BFloat16,Val((128,64)),Val(2))
    @test TMALoad(BFloat16,Val((128,64)),Val(2)) isa TMALoad{BFloat16,(128,64),2}
    for (shape,axis) in (((64,32),2),((7,64),2),((264,64),2),((64,8),0))
        @test_throws ArgumentError TMALoad(BFloat16,Val(shape),Val(axis))
    end
    @test_throws ArgumentError TMATile(Float32,Val((64,64)),Val(2))
    @test_throws ArgumentError TMATile(Float32,Val((32,32)),Val(2),Val(64))
    @test_throws ArgumentError TMATile(BFloat16,Val((64,64)),Val(2),Val(256))
    @test_throws ArgumentError TMATile(BFloat16,Val((64,64)),Val(2),Val(48))
    @test_throws ArgumentError TMATile(Float64,Val((64,16)),Val(2))
    @test_throws ArgumentError TMATile(UInt8,Val((64,64)),Val(2))
end

@testset "WGMMA ownership" begin
    for T in (BFloat16,Float16), n in 8:8:256
        p = WGMMA64(T,Val(n))
        l = operand_layout(p,Accumulator())
        coordinates = [Tylo.Layouts.coordinate(l,Int32(t),Val(e)) for t in 0:127,e in 0:n÷2-1]
        @test Set(coordinates) == Set((r,c) for r in 0:63,c in 0:n-1)
        @test all(c -> c isa Tuple{Int32,Int32},coordinates)
        @test all(iszero,finish_mma(zero_accumulator(p)).data)
    end
    for k in (16,32,64), partials in (1,k÷16)
        p = WGMMA64(BFloat16,Val(8),Val(k),Val(partials))
        c = Tylo.WGMMAAccumulator(p,ntuple(Float32,4partials))
        f = finish_mma(c)
        @test f.data == ntuple(i -> sum(Float32(i+4j) for j in 0:partials-1),4)
        @test map(x -> 2f0*x,f).data == scale(f,2f0).data
    end
    @test_throws ArgumentError WGMMA64(Float32,Val(8))
    @test_throws ArgumentError WGMMA64(BFloat16,Val(9))
    @test_throws ArgumentError WGMMA64(BFloat16,Val(8),Val(48))
    @test_throws ArgumentError WGMMA64(BFloat16,Val(8),Val(64),Val(3))
    @test_throws ArgumentError WGMMA64(BFloat16,Val(256),Val(64),Val(4))
    @test_throws ArgumentError Tylo.WGMMAAccumulator(WGMMA64(BFloat16,Val(8)),(0f0,))
end

@testset "WGMMA descriptor geometry" begin
    for k in (16,32,64), n in (8,16,64,256)
        p = WGMMA64(BFloat16,Val(n),Val(k))
        a = shared_layout(TMALoad(BFloat16,Val((128,64)),Val(2)))
        b = shared_layout(TMALoad(BFloat16,Val((64,256)),Val(1)))
        for kk in 0:16:64-k, rr in (0,8,64)
            @test validate_wgmma(p,OperandA(),a,(rr,kk)) === nothing
        end
        for kk in 0:16:64-k
            @test validate_wgmma(p,OperandB(),b,(kk,256-n)) === nothing
        end
        @test_throws ArgumentError validate_wgmma(p,OperandA(),b)
        @test_throws ArgumentError validate_wgmma(p,OperandB(),a)
        @test_throws ArgumentError validate_wgmma(p,OperandA(),a,(1,0))
        @test_throws ArgumentError validate_wgmma(p,OperandA(),a,(0,8))
        @test_throws ArgumentError validate_wgmma(p,OperandA(),a,(72,0))
        @test_throws ArgumentError validate_wgmma(p,OperandA(),a,(0,64))
    end
    # Narrower rows: the plan's K must fit one row, origins stay inside it.
    for W in (32,64), k in (16,32,64)
        row = W ÷ 2
        a = shared_layout(TMATile(BFloat16,Val((64,row)),Val(2),Val(W)))
        p = WGMMA64(BFloat16,Val(16),Val(k))
        if k <= row
            for kk in 0:16:row-k
                @test validate_wgmma(p,OperandA(),a,(0,kk)) === nothing
            end
            @test_throws ArgumentError validate_wgmma(p,OperandA(),a,(0,row-k+16))
        else
            @test_throws ArgumentError validate_wgmma(p,OperandA(),a)
        end
    end
end
