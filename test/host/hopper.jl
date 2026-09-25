@testset "Canonical TMA storage" begin
    # The hardware rule in bytes: 16-byte chunk bits XORed with bits 7..9 of
    # the byte address, masked to the row width.
    swizzle(a,W) = a ⊻ (((a >> 7) & (W ÷ 16 - 1)) << 4)
    for T in (UInt8,BFloat16,Float16,Float32), W in (32,64,128), axis in (1,2), rows in (8,16,64,128,256)
        row = W ÷ sizeof(T)
        shape = axis == 2 ? (rows,row) : (row,rows)
        p = TMATile(T,shape,axis,W); l = shared_layout(p)
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
    # Several stripes: each W bytes of the inner axis, stored after the last.
    for T in (UInt8,BFloat16,Float32), W in (32,128), axis in (1,2), stripes in (2,3)
        row = W ÷ sizeof(T); rows = 16
        shape = axis == 2 ? (rows,stripes*row) : (stripes*row,rows)
        p = TMATile(T,shape,axis,W); l = shared_layout(p)
        @test transfer_bytes(p) == stripes*W*rows && Tylo._box(p) == (row,rows)
        for r in 0:rows-1, k in 0:stripes*row-1
            s, kk = divrem(k,row)
            c = axis == 2 ? (Int32(r),Int32(k)) : (Int32(k),Int32(r))
            @test Int(l(c)) == s*row*rows + swizzle(sizeof(T)*(row*r+kk),W) ÷ sizeof(T)
        end
        st = Tylo.swizzled_structure(typeof(l),T,axis)
        @test st.major === :K && st.groups == stripes && st.row_elements == row
    end
    @test TMATile(BFloat16,(128,64),2) === TMATile(BFloat16,(128,64),2)
    @test TMATile(BFloat16,(128,64),2) isa TMATile{BFloat16,(128,64),2,128}
    for (shape,axis) in (((64,32),2),((7,64),2),((264,64),2),((64,8),0))
        @test_throws ArgumentError TMATile(BFloat16,shape,axis)
    end
    @test Tylo._stripes(TMATile(Float32,(64,64),2)) == 2 && Tylo._stripes(TMATile(Float32,(32,32),2,64)) == 2
    @test_throws ArgumentError TMATile(Float32,(64,40),2)      # 160 bytes: not whole rows
    @test_throws ArgumentError TMATile(Float32,(32,24),2,64)
    @test_throws ArgumentError TMATile(BFloat16,(64,64),2,256)
    @test_throws ArgumentError TMATile(BFloat16,(64,64),2,48)
    @test_throws ArgumentError TMATile(Float64,(64,16),2)
    @test_throws ArgumentError TMATile(UInt8,(64,64),2)
end

@testset "WGMMA ownership" begin
    for T in (BFloat16,Float16), n in 8:8:256
        p = WGMMAAtom(T,n)
        l = operand_layout(p,Accumulator())
        coordinates = [Tylo.Layouts.coordinate(l,Int32(t),e) for t in 0:127,e in 0:n÷2-1]
        @test Set(coordinates) == Set((r,c) for r in 0:63,c in 0:n-1)
        @test all(c -> c isa Tuple{Int32,Int32},coordinates)
        @test all(iszero,finish_mma(zero_accumulator(p)).data)
    end
    # One partial: the accumulator is a Fragment, finished already.
    one = zero_accumulator(WGMMAAtom(BFloat16,64))
    @test one isa Fragment{Float32} && Tylo.Layouts.layout(one) == operand_layout(WGMMAAtom(BFloat16,64),Accumulator())
    @test finish_mma(one) === one
    @test zero_accumulator(WGMMAAtom(BFloat16,8,64,4)) isa Tylo.WGMMAAccumulator
    for k in (16,32,64), partials in (1,k÷16)
        p = WGMMAAtom(BFloat16,8,k,partials)
        c = Tylo.WGMMAAccumulator(p,ntuple(Float32,4partials))
        f = finish_mma(c)
        @test f.data == ntuple(i -> sum(Float32(i+4j) for j in 0:partials-1),4)
        @test map(x -> 2f0*x,f).data == (f .* 2f0).data
    end
    @test_throws ArgumentError WGMMAAtom(Float32,8)
    @test_throws ArgumentError WGMMAAtom(BFloat16,9)
    @test_throws ArgumentError WGMMAAtom(BFloat16,8,48)
    @test_throws ArgumentError WGMMAAtom(BFloat16,8,64,3)
    @test_throws ArgumentError WGMMAAtom(BFloat16,256,64,4)
    @test_throws ArgumentError Tylo.WGMMAAccumulator(WGMMAAtom(BFloat16,8),(0f0,))
end

@testset "WGMMA descriptor geometry" begin
    for k in (16,32,64), n in (8,16,64,256)
        p = WGMMAAtom(BFloat16,n,k)
        a = shared_layout(TMATile(BFloat16,(128,64),2))
        b = shared_layout(TMATile(BFloat16,(64,256),1))
        for kk in 0:16:64-k, rr in (0,8,64)
            @test validate_wgmma(p,OperandA(),a,(rr,kk)) === nothing
        end
        for kk in 0:16:64-k
            @test validate_wgmma(p,OperandB(),b,(kk,256-n)) === nothing
        end
        # Swapped, each tile is MN-major for the other role: valid for
        # 16-bit inputs where the outer extent fits the contiguous rows.
        @test validate_wgmma(p,OperandA(),b) === nothing
        if n <= 64
            @test validate_wgmma(p,OperandB(),a) === nothing
        else
            @test_throws ArgumentError validate_wgmma(p,OperandB(),a)
        end
        f = WGMMAAtom(Float8_E4M3FN,n,2k)                  # FP8 operands are K-major only
        @test_throws ArgumentError validate_wgmma(f,OperandA(),shared_layout(TMATile(Float8_E4M3FN,(64,256),1)))
        @test_throws ArgumentError validate_wgmma(p,OperandA(),a,(1,0))
        @test_throws ArgumentError validate_wgmma(p,OperandA(),a,(0,8))
        @test_throws ArgumentError validate_wgmma(p,OperandA(),a,(72,0))
        @test_throws ArgumentError validate_wgmma(p,OperandA(),a,(0,64))
    end
    # Narrower rows: the plan's K must fit one row, origins stay inside it.
    for W in (32,64), k in (16,32,64)
        row = W ÷ 2
        a = shared_layout(TMATile(BFloat16,(64,row),2,W))
        p = WGMMAAtom(BFloat16,16,k)
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
