@testset "Adjacent C atoms supply A in the same lanes" begin
    for T in (BFloat16,Float16),wm in (1,2),rm in (1,2),rn in (2,4)
        atom=MMA16x8x16(T)
        p=TiledMMA(atom,Val((wm,1)),Val((rm,rn)),Val(16))
        c=Tylo.Layouts.layout(zero_accumulator(p))
        a=operand_layout(atom,OperandA())
        for tid in 0:32wm-1,m in 0:rm-1,k in 0:rn÷2-1,e in 0:7
            # Independent ISA formula, including repeated M and paired N atoms.
            expected=(16rm*(tid÷32)+16m+(tid%32)÷4+8*((e%4)÷2),
                      16k+2*(tid%4)+e%2+8*(e÷4))
            src=4*(m+(2k+e÷4)*rm)+e%4
            ac=Tylo.Layouts.coordinate(a,tid%32,Val(e))
            @test Tylo.Layouts.coordinate(c,tid,Val(src)) == expected
            @test (ac[1]+16rm*(tid÷32)+16m,ac[2]+16k) == expected
        end
    end
    atom=MMA16x8x16(BFloat16)
    for (w,r) in (((1,2),(1,2)),((1,1),(1,3)))
        a=zero_accumulator(TiledMMA(atom,Val(w),Val(r),Val(16)))
        @test_throws ArgumentError pack_operand_a(atom,a,Val(0),Val(0))
    end
    a=zero_accumulator(TiledMMA(atom,Val((1,1)),Val((2,4)),Val(16)))
    @test_throws BoundsError pack_operand_a(atom,a,Val(2),Val(0))
    @test_throws BoundsError pack_operand_a(atom,a,Val(0),Val(2))
end
