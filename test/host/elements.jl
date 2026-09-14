@testset "FP8 element formats" begin
    for T in (Float8E4M3,Float8E5M2)
        # Every finite pattern survives decode/encode; NaN stays NaN; infinities saturate like cvt.satfinite.
        for b in 0x00:0xff
            x = reinterpret(T,b)
            if isnan(x)
                @test isnan(T(Float32(x)))
            elseif !isfinite(x)
                @test Float32(T(Float32(x))) == copysign(Float32(floatmax(T)),Float32(x))
            else
                @test reinterpret(UInt8,T(Float32(x))) == b
            end
        end
        finite = sort([Float32(reinterpret(T,b)) for b in 0x00:0x7f if isfinite(reinterpret(T,b))])
        @test issorted(finite) && finite[1] == 0f0
        # Round to nearest even between neighbours.
        for i in 2:length(finite)-1
            lo,hi = finite[i],finite[i+1]
            mid = Float32((Float64(lo)+Float64(hi))/2) # exactly representable in FP32
            @test Float32(T(prevfloat(mid))) == lo
            @test Float32(T(nextfloat(mid))) == hi
            # Ties go to the even mantissa.
            tie = Float32(T(mid))
            @test tie == (iseven(reinterpret(UInt8,T(lo))) ? lo : hi)
        end
        @test Float32(T(1f6)) == Float32(floatmax(T)) && Float32(T(-1f6)) == -Float32(floatmax(T))
        @test T(0.5) == T(0.5f0) && Float32(-T(2)) == -2f0
        @test zero(T) == T(0) && Float32(one(T)) == 1f0
    end
    @test reinterpret(UInt8,Float8E4M3(1f0)) == 0x38 && reinterpret(UInt8,Float8E5M2(1f0)) == 0x3c
    @test Float32(floatmax(Float8E4M3)) == 448f0 && Float32(floatmax(Float8E5M2)) == 57344f0
    @test isnan(reinterpret(Float8E4M3,0x7f)) && !isfinite(reinterpret(Float8E5M2,0x7c))
    @test Float32(Float8E5M2(3f0)) == 3f0 && Float32(Float8E4M3(3.0625f0)) in (3f0,3.25f0)
end

@testset "8-bit and 16-bit register packing" begin
    for (T,values) in ((Int8,ntuple(i -> Int8(i-5),8)),(UInt8,ntuple(i -> UInt8(240+i),8)),
                       (Float8E4M3,ntuple(i -> Float8E4M3(i/4),8)),(Float16,ntuple(i -> Float16(i),4)))
        f = Fragment(values,Tylo.Layouts.LocalOwnership{length(values),2}())
        p = pack(f)
        @test length(p.data) == length(values)*Tylo._element_bits(T)÷32
        @test unpack(p).data == values
        @test eltype(p) === T
    end
    @test pack(Fragment((Int8(1),Int8(2),Int8(3),Int8(4)),Tylo.Layouts.LocalOwnership{4,2}())).data == (0x04030201,)
    @test_throws ArgumentError pack(Fragment((Int8(1),Int8(2)),Tylo.Layouts.LocalOwnership{2,2}()))
    @test_throws ArgumentError pack(Fragment((1f0,2f0),Tylo.Layouts.LocalOwnership{2,2}()))
    q = pack(Fragment(ntuple(i -> Int8(i),16),Tylo.Layouts.LocalOwnership{16,2}()))
    @test window(q,Val((0,4)),Val((32,8))).data == q.data[2:3]
    @test_throws ArgumentError window(q,Val((0,2)),Val((32,8)))
    converted = pack(Float8E4M3,Fragment((1f0,-2f0,0.5f0,1000f0),Tylo.Layouts.LocalOwnership{4,2}()))
    @test Float32.(unpack(converted).data) == (1f0,-2f0,0.5f0,448f0)
end

@testset "Atom acceptance" begin
    for (shape,TA,TC) in (((16,8,8),BFloat16,Float32),((16,8,16),Float16,Float16),((16,8,4),Float32,Float32),
                          ((16,8,8),Float32,Float32),((16,8,32),Float8E4M3,Float32),((16,8,16),Int8,Int32),((16,8,32),UInt8,Int32))
        a = MMAAtom(shape,TA,TC)
        @test size(a) == shape && eltype(a,Accumulator()) === TC
        for role in (OperandA(),OperandB(),Accumulator())
            o = operand_layout(a,role)
            @test Tylo.is_complete(o) && Tylo.is_injective(o)
        end
        @test Tylo._words(a,OperandA()) == 2*Tylo._words(a,OperandB())
    end
    @test MMAAtom((16,8,32),Float8E4M3,Float8E5M2,Float32) isa MMAAtom
    @test MMAAtom((16,8,16),Int8,UInt8,Int32) isa MMAAtom
    @test MMAAtom((16,8,16),Int8) isa MMAAtom{(16,8,16),Int8,Int8,Int32}
    @test_throws ArgumentError MMAAtom((16,8,16),Float8E4M3,Int8,Float32)
    @test_throws ArgumentError MMAAtom((16,8,16),BFloat16,Float16,Float32)
    @test_throws ArgumentError MMAAtom((16,8,16),Int8,Int8,Float32)
    @test_throws ArgumentError MMAAtom((16,8,16),BFloat16,BFloat16,Float16)
    @test_throws ArgumentError MMAAtom((16,8,64),Int8)
    @test_throws ArgumentError MMAAtom((16,8,16),Float32)
    @test length(Tylo.instruction_atoms()) == 24
end
