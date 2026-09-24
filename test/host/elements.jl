@testset "FP8 element formats" begin
    # Tylo converts into the Microfloats types the way `cvt.rn.satfinite` does,
    # whatever overflow policy the type declares.
    enc(T,x) = Tylo.convert_element(T,x)
    for T in (Float8_E4M3FN,Float8_E5M2)
        # Every finite pattern survives decode/encode; NaN stays NaN; infinities saturate like cvt.satfinite.
        for b in 0x00:0xff
            x = reinterpret(T,b)
            if isnan(x)
                @test isnan(enc(T,Float32(x)))
            elseif !isfinite(x)
                @test Float32(enc(T,Float32(x))) == copysign(Float32(floatmax(T)),Float32(x))
            else
                @test reinterpret(UInt8,enc(T,Float32(x))) == b
            end
        end
        finite = sort([Float32(reinterpret(T,b)) for b in 0x00:0x7f if isfinite(reinterpret(T,b))])
        @test issorted(finite) && finite[1] == 0f0
        # Round to nearest even between neighbours.
        for i in 2:length(finite)-1
            lo,hi = finite[i],finite[i+1]
            mid = Float32((Float64(lo)+Float64(hi))/2) # exactly representable in FP32
            @test Float32(enc(T,prevfloat(mid))) == lo
            @test Float32(enc(T,nextfloat(mid))) == hi
            # Ties go to the even mantissa.
            tie = Float32(enc(T,mid))
            @test tie == (iseven(reinterpret(UInt8,enc(T,lo))) ? lo : hi)
        end
        @test Float32(enc(T,1f6)) == Float32(floatmax(T)) && Float32(enc(T,-1f6)) == -Float32(floatmax(T))
        @test enc(T,Inf32) == floatmax(T) && enc(T,-Inf32) == -floatmax(T)
        @test enc(T,0.5) == enc(T,0.5f0) && Float32(-enc(T,2)) == -2f0
        @test enc(T,enc(T,1.5f0)) === enc(T,1.5f0)
        # 16-bit sources convert like their FP32 value.
        for x in (0.3f0,-7.25f0,500f0,6f4,-1f9)
            @test enc(T,Float16(x)) === enc(T,Float32(Float16(x)))
            @test enc(T,BFloat16(x)) === enc(T,Float32(BFloat16(x)))
        end
    end
    @test Tylo.convert_element(Int8,5) === Int8(5) && Tylo.convert_element(BFloat16,1.5f0) === BFloat16(1.5f0)
    @test reinterpret(UInt8,Float8_E4M3FN(1f0)) == 0x38 && reinterpret(UInt8,Float8_E5M2(1f0)) == 0x3c
    @test Float32(floatmax(Float8_E4M3FN)) == 448f0 && Float32(floatmax(Float8_E5M2)) == 57344f0
    @test isnan(reinterpret(Float8_E4M3FN,0x7f)) && !isfinite(reinterpret(Float8_E5M2,0x7c))
    @test Float32(Float8_E5M2(3f0)) == 3f0 && Float32(Float8_E4M3FN(3.0625f0)) in (3f0,3.25f0)
end

@testset "8-bit and 16-bit register packing" begin
    for (T,values) in ((Int8,ntuple(i -> Int8(i-5),8)),(UInt8,ntuple(i -> UInt8(240+i),8)),
                       (Float8_E4M3FN,ntuple(i -> Float8_E4M3FN(i/4),8)),(Float16,ntuple(i -> Float16(i),4)))
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
    converted = pack(Float8_E4M3FN,Fragment((1f0,-2f0,0.5f0,1000f0),Tylo.Layouts.LocalOwnership{4,2}()))
    @test Float32.(unpack(converted).data) == (1f0,-2f0,0.5f0,448f0)
    # Numeric packing saturates from every source type and is the identity on its own type.
    for S in (Float32,Float16,BFloat16,Float64), T in (Float8_E4M3FN,Float8_E5M2)
        values = (S(1),S(-2.5),S(0.5),S(3e4),S(-Inf),S(0),S(448),S(-0.0))
        p = pack(T,Fragment(values,Tylo.Layouts.LocalOwnership{8,2}()))
        @test unpack(p).data == map(x -> Tylo.convert_element(T,x),values)
        @test pack(T,unpack(p)).data == p.data
        # Widening unpack is exact and keeps ownership.
        for F in (Float16,BFloat16,Float32,Float64)
            w = unpack(F,p)
            @test w.data == map(F,unpack(p).data) && eltype(w.data) === F
            @test Tylo.Layouts.layout(w) === Tylo.Layouts.layout(p)
        end
    end
end

@testset "Atom acceptance" begin
    for (shape,TA,TC) in (((16,8,8),BFloat16,Float32),((16,8,16),Float16,Float16),((16,8,4),Float32,Float32),
                          ((16,8,8),Float32,Float32),((16,8,32),Float8_E4M3FN,Float32),((16,8,16),Int8,Int32),((16,8,32),UInt8,Int32))
        a = MMAAtom(shape,TA,TC)
        @test size(a) == shape && eltype(a,Accumulator()) === TC
        for role in (OperandA(),OperandB(),Accumulator())
            o = operand_layout(a,role)
            @test Tylo.is_complete(o) && Tylo.is_injective(o)
        end
        @test Tylo._words(a,OperandA()) == 2*Tylo._words(a,OperandB())
    end
    @test MMAAtom((16,8,32),Float8_E4M3FN,Float8_E5M2,Float32) isa MMAAtom
    @test MMAAtom((16,8,16),Int8,UInt8,Int32) isa MMAAtom
    @test MMAAtom((16,8,16),Int8) isa MMAAtom{(16,8,16),Int8,Int8,Int32}
    @test_throws ArgumentError MMAAtom((16,8,16),Float8_E4M3FN,Int8,Float32)
    @test_throws ArgumentError MMAAtom((16,8,16),BFloat16,Float16,Float32)
    @test_throws ArgumentError MMAAtom((16,8,16),Int8,Int8,Float32)
    @test_throws ArgumentError MMAAtom((16,8,16),BFloat16,BFloat16,Float16)
    @test_throws ArgumentError MMAAtom((16,8,64),Int8)
    @test_throws ArgumentError MMAAtom((16,8,16),Float32)
    @test length(Tylo.instruction_atoms()) == 24
end

@testset "Block-scaled atom acceptance" begin
    for TA in (Float8_E4M3FN,Float8_E5M2), TB in (Float8_E4M3FN,Float8_E5M2)
        a = ScaledMMAAtom((16,8,32),TA,TB)
        @test size(a) == (16,8,32) && Tylo.threads(a) == 32
        @test eltype(a,OperandA()) === TA && eltype(a,OperandB()) === TB && eltype(a,Accumulator()) === Float32
        @test eltype(a,ScaleA()) === eltype(a,ScaleB()) === Float8_E8M0FNU
        dense = MMAAtom((16,8,32),TA,TB,Float32)
        for role in (OperandA(),OperandB(),Accumulator())
            @test operand_layout(a,role) === operand_layout(dense,role)
        end
        # One register of four scales per lane; every scale is held, by more than one lane.
        for (role,shape) in ((ScaleA(),(16,4)),(ScaleB(),(4,8)))
            o = operand_layout(a,role)
            @test size(o) == shape && Tylo._register_count(o) == 4
            @test Tylo.is_complete(o) && !Tylo.is_injective(o)
        end
        # The measured selectors: lanes 4q and 4q+1 hold rows q and q+8, lane bit 1
        # replicates, and every lane of quad q holds column q; slot s is K block s.
        for lane in 0:31, s in 0:3
            q,l = divrem(lane,4)
            @test Tylo.Layouts.coordinate(operand_layout(a,ScaleA()),lane,Val(s)) == (q+8*(l&1),s)
            @test Tylo.Layouts.coordinate(operand_layout(a,ScaleB()),lane,Val(s)) == (s,q)
        end
    end
    @test ScaledMMAAtom((16,8,32),Float8_E4M3FN) isa ScaledMMAAtom{(16,8,32),Float8_E4M3FN,Float8_E4M3FN,Float8_E8M0FNU}
    @test_throws ArgumentError ScaledMMAAtom((16,8,16),Float8_E4M3FN)
    @test_throws ArgumentError ScaledMMAAtom((16,8,32),BFloat16)
    @test_throws ArgumentError ScaledMMAAtom((16,8,32),Float8_E4M3FN,Int8)
    @test_throws ArgumentError ScaledMMAAtom((16,8,32),Float8_E4M3FN,Float8_E4M3FN,Float8_E5M2)
    @test length(scaled_instruction_atoms()) == 4
    @test Tylo._element_bits(Float8_E8M0FNU) == 8
end
