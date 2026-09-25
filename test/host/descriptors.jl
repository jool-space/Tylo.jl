# Descriptor operands against the PTX ISA, without the hardware. Each
# descriptor Tylo emits, for every instruction of a K loop, is decoded with
# the specification's bit layout (§9.7.17.5.1.2.2 for WGMMA, Table 49 for
# tcgen05), independently of PTX.jl's encoder. The canonical layouts of
# §9.7.18.3.3 then give the shared-memory byte the tensor core reads for each
# element; it must be the byte where the tile stores that element.
using Tylo.Layouts: @Layout, Layout, Swizzle, compose, static

field(d,lo,n) = Int((d >> lo) & ((UInt64(1) << n) - 1))
function decode_wgmma(d::UInt64)
    (; start=field(d,0,14) << 4, lbo=field(d,16,14) << 4, sbo=field(d,32,14) << 4, base=field(d,49,3),
       swizzle=(0,128,64,32)[field(d,62,2)+1], reserved=field(d,14,2) | field(d,30,2) | field(d,46,3) | field(d,52,10))
end
function decode_tcgen05(d::UInt64)
    mode = field(d,61,3)
    (; start=field(d,0,14) << 4, lbo=field(d,16,14) << 4, sbo=field(d,32,14) << 4, base=field(d,49,3),
       swizzle=get(Dict(0 => 0, 2 => 128, 4 => 64, 6 => 32),mode,-1), constant=field(d,46,3),
       reserved=field(d,14,2) | field(d,30,2) | field(d,52,9))
end

# The swizzle of an absolute byte address, CuTe's Swizzle<B,4,3>: the 16-byte
# chunk bits XOR the row bits of the W-byte pattern.
swizzled(a,W) = W == 0 ? a : a ⊻ (((a >> 7) & (W ÷ 16 - 1)) << 4)
# The canonical layout, in elements, of an MN × K operand: `r` along M or N,
# `k` along K. t elements make 16 bytes, c 16-byte chunks make a W-byte row.
function canonical(major,W,::Type{T},lbo,sbo,r,k) where T
    t, c = 16 ÷ sizeof(T), W ÷ 16
    lbo, sbo = lbo ÷ sizeof(T), sbo ÷ sizeof(T)
    major === :K ? (r % 8)*c*t + (r ÷ 8)*sbo + k :                       # ((8,m),(t,2k)):((c t,SBO),(1,t))
        (r % t) + ((r ÷ t) % c)*t + (r ÷ (t*c))*lbo + (k % 8)*c*t + (k ÷ 8)*sbo   # ((t,c,m),(8,k)):((1,t,LBO),(c t,SBO))
end
# Every element of one instruction's rows × kk operand block, starting at a
# logical coordinate of the tile, is read where the tile stores it.
function reads_tile(desc,major,tile,::Type{T},rows,kk,first,kaxis) where T
    W = desc.swizzle
    all(0:rows-1) do r
        all(0:kk-1) do k
            c = kaxis == 2 ? (first[1]+r,first[2]+k) : (first[1]+k,first[2]+r)
            stored = Int(Tylo._smem_address(tile.ptr)) + Int(tile.layout(map(Int32,c)))*sizeof(T)
            swizzled(desc.start + sizeof(T)*canonical(major,W,T,desc.lbo,desc.sbo,r,k),W) == stored
        end
    end
end
sptr(::Type{T},a) where T = reinterpret(Core.LLVMPtr{T,3},UInt64(a))

@testset "WGMMA descriptors read the tile (PTX ISA canonical layouts)" begin
    for T in (BFloat16,Float16), W in (32,64,128), base in (0x400,0x8000)
        row = W ÷ sizeof(T)
        a = SharedTile(sptr(T,base),shared_layout(TMATile(T,(128,row),2,W)))
        b = SharedTile(sptr(T,base+0x4000),shared_layout(TMATile(T,(row,64),1,W)))
        for k in (16,32,64), n in (8,16,64)
            k <= row || continue
            p = WGMMAAtom(T,n,k)
            for rr in (0,64), kk in 0:16:row-k
                o = operand(p,OperandA(),a,(Int32(rr),Int32(kk)))
                for j in 0:k÷16-1
                    d = decode_wgmma(Tylo._k_step(o,Val(j)))
                    @test d.swizzle == W && d.base == 0 && d.reserved == 0
                    @test reads_tile(d,:K,a,T,64,16,(rr,kk+16j),2)
                end
            end
            for kk in 0:16:row-k, cc in (0,64-n)
                o = operand(p,OperandB(),b,(Int32(kk),Int32(cc)))
                for j in 0:k÷16-1
                    d = decode_wgmma(Tylo._k_step(o,Val(j)))
                    @test d.swizzle == W && d.base == 0
                    @test reads_tile(d,:K,b,T,n,16,(kk+16j,cc),1)
                end
            end
        end
    end
end

# Shared operands for tcgen05: K-major and MN-major, one row-width group or
# several side by side, for every input kind.
function tcgen05_tile(::Type{T},W,role,major,groups,base) where T
    row = W ÷ sizeof(T)
    s(x) = static(x)
    sw = Swizzle{trailing_zeros(W ÷ 16),trailing_zeros(16 ÷ sizeof(T)),3}()
    outer = 128                                     # rows along M or N
    if major === :K                                 # K contiguous, groups of rows along K
        kext = row*groups
        l = Layout((s(outer),(s(row),s(groups))),(s(row),(s(1),s(row*outer))))
        l = role isa OperandA ? l : Layout(reverse(l.shape),reverse(l.strides))
    else                                            # M or N contiguous, groups along it
        kext = 64
        l = Layout(((s(row),s(outer ÷ row)),s(kext)),((s(1),s(row*kext)),s(row)))
        l = role isa OperandA ? l : Layout(reverse(l.shape),reverse(l.strides))
    end
    SharedTile(sptr(T,base),compose(sw,l)), kext
end

@testset "tcgen05 descriptors read the tile (PTX ISA canonical layouts)" begin
    for T in (BFloat16,Float16,Float32,Float8_E4M3FN,Int8), W in (32,64,128), major in (:K,:MN), groups in (1,2)
        (major === :MN && groups == 2) && continue
        TC = T === Int8 ? Int32 : Float32
        katom = 256 ÷ (8sizeof(T))
        row = W ÷ sizeof(T)
        major === :MN && row > 128 && continue
        for role in (OperandA(),OperandB())
            n = 64
            atom = Tcgen05Atom((128,n,katom),T,T,TC)
            tile, kext = tcgen05_tile(T,W,role,major,groups,0x800)
            kaxis = role isa OperandA ? 2 : 1
            outer = role isa OperandA ? 128 : n
            starts = major === :K ? (0:katom:kext-katom) : (0:8:kext-katom)
            for k0 in first(starts,3), m0 in (major === :K ? (0,) : (0,))
                origin = kaxis == 2 ? (Int32(m0),Int32(k0)) : (Int32(k0),Int32(m0))
                o = operand(atom,role,tile,origin)
                @test typeof(o).parameters[3] === major
                for dk in 0:katom:min(kext-k0,2katom)-katom
                    d = decode_tcgen05(Tylo._k_step(o,Val(dk)))
                    @test d.swizzle == W && d.constant == 0b001 && d.base == 0 && d.reserved == 0
                    first_c = kaxis == 2 ? (m0,k0+dk) : (k0+dk,m0)
                    @test reads_tile(d,major,tile,T,outer,katom,first_c,kaxis)
                end
            end
        end
    end
end

# The instruction descriptor (Table 51), decoded bit by bit.
@testset "tcgen05 instruction descriptors (PTX ISA Table 51)" begin
    types = Dict(BFloat16 => (1,1), Float16 => (1,0), Float32 => (1,2), Float8_E4M3FN => (1,0),
                 Float8_E5M2 => (1,1), Int8 => (2,1), UInt8 => (2,0))
    for (TA,TB) in ((BFloat16,BFloat16),(Float16,Float16),(Float32,Float32),(Float8_E4M3FN,Float8_E5M2),(Int8,UInt8)),
            n in (16,64,256), am in (:K,:MN), bm in (:K,:MN)
        TC = TA <: Integer ? Int32 : Float32
        atom = Tcgen05Atom((128,n,256 ÷ (8sizeof(TA))),TA,TB,TC)
        d = UInt64(Tylo.instruction_descriptor(atom;a_major=am,b_major=bm))
        @test field(d,0,4) == 0                        # dense, no saturation
        @test field(d,4,2) == types[TA][1]             # D: F32 = 1, S32 = 2
        @test field(d,7,3) == types[TA][2] && field(d,10,3) == types[TB][2]
        @test field(d,13,2) == 0                       # no negation
        @test field(d,15,1) == (am === :MN) && field(d,16,1) == (bm === :MN)
        @test field(d,17,6) == n >> 3 && field(d,23,1) == 0 && field(d,24,5) == 128 >> 4
        @test field(d,29,3) == 0
    end
end

# Operands straight from TMA storage, several stripes along K included.
@testset "Descriptors of multi-stripe TMA tiles" begin
    for T in (BFloat16,Float8_E4M3FN), stripes in (1,2)
        row = 128 ÷ sizeof(T)
        katom = 256 ÷ (8sizeof(T))
        a = shared_tile(TMATile(T,(128,stripes*row),2),sptr(T,0x400))
        b = shared_tile(TMATile(T,(stripes*row,64),1),sptr(T,0x400+128*stripes*128))
        atom = Tcgen05Atom((128,64,katom),T,T,Float32)
        for k0 in 0:katom:stripes*row-katom
            oa, ob = operand(atom,OperandA(),a,(0,k0)), operand(atom,OperandB(),b,(k0,0))
            @test reads_tile(decode_tcgen05(Tylo._k_step(oa,Val(0))),:K,a,T,128,katom,(0,k0),2)
            @test reads_tile(decode_tcgen05(Tylo._k_step(ob,Val(0))),:K,b,T,64,katom,(k0,0),1)
        end
        T === BFloat16 || continue
        p = WGMMAAtom(T,64)
        for k0 in 0:64:stripes*row-64, j in 0:3
            @test reads_tile(decode_wgmma(Tylo._k_step(operand(p,OperandA(),a,(0,k0)),Val(j))),:K,a,T,64,16,(0,k0+16j),2)
        end
    end
end
