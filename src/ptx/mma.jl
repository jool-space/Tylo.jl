# Instruction bindings for the atoms Tylo implements. Every other operation
# on the atom derives from its ownership layouts. The table is the dense
# warp-MMA surface PTX.jl exposes for these element types.
const _PTX_NAMES = Dict(BFloat16=>"bf16",Float16=>"f16",Float32=>"tf32",Float8_E4M3FN=>"e4m3",
                        Float8_E5M2=>"e5m2",Int8=>"s8",UInt8=>"u8",Int32=>"s32")
_mma_instruction(::Type{A}) where A<:MMAAtom = nothing
for (shape,TA,TB,TC) in vcat(
        [((16,8,k),T,T,Float32) for k in (8,16) for T in (BFloat16,Float16)],
        [((16,8,k),Float16,Float16,Float16) for k in (8,16)],
        [((16,8,k),Float32,Float32,Float32) for k in (4,8)],
        [((16,8,k),TA,TB,Float32) for k in (16,32) for TA in (Float8_E4M3FN,Float8_E5M2) for TB in (Float8_E4M3FN,Float8_E5M2)],
        [((16,8,k),TA,TB,Int32) for k in (16,32) for TA in (Int8,UInt8) for TB in (Int8,UInt8)])
    m,n,k = shape
    tc = _PTX_NAMES[TC] === "tf32" ? "f32" : _PTX_NAMES[TC]
    name = "mma.sync.aligned.m$(m)n$(n)k$(k).row.col.$tc.$(_PTX_NAMES[TA]).$(_PTX_NAMES[TB]).$tc"
    instruction = Meta.parse("ptx\"$name\"")
    @eval _mma_instruction(::Type{MMAAtom{$shape,$TA,$TB,$TC}}) = $instruction
    push!(_INSTRUCTION_ATOMS,MMAAtom{shape,TA,TB,TC}())
end

# Word carriers: PTX.jl takes UInt32 words for A, B and scales, FP32 or
# Int32 accumulators as themselves and FP16 accumulators as UInt32 words;
# both directions are bit operations chosen while generating.
@inline _accumulator_words(c::Fragment{Float16}) = _pack_words(c)
@inline _accumulator_words(c::Fragment) = c.data
@inline _accumulator_fragment(::Type{Float16},d::NTuple{N,UInt32},ownership) where N = _from_words(Float16,d,ownership)
@inline _accumulator_fragment(::Type{T},d::NTuple{N,T},ownership) where {T,N} = Fragment(d,ownership)

# Operand ownerships are checked while generating: a fragment from any load
# whose ownership enumerates to the atom's operand ownership is accepted.
@generated function mma(atom::A,a::FA,b::FB,c::FC) where {A<:MMAAtom,FA,FB,FC}
    instruction = _mma_instruction(A)
    instruction === nothing && return :(throw(ArgumentError("no instruction binding for this MMA atom")))
    atom_instance = A()
    for (F,role) in ((FA,OperandA()),(FB,OperandB()),(FC,Accumulator()))
        F <: Fragment || return :(throw(ArgumentError("MMA operands are fragments")))
        eltype(F) === eltype(atom_instance,role) ||
            return :(throw(ArgumentError("MMA operand element type differs from the atom")))
        same_distribution(_static_instance(F.parameters[3]),operand_layout(atom_instance,role)) ||
            return :(throw(ArgumentError("MMA operand ownership differs from the atom")))
    end
    TC = eltype(atom_instance,Accumulator())
    quote
        Base.@inline
        d = $instruction(_pack_words(a),_pack_words(b),_accumulator_words(c))
        _accumulator_fragment($TC,d,operand_layout(atom,Accumulator()))
    end
end

# Block-scaled atoms: `.kind::mxf8f6f4` with one UE8M0 scale per row and column
# (`.scale_vec::1X`). The byte selectors take the block; the thread selectors
# are zero, which the replicated scale ownerships make as good as any.
_scaled_mma_instruction(::Type{A}) where A<:ScaledMMAAtom = nothing
for TA in (Float8_E4M3FN,Float8_E5M2), TB in (Float8_E4M3FN,Float8_E5M2)
    name = "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32." *
           "$(_PTX_NAMES[TA]).$(_PTX_NAMES[TB]).f32.ue8m0"
    instruction = Meta.parse("ptx\"$name\"")
    @eval _scaled_mma_instruction(::Type{ScaledMMAAtom{(16,8,32),$TA,$TB,Float8_E8M0FNU}}) = $instruction
    push!(_INSTRUCTION_ATOMS,ScaledMMAAtom{(16,8,32),TA,TB,Float8_E8M0FNU}())
end
@generated function mma(atom::A,a::FA,b::FB,c::FC,sa::FSA,sb::FSB,block::Integer) where {A<:ScaledMMAAtom,FA,FB,FC,FSA,FSB}
    instruction = _scaled_mma_instruction(A)
    instruction === nothing && return :(throw(ArgumentError("no instruction binding for this block-scaled MMA atom")))
    atom_instance = A()
    for (F,role) in ((FA,OperandA()),(FB,OperandB()),(FC,Accumulator()),(FSA,ScaleA()),(FSB,ScaleB()))
        F <: Fragment || return :(throw(ArgumentError("MMA operands are fragments")))
        eltype(F) === eltype(atom_instance,role) ||
            return :(throw(ArgumentError("MMA operand element type differs from the atom")))
        same_distribution(_static_instance(F.parameters[3]),operand_layout(atom_instance,role)) ||
            return :(throw(ArgumentError("MMA operand ownership differs from the atom")))
    end
    quote
        Base.@inline
        @boundscheck 0 <= block < SCALE_BLOCKS || throw(BoundsError())
        byte = block % UInt16
        d = $instruction(_pack_words(a),_pack_words(b),c.data,_pack_words(sa)[1],byte,0x0000,_pack_words(sb)[1],byte,0x0000)
        Fragment(d,operand_layout(atom,Accumulator()))
    end
end
@generated function mma(p::TiledMMA{A,W,R,K},
        a::SharedTile{TA},b::SharedTile{TB},
        accum::F,tid::Integer) where {A,W,R,K,TA,TB,F}
    _is_accumulator(TiledMMA{A,W,R,K},F) || return :(throw(ArgumentError("the accumulator is not this plan's")))
    atom = A()
    TA === eltype(atom,OperandA()) && TB === eltype(atom,OperandB()) ||
        return :(throw(ArgumentError("shared tiles must hold the atom's input types")))
    _element_bits(eltype(atom,Accumulator())) == 32 ||
        return :(throw(ArgumentError("tiled shared-tile MMA requires a 32-bit accumulator")))
    m,n,ka = size(atom)
    cs = [Symbol(:c_,i) for i in 1:prod(R)]
    statements = [:( $(cs[i]) = _atom_accumulator(p,accum,Val($i)) ) for i in eachindex(cs)]
    for k in 0:ka:K-ka
        av = [Symbol(:a_,i) for i in 1:R[1]]
        bv = [Symbol(:b_,j) for j in 1:R[2]]
        for i in 1:R[1]
            push!(statements,:($(av[i]) = load_fragment(operand_layout(p.atom,OperandA()),
                window(a,(wm*oftype(tid,$(m*R[1]))+oftype(tid,$(m*(i-1))),oftype(tid,$k)),($m,$ka)),lane)))
        end
        for j in 1:R[2]
            push!(statements,:($(bv[j]) = load_fragment(operand_layout(p.atom,OperandB()),
                window(b,(oftype(tid,$k),wn*oftype(tid,$(n*R[2]))+oftype(tid,$(n*(j-1)))),($ka,$n)),lane)))
        end
        for j in 1:R[2],i in 1:R[1]
            c = cs[i+(j-1)*R[1]]
            push!(statements,:($c = mma(p.atom,$(av[i]),$(bv[j]),$c)))
        end
    end
    values = [:($c.data[$k]) for c in cs for k in 1:4]
    rows,cols = m*W[1]*R[1],n*W[2]*R[2]
    quote
        Base.@inline
        size(a) == ($rows,$K) && size(b) == ($K,$cols) || throw(DimensionMismatch("MMA plan/view shapes differ"))
        @boundscheck 0 <= tid < $(32prod(W)) || throw(BoundsError())
        lane = tid & oftype(tid,31)
        warp = tid >> 5
        wm,wn = warp % oftype(tid,$(W[1])),warp ÷ oftype(tid,$(W[1]))
        @inbounds begin
            $(statements...)
        end
        Fragment(($(values...),),Layouts.layout(accum))
    end
end

@generated function store!(p::TiledMMA{A,W,R,K},dst::GlobalTile{T},
        accum::F,tid::Integer) where {A,W,R,K,T,F}
    _is_accumulator(TiledMMA{A,W,R,K},F) || return :(throw(ArgumentError("the accumulator is not this plan's")))
    m,n,_ = size(A())
    T === eltype(A(),Accumulator()) ||
        return :(throw(ArgumentError("the destination must hold the accumulator type; the origin form converts")))
    stores = [:(store!(window(dst,
        (wm*oftype(tid,$(m*R[1]))+oftype(tid,$(m*(i-1))),
         wn*oftype(tid,$(n*R[2]))+oftype(tid,$(n*(j-1)))),($m,$n)),
        _atom_accumulator(p,accum,Val($(i+(j-1)*R[1]))),lane)) for j in 1:R[2] for i in 1:R[1]]
    quote
        Base.@inline
        size(dst) == ($(m*W[1]*R[1]),$(n*W[2]*R[2])) || throw(DimensionMismatch("output plan/view shapes differ"))
        @boundscheck 0 <= tid < $(32prod(W)) || throw(BoundsError())
        lane = tid & oftype(tid,31)
        warp = tid >> 5
        wm,wn = warp % oftype(tid,$(W[1])),warp ÷ oftype(tid,$(W[1]))
        @inbounds begin
            $(stores...)
        end
        nothing
    end
end

# Bounds-aware epilogue: use precisely the ownership map of the accumulator.
# A full destination view supplies the logical bounds; origin may select an
# edge tile. No invalid global pointer is formed, even for a completely masked
# thread. Conversion to the destination element type occurs at the final store.
@inline function store!(p::TiledMMA{A,W,R,K},dst::GlobalTile{T},
        acc::Fragment,origin::Tuple,tid::Integer) where {A,W,R,K,T}
    length(origin)==2 || throw(ArgumentError("store origin must have two coordinates"))
    _same_distribution(Layouts.layout(acc),operand_layout(p,Accumulator())) ||
        throw(ArgumentError("the accumulator is not this plan's"))
    @boundscheck 0 <= tid < 32prod(W) || throw(BoundsError())
    _bounded_store!(dst,acc,origin,tid)
end
# Each slot's coordinate is the thread's part plus a constant: form the
# thread's origin once, then check and address each slot from it.
@generated function _bounded_store!(dst::GlobalTile{T,ML},acc::Fragment{S,N,L},origin,tid) where {T,ML,S,N,L}
    o = _static_instance(L)
    stores = map(0:N-1) do e
        cr, cc = Layouts._slot_coordinate(o,e)
        address = _affine(ML) ? :(_slot_address(dst,pointer(dst,base),($cr,$cc))) : :(pointer(dst,q))
        quote
            q = (base[1]+$cr,base[2]+$cc)
            if _valid_coordinate(dst,q)
                unsafe_store!($address,convert_element(T,acc.data[$(e+1)]))
            end
        end
    end
    quote
        Base.@inline
        thread = Layouts._thread_coordinate(Layouts.layout(acc),tid)
        base = (Int(origin[1])+Int(thread[1]),Int(origin[2])+Int(thread[2]))
        $(stores...)
        nothing
    end
end

@inline _pack_mma_pair(::Type{BFloat16},lo,hi) = PTX.bf16x2_pack(lo,hi)
@inline _pack_mma_pair(::Type{Float16},lo,hi) = ptx"cvt.rn.f16x2.f32"(hi,lo)
