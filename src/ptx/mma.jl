# Instruction bindings for the atoms Tylo implements. Every other operation
# on the atom derives from its ownership layouts. The table is the dense
# warp-MMA surface PTX.jl exposes for these element types.
const _PTX_NAMES = Dict(BFloat16=>"bf16",Float16=>"f16",Float32=>"tf32",Float8E4M3=>"e4m3",
                        Float8E5M2=>"e5m2",Int8=>"s8",UInt8=>"u8",Int32=>"s32")
_mma_instruction(::Type{A}) where A<:MMAAtom = nothing
for (shape,TA,TB,TC) in vcat(
        [((16,8,k),T,T,Float32) for k in (8,16) for T in (BFloat16,Float16)],
        [((16,8,k),Float16,Float16,Float16) for k in (8,16)],
        [((16,8,k),Float32,Float32,Float32) for k in (4,8)],
        [((16,8,k),TA,TB,Float32) for k in (16,32) for TA in (Float8E4M3,Float8E5M2) for TB in (Float8E4M3,Float8E5M2)],
        [((16,8,k),TA,TB,Int32) for k in (16,32) for TA in (Int8,UInt8) for TB in (Int8,UInt8)])
    m,n,k = shape
    tc = _PTX_NAMES[TC] === "tf32" ? "f32" : _PTX_NAMES[TC]
    name = "mma.sync.aligned.m$(m)n$(n)k$(k).row.col.$tc.$(_PTX_NAMES[TA]).$(_PTX_NAMES[TB]).$tc"
    instruction = Meta.parse("ptx\"$name\"")
    @eval _mma_instruction(::Type{MMAAtom{$shape,$TA,$TB,$TC}}) = $instruction
    push!(_INSTRUCTION_ATOMS,MMAAtom{shape,TA,TB,TC}())
end

# Word carriers: PTX.jl takes UInt32 words for A and B, FP32 or Int32
# accumulators as themselves and packed FP16 accumulators as UInt32 words;
# both directions are bit reinterpretations chosen while generating.
_accumulator_carrier(::Type{Float32}) = Float32
_accumulator_carrier(::Type{Int32}) = Int32
_accumulator_carrier(::Type{Float16}) = UInt32
@inline _as_words(f::PackedFragment) = f.data
@inline _as_words(f::Fragment{Float32,N}) where N = @rtuple(i -> reinterpret(UInt32,f.data[i]), 1:N)
@inline _to_carrier(::Type{U},data::NTuple{N,U}) where {U,N} = data
@inline _to_carrier(::Type{U},data::NTuple{N,T}) where {U,N,T} = @rtuple(i -> reinterpret(U,data[i]), 1:N)
@inline _from_carrier(::Type{T},data::NTuple{N,T}) where {T,N} = data
@inline _from_carrier(::Type{T},data::NTuple{N,U}) where {T,N,U} = @rtuple(i -> reinterpret(T,data[i]), 1:N)

# Operand ownerships are checked while generating: a fragment from any load
# whose ownership enumerates to the atom's operand ownership is accepted.
@generated function mma(atom::A,a::FA,b::FB,c::FC) where {A<:MMAAtom,FA,FB,FC}
    instruction = _mma_instruction(A)
    instruction === nothing && return :(throw(ArgumentError("no instruction binding for this MMA atom")))
    atom_instance = A()
    for (F,role) in ((FA,OperandA()),(FB,OperandB()),(FC,Accumulator()))
        F <: Union{Fragment,PackedFragment} || return :(throw(ArgumentError("MMA operands are fragments")))
        eltype(F) === eltype(atom_instance,role) ||
            return :(throw(ArgumentError("MMA operand element type differs from the atom")))
        L = F.parameters[3]
        same_distribution(_static_instance(L),operand_layout(atom_instance,role)) ||
            return :(throw(ArgumentError("MMA operand ownership differs from the atom")))
    end
    TC = eltype(atom_instance,Accumulator())
    carrier = _accumulator_carrier(TC)
    result = _element_bits(TC) == 32 ?
        :(Fragment(_from_carrier($TC,d),operand_layout(atom,Accumulator()))) :
        :(PackedFragment($TC,_from_carrier(UInt32,d),operand_layout(atom,Accumulator())))
    quote
        Base.@inline
        d = $instruction(_as_words(a),_as_words(b),_to_carrier($carrier,c.data))
        $result
    end
end

# Operand loads derive from the atom's ownership: ldmatrix where the
# plan exists, scalar loads otherwise. Windows select the operand tile.
Base.@propagate_inbounds function load_a(a::MMAAtom,t::SharedTile,lane::Integer)
    m,n,k = size(a)
    size(t) == (m,k) || throw(DimensionMismatch("A operand must be $(m)×$(k)"))
    eltype(t) === eltype(a,OperandA()) || throw(ArgumentError("A tile element type differs from the atom"))
    load_fragment(operand_layout(a,OperandA()),t,lane)
end
Base.@propagate_inbounds function load_b(a::MMAAtom,t::SharedTile,lane::Integer)
    m,n,k = size(a)
    size(t) == (k,n) || throw(DimensionMismatch("B operand must be $(k)×$(n)"))
    eltype(t) === eltype(a,OperandB()) || throw(ArgumentError("B tile element type differs from the atom"))
    load_fragment(operand_layout(a,OperandB()),t,lane)
end

@generated function mma(p::TiledMMA{A,W,R,K},
        a::SharedTile{TA},b::SharedTile{TB},
        accum::TiledAccumulator{TiledMMA{A,W,R,K}},tid::Integer) where {A,W,R,K,TA,TB}
    atom = A()
    TA === eltype(atom,OperandA()) && TB === eltype(atom,OperandB()) ||
        return :(throw(ArgumentError("shared tiles must hold the atom's input types")))
    _element_bits(eltype(atom,Accumulator())) == 32 ||
        return :(throw(ArgumentError("tiled shared-tile MMA requires a 32-bit accumulator")))
    m,n,ka = size(atom)
    cs = [Symbol(:c_,i) for i in 1:prod(R)]
    statements = [:( $(cs[i]) = _atom_accumulator(accum,Val($i)) ) for i in eachindex(cs)]
    for k in 0:ka:K-ka
        av = [Symbol(:a_,i) for i in 1:R[1]]
        bv = [Symbol(:b_,j) for j in 1:R[2]]
        for i in 1:R[1]
            push!(statements,:($(av[i]) = load_fragment(operand_layout(p.atom,OperandA()),
                window(a,(wm*oftype(tid,$(m*R[1]))+oftype(tid,$(m*(i-1))),oftype(tid,$k)),Val(($m,$ka))),lane)))
        end
        for j in 1:R[2]
            push!(statements,:($(bv[j]) = load_fragment(operand_layout(p.atom,OperandB()),
                window(b,(oftype(tid,$k),wn*oftype(tid,$(n*R[2]))+oftype(tid,$(n*(j-1)))),Val(($ka,$n))),lane)))
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
        accum::TiledAccumulator{TiledMMA{A,W,R,K}},tid::Integer) where {A,W,R,K,T}
    m,n,_ = size(A())
    T === eltype(A(),Accumulator()) ||
        return :(throw(ArgumentError("the destination must hold the accumulator type; the origin form converts")))
    stores = [:(store!(window(dst,
        (wm*oftype(tid,$(m*R[1]))+oftype(tid,$(m*(i-1))),
         wn*oftype(tid,$(n*R[2]))+oftype(tid,$(n*(j-1)))),Val(($m,$n))),
        _atom_accumulator(accum,Val($(i+(j-1)*R[1]))),lane)) for j in 1:R[2] for i in 1:R[1]]
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
        acc::TiledAccumulator{TiledMMA{A,W,R,K}},origin::Tuple,tid::Integer) where {A,W,R,K,T}
    length(origin)==2 || throw(ArgumentError("store origin must have two coordinates"))
    @boundscheck 0 <= tid < 32prod(W) || throw(BoundsError())
    @rtuple(0:4prod(R)-1) do e
        c = Layouts.coordinate(Layouts.layout(acc),tid,Val(e))
        q = (Int(origin[1])+Int(c[1]),Int(origin[2])+Int(c[2]))
        if _valid_coordinate(dst,q)
            unsafe_store!(pointer(dst,q),T(acc.data[e+1]))
        end
    end
    nothing
end

@inline _pack_mma_pair(::Type{BFloat16},lo,hi) = PTX.bf16x2_pack(lo,hi)
@inline _pack_mma_pair(::Type{Float16},lo,hi) = ptx"cvt.rn.f16x2.f32"(hi,lo)
