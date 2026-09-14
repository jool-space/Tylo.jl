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

Base.@propagate_inbounds function load_a(a::MMAAtom{(16,8,16),T},t::SharedTile{T},lane::Integer) where T
    size(t) == (16,16) || throw(DimensionMismatch("A operand must be 16×16"))
    @boundscheck 0 <= lane < 32 || throw(BoundsError())
    c = (lane & oftype(lane,15),(lane >> 4)*oftype(lane,8))
    @boundscheck _check_vector(t.layout,c,2,Val(8),T)
    data = ptx"ldmatrix.sync.aligned.m8n8.x4.shared.b16"(pointer(t,c))
    PackedFragment(T,data,operand_layout(a,OperandA()))
end
Base.@propagate_inbounds function load_b(a::MMAAtom{(16,8,16),T},t::SharedTile{T},lane::Integer) where T
    size(t) == (16,8) || throw(DimensionMismatch("B operand must be 16×8"))
    @boundscheck 0 <= lane < 32 || throw(BoundsError())
    # Lanes 16:31 also provide valid addresses, duplicating lanes 0:15.
    c = (((lane >> 3) & one(lane))*oftype(lane,8),lane & oftype(lane,7))
    @boundscheck _check_vector(t.layout,c,1,Val(8),T)
    data = ptx"ldmatrix.sync.aligned.m8n8.x2.shared.b16"(pointer(t,c))
    PackedFragment(T,data,operand_layout(a,OperandB()))
end

# Generic scalar loads and stores at the ownership's coordinates. They are
# correct for every static ownership and serve as the oracle for the
# instruction-specific paths.
@generated function load_fragment(::L,tile::MemoryTile{T},thread::Integer) where {L,T}
    o = _static_instance(L)
    o === nothing && return :(throw(ArgumentError("loads require a static ownership")))
    n = _register_count(o)
    loads = [:(unsafe_load(pointer(tile,Layouts.coordinate(ownership,thread,Val($e))),1,Val($(sizeof(T))))) for e in 0:n-1]
    packed = _element_bits(T) < 32
    quote
        Base.@inline
        ownership = $o
        values = Fragment(($(loads...),),ownership)
        $(packed ? :(pack(values)) : :(values))
    end
end
@generated function store!(tile::MemoryTile{T},f::Fragment{T,N,L},thread::Integer) where {T,N,L}
    o = _static_instance(L)
    o === nothing && return :(throw(ArgumentError("stores require a static ownership")))
    stores = [:(unsafe_store!(pointer(tile,Layouts.coordinate(ownership,thread,Val($e))),f.data[$(e+1)],1,Val($(sizeof(T))))) for e in 0:N-1]
    quote
        Base.@inline
        ownership = Layouts.layout(f)
        $(stores...)
        nothing
    end
end
@inline store!(tile::MemoryTile{T},f::PackedFragment{T},thread::Integer) where T =
    store!(tile,unpack(f),thread)

@generated function mma(p::TiledMMA{A,W,R,K},
        a::SharedTile{T},b::SharedTile{T},
        accum::TiledAccumulator{TiledMMA{A,W,R,K}},tid::Integer) where {A,W,R,K,T}
    size(A()) == (16,8,16) && _element_bits(T) == 16 ||
        return :(throw(ArgumentError("shared-tile MMA uses ldmatrix loads, which serve 16-bit m16n8k16 atoms")))
    cs = [Symbol(:c_,i) for i in 1:prod(R)]
    statements = [:( $(cs[i]) = _atom_accumulator(accum,Val($i)) ) for i in eachindex(cs)]
    for k in 0:16:K-16
        av = [Symbol(:a_,i) for i in 1:R[1]]
        bv = [Symbol(:b_,j) for j in 1:R[2]]
        for i in 1:R[1]
            push!(statements,:($(av[i]) = load_a(p.atom,
                window(a,(wm*oftype(tid,$(16R[1]))+oftype(tid,$(16(i-1))),oftype(tid,$k)),Val((16,16))),lane)))
        end
        for j in 1:R[2]
            push!(statements,:($(bv[j]) = load_b(p.atom,
                window(b,(oftype(tid,$k),wn*oftype(tid,$(8R[2]))+oftype(tid,$(8(j-1)))),Val((16,8))),lane)))
        end
        for j in 1:R[2],i in 1:R[1]
            c = cs[i+(j-1)*R[1]]
            push!(statements,:($c = mma(p.atom,$(av[i]),$(bv[j]),$c)))
        end
    end
    values = [:($c.data[$k]) for c in cs for k in 1:4]
    m,n = 16W[1]*R[1],8W[2]*R[2]
    quote
        Base.@inline
        size(a) == ($m,$K) && size(b) == ($K,$n) || throw(DimensionMismatch("MMA plan/view shapes differ"))
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

@generated function store!(p::TiledMMA{A,W,R,K},dst::GlobalTile{Float32},
        accum::TiledAccumulator{TiledMMA{A,W,R,K}},tid::Integer) where {A,W,R,K}
    stores = [:(store!(window(dst,
        (wm*oftype(tid,$(16R[1]))+oftype(tid,$(16(i-1))),
         wn*oftype(tid,$(8R[2]))+oftype(tid,$(8(j-1)))),Val((16,8))),
        _atom_accumulator(accum,Val($(i+(j-1)*R[1]))),lane)) for j in 1:R[2] for i in 1:R[1]]
    quote
        Base.@inline
        size(dst) == ($(16W[1]*R[1]),$(8W[2]*R[2])) || throw(DimensionMismatch("output plan/view shapes differ"))
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
