# Instruction bindings for the atoms this extension implements. Every other
# operation on the atom derives from its ownership layouts.
_mma_instruction(::Type{MMAAtom{(16,8,16),BFloat16,BFloat16,Float32}}) =
    ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
_mma_instruction(::Type{MMAAtom{(16,8,16),Float16,Float16,Float32}}) =
    ptx"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
_mma_instruction(::Type{A}) where A<:MMAAtom = nothing

@generated function _mma_words(f::PackedFragment{T,W}) where {T,W}
    :(f.data)
end
@generated function _mma_words(f::Fragment{Float32,N}) where N
    :(@rtuple(i -> reinterpret(UInt32,f.data[i]), 1:$N))
end

# Operand ownerships are checked while generating: a fragment from any load
# whose ownership enumerates to the atom's operand ownership is accepted.
@generated function Tylo.mma(atom::A,a::FA,b::FB,c::Fragment{TC,NC,LC}) where {A<:MMAAtom,FA,FB,TC,NC,LC}
    instruction = _mma_instruction(A)
    instruction === nothing && return :(throw(ArgumentError("no instruction binding for this MMA atom")))
    atom_instance = A()
    for (F,role) in ((FA,OperandA()),(FB,OperandB()),(Fragment{TC,NC,LC},Accumulator()))
        F <: Union{Fragment,PackedFragment} || return :(throw(ArgumentError("MMA operands are fragments")))
        eltype(F) === eltype(atom_instance,role) ||
            return :(throw(ArgumentError("MMA operand element type differs from the atom")))
        L = F.parameters[3]
        Tylo.same_distribution(Tylo._static_instance(L),operand_layout(atom_instance,role)) ||
            return :(throw(ArgumentError("MMA operand ownership differs from the atom")))
    end
    quote
        Base.@inline
        Fragment($instruction(_mma_words(a),_mma_words(b),c.data),operand_layout(atom,Accumulator()))
    end
end

Base.@propagate_inbounds function Tylo.load_a(a::MMAAtom{(16,8,16),T},t::SharedTile{T},lane::Integer) where T
    size(t) == (16,16) || throw(DimensionMismatch("A operand must be 16×16"))
    @boundscheck 0 <= lane < 32 || throw(BoundsError())
    c = (lane & oftype(lane,15),(lane >> 4)*oftype(lane,8))
    @boundscheck _check_vector(t.layout,c,2,Val(8),T)
    data = ptx"ldmatrix.sync.aligned.m8n8.x4.shared.b16"(pointer(t,c))
    PackedFragment(T,data,operand_layout(a,OperandA()))
end
Base.@propagate_inbounds function Tylo.load_b(a::MMAAtom{(16,8,16),T},t::SharedTile{T},lane::Integer) where T
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
@generated function Tylo.load_fragment(::L,tile::Tylo.MemoryTile{T},thread::Integer) where {L,T}
    o = Tylo._static_instance(L)
    o === nothing && return :(throw(ArgumentError("loads require a static ownership")))
    n = Tylo._register_count(o)
    loads = [:(unsafe_load(pointer(tile,Tylo.Layouts.coordinate(ownership,thread,Val($e))),1,Val($(sizeof(T))))) for e in 0:n-1]
    packed = T === BFloat16 || T === Float16
    quote
        Base.@inline
        ownership = $o
        values = Fragment(($(loads...),),ownership)
        $(packed ? :(pack(values)) : :(values))
    end
end
@generated function Tylo.store!(tile::Tylo.MemoryTile{T},f::Fragment{T,N,L},thread::Integer) where {T,N,L}
    o = Tylo._static_instance(L)
    o === nothing && return :(throw(ArgumentError("stores require a static ownership")))
    stores = [:(unsafe_store!(pointer(tile,Tylo.Layouts.coordinate(ownership,thread,Val($e))),f.data[$(e+1)],1,Val($(sizeof(T))))) for e in 0:N-1]
    quote
        Base.@inline
        ownership = Tylo.Layouts.layout(f)
        $(stores...)
        nothing
    end
end
@inline Tylo.store!(tile::Tylo.MemoryTile{T},f::PackedFragment{T},thread::Integer) where T =
    store!(tile,unpack(f),thread)

@generated function Tylo.mma(p::TiledMMA{A,W,R,K},
        a::SharedTile{T},b::SharedTile{T},
        accum::Tylo.TiledAccumulator{TiledMMA{A,W,R,K}},tid::Integer) where {A,W,R,K,T}
    cs = [Symbol(:c_,i) for i in 1:prod(R)]
    statements = [:( $(cs[i]) = Tylo._atom_accumulator(accum,Val($i)) ) for i in eachindex(cs)]
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
        Fragment(($(values...),),Tylo.Layouts.layout(accum))
    end
end

@generated function Tylo.store!(p::TiledMMA{A,W,R,K},dst::GlobalTile{Float32},
        accum::Tylo.TiledAccumulator{TiledMMA{A,W,R,K}},tid::Integer) where {A,W,R,K}
    stores = [:(store!(window(dst,
        (wm*oftype(tid,$(16R[1]))+oftype(tid,$(16(i-1))),
         wn*oftype(tid,$(8R[2]))+oftype(tid,$(8(j-1)))),Val((16,8))),
        Tylo._atom_accumulator(accum,Val($(i+(j-1)*R[1]))),lane)) for j in 1:R[2] for i in 1:R[1]]
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
@inline function Tylo.store!(p::TiledMMA{A,W,R,K},dst::GlobalTile{T},
        acc::Tylo.TiledAccumulator{TiledMMA{A,W,R,K}},origin::Tuple,tid::Integer) where {A,W,R,K,T}
    length(origin)==2 || throw(ArgumentError("store origin must have two coordinates"))
    @boundscheck 0 <= tid < 32prod(W) || throw(BoundsError())
    @rtuple(0:4prod(R)-1) do e
        c = Tylo.Layouts.coordinate(Tylo.Layouts.layout(acc),tid,Val(e))
        q = (Int(origin[1])+Int(c[1]),Int(origin[2])+Int(c[2]))
        if Tylo._valid_coordinate(dst,q)
            unsafe_store!(pointer(dst,q),T(acc.data[e+1]))
        end
    end
    nothing
end

@inline _pack_mma_pair(::Type{BFloat16},lo,hi) = PTX.bf16x2_pack(lo,hi)
@inline _pack_mma_pair(::Type{Float16},lo,hi) = ptx"cvt.rn.f16x2.f32"(hi,lo)
