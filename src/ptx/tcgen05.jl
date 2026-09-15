# tcgen05.mma issue. Operand descriptors come from the tile's base and the
# origin's element offset through the tile's own (swizzled) layout: at
# origins aligned to the swizzle cycle the swizzle is the identity, and the
# hardware applies it to every access from the descriptor's base. K steps
# go through the same layout, which places stripes for K-major storage and
# rows for MN-major storage without separate arithmetic.
@generated function _operand_structure(::Type{L},::Type{T},::Type{Role}) where {L,T,Role}
    s = swizzled_structure(L,T,_k_axis(Role))
    s === nothing && return :(throw(ArgumentError("the shared operand is not a canonical swizzled tile; see swizzled_structure")))
    :($s)
end
Base.@propagate_inbounds function tcgen05_operand(a::A,role::Role,t::SharedTile{T,L},
        origin::Tuple{Int32,Int32}=(Int32(0),Int32(0))) where {A<:Tcgen05MMA,Role<:Union{OperandA,OperandB},T,L}
    T === eltype(a,role) || throw(ArgumentError("operand tile element type differs from the atom"))
    s = _operand_structure(L,T,Role)
    m,n,k = size(a)
    outer = Role === OperandA ? m : n
    kaxis = _k_axis(role)
    chunk = Int32(16 ÷ sizeof(T))
    @boundscheck begin
        PTX.smem_addr_u32(t.ptr) % UInt32(8s.swizzle_bytes) == 0 || throw(ArgumentError("tcgen05 operand storage must be aligned to eight swizzle rows"))
        origin[kaxis] >= 0 && origin[3-kaxis] >= 0 && origin[3-kaxis] % Int32(8) == 0 &&
            (s.major === :K ? origin[kaxis] % chunk == 0 : origin[kaxis] % Int32(8) == 0 && origin[3-kaxis] % Int32(s.row_elements) == 0) ||
            throw(ArgumentError("operand origin must start a swizzle chunk"))
        origin[3-kaxis] + outer <= size(t)[3-kaxis] || throw(BoundsError(t,origin))
        s.major === :K || origin[3-kaxis] + outer <= s.row_elements*s.groups || throw(BoundsError(t,origin))
    end
    address = PTX.smem_addr_u32(t.ptr)
    # Bounds checks on: PTX.jl's checked builder validates the address window
    # and alignment. Elided: the constant fields with the address masked in.
    base = _descriptor_fields(Val(s.leading_bytes),Val(s.stride_bytes),Val(s.swizzle_bytes)) | UInt64((address & UInt32(0x3FFF0)) >> 4)
    @boundscheck base = PTX.tcgen05_descriptor(address;leading_bytes=s.leading_bytes,
                                               stride_bytes=s.stride_bytes,swizzle=_tcgen05_swizzle(s.swizzle_bytes))
    layout = _unswizzled(t.layout)
    Tcgen05Operand{typeof(a),Role,s.major,T,typeof(layout)}(base,base + _units(layout,origin,T),layout,origin)
end
@inline _units(layout,c,::Type{T}) where T = ((layout(c) % UInt32) * UInt32(sizeof(T)) >> 4) % UInt64
# Descriptor offsets come from the unswizzled layout: at chunk-aligned
# origins the swizzle is the identity, and the hardware applies it to every
# access itself. Stepping through the plain affine map keeps each K step a
# constant or a linear expression of the origin.
@inline _unswizzled(l::Layouts.Composition{<:Layouts.Swizzle}) = l.inner
@inline _unswizzled(l::Layouts.Window{S}) where S =
    Layouts.Window{S,typeof(_unswizzled(l.parent)),typeof(l.origin)}(_unswizzled(l.parent),l.origin)
# The descriptor's constant fields, packed by PTX.jl while generating; the
# address field is masked in at runtime without a checked call.
@generated _descriptor_fields(::Val{Leading},::Val{Stride},::Val{Bytes}) where {Leading,Stride,Bytes} =
    :($(PTX.tcgen05_descriptor(UInt32(0);leading_bytes=Leading,stride_bytes=Stride,swizzle=_tcgen05_swizzle(Bytes))))
# The extent of the first mode along an axis of the unswizzled layout: for
# MN-major operands, one swizzle row of elements.
@generated function _group_extent(::Type{L},::Val{Axis}) where {L,Axis}
    l = _static_instance(L <: Layouts.Window ? L.parameters[2] : L)
    :($(_axis_pairs(Layouts.shape(l)[Axis],strides(l)[Axis])[1][1]))
end
"""
    tcgen05_operand(operand::Tcgen05Operand, origin)

The same tile at another logical origin, without recomputing the descriptor.
"""
Base.@propagate_inbounds function tcgen05_operand(o::Tcgen05Operand{P,Role,Major,T,L},origin::Tuple{Int32,Int32}) where {P,Role,Major,T,L}
    kaxis = _k_axis(Role)
    chunk = Int32(16 ÷ sizeof(T))
    m,n,k = size(P())
    outer = Role === OperandA ? m : n
    @boundscheck begin
        origin[kaxis] >= 0 && origin[3-kaxis] >= 0 && origin[3-kaxis] % Int32(8) == 0 &&
            (Major === :K ? origin[kaxis] % chunk == 0 : origin[kaxis] % Int32(8) == 0 && origin[3-kaxis] % Int32(_group_extent(L,Val(3-kaxis))) == 0) ||
            throw(ArgumentError("operand origin must start a swizzle chunk"))
        origin[3-kaxis] + outer <= size(o.layout)[3-kaxis] || throw(BoundsError(o,origin))
    end
    Tcgen05Operand{P,Role,Major,T,L}(o.base,o.base + _units(o.layout,origin,T),o.layout,origin)
end
# A K step: through the layout in general, so stripes and row groups land
# where the layout puts them; by a constant when the K axis is one flat
# mode, so a runtime origin costs one add per instruction.
@inline function _step(o::Tcgen05Operand{P,Role,Major,T},::Val{DK}) where {P,Role,Major,T,DK}
    c = _k_axis(Role) == 1 ? (o.origin[1]+Int32(DK),o.origin[2]) : (o.origin[1],o.origin[2]+Int32(DK))
    o.base + _units(o.layout,c,T)
end
_flat_k_stride(::Type,kaxis) = nothing
function _flat_k_stride(::Type{Layouts.Layout{S,D}},kaxis) where {S,D}
    S.parameters[kaxis] <: Layouts.StaticInt && D.parameters[kaxis] <: Layouts.StaticInt || return nothing
    D.parameters[kaxis].parameters[1]
end
_flat_k_stride(::Type{Layouts.Window{S,L,O}},kaxis) where {S,L,O} = _flat_k_stride(L,kaxis)
function _step_expr(name,::Type{O},::Type{T},DK) where {O<:Tcgen05Operand,T}
    stride = _flat_k_stride(O.parameters[5],_k_axis(O.parameters[2]))
    stride === nothing ? :(_step($name,Val($DK))) : :($name.at + $(UInt64(DK*stride*sizeof(T) ÷ 16)))
end
_step_expr(name,::Type{<:TmemTile},::Type,DK) = :(_step($name,Val($DK)))
@inline _k_extent(o::Tcgen05Operand{P,Role}) where {P,Role} = Int(size(o.layout)[_k_axis(Role)]) - Int(o.origin[_k_axis(Role)])
@inline _tmem_offset(::Layouts.Layout,::Type) = UInt32(0)
@inline _tmem_offset(w::Layouts.Window,::Type{T}) where T =
    ((w.origin[1] % UInt32) << UInt32(16)) + (w.origin[2] % UInt32) ÷ UInt32(_tmem_packing(T))
@inline _tmem_start(t::TmemTile{T}) where T = t.address + _tmem_offset(t.layout,T)
@inline _step(t::TmemTile{T},::Val{DK}) where {T,DK} = _tmem_start(t) + UInt32(DK ÷ _tmem_packing(T))
@inline _k_extent(t::TmemTile) = Int(size(t)[2])

"""
    mma(atom::Tcgen05MMA, d::TmemTile, a, b, Val(K), accumulate::Bool)

Issue `K ÷ k` instructions accumulating `a*b` over logical K into the TMEM
accumulator `d`, from one thread. `a` is a [`Tcgen05Operand`](@ref) or a
`TmemTile` of the atom's A type in the accumulator convention; `b` is a
[`Tcgen05Operand`](@ref). With `accumulate=false` the first instruction
overwrites `d`. Nothing waits: order TMEM and shared-memory readiness with
fences and barriers, and observe completion through [`commit_mma`](@ref).
"""
@generated function mma(atom::A,d::TmemTile{TC,LD},a::OA,b::OB,::Val{K},accumulate::Bool) where {A<:Tcgen05MMA,TC,LD,OA,OB,K}
    inst = A()
    m,n,katom = size(inst)
    TC === eltype(inst,Accumulator()) || return :(throw(ArgumentError("accumulator element type differs from the atom")))
    _tmem_canonical(LD,m) || return :(throw(ArgumentError("accumulator layout must be operand_layout(atom, Accumulator()) or a window of it")))
    K isa Int && K > 0 && K % katom == 0 || return :(throw(ArgumentError("K must be a positive multiple of the atom's K")))
    if OA <: Tcgen05Operand
        OA.parameters[1] === A && OA.parameters[2] === OperandA ||
            return :(throw(ArgumentError("A operand belongs to a different atom or role")))
        amajor = OA.parameters[3]
    elseif OA <: TmemTile
        OA.parameters[1] === eltype(inst,OperandA()) || return :(throw(ArgumentError("TMEM A element type differs from the atom")))
        _tmem_canonical(OA.parameters[2],m) || return :(throw(ArgumentError("TMEM A layout must follow the accumulator convention")))
        amajor = :K
    else
        return :(throw(ArgumentError("A is a Tcgen05Operand or a TmemTile")))
    end
    OB <: Tcgen05Operand && OB.parameters[1] === A && OB.parameters[2] === OperandB ||
        return :(throw(ArgumentError("B operand belongs to a different atom or role")))
    bmajor = OB.parameters[3]
    idesc = instruction_descriptor(inst;a_major=amajor,b_major=bmajor)
    instruction = Meta.parse("ptx\"tcgen05.mma.cta_group::1.kind::$(_tcgen05_kind(eltype(inst,OperandA())))\"")
    TA = eltype(inst,OperandA())
    issues = [:($instruction(dstart,$(_step_expr(:a,OA,TA,kk*katom)),$(_step_expr(:b,OB,eltype(inst,OperandB()),kk*katom)),
                             $idesc,$(kk == 0 ? :accumulate : true)))
              for kk in 0:K÷katom-1]
    quote
        Base.@inline
        @boundscheck _k_extent(a) >= $K && _k_extent(b) >= $K || throw(BoundsError())
        dstart = _tmem_start(d)
        $(issues...)
        nothing
    end
end
@inline commit_mma(barrier::Core.LLVMPtr{UInt64,PTX.AS.Shared}) = commit_mma(PTX.smem_addr_u32(barrier))
@inline commit_mma(barrier::UInt32) =
    ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64"(barrier)

@inline function _check_tmem_columns(c)
    c isa Int && ispow2(c) && 32 <= c <= 512 || throw(ArgumentError("TMEM allocations are powers of two from 32 to 512 columns"))
end
@inline function tmem_allocate!(slot::Core.LLVMPtr{UInt32,PTX.AS.Shared},::Val{C}) where C
    _check_tmem_columns(C)
    ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(PTX.smem_addr_u32(slot),UInt32(C))
end
@inline function tmem_deallocate!(address::UInt32,::Val{C}) where C
    _check_tmem_columns(C)
    ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(address,UInt32(C))
end
@inline tmem_relinquish_permit() = ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
