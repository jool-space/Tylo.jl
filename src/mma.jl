abstract type OperandRole end
struct OperandA <: OperandRole end
struct OperandB <: OperandRole end
struct Accumulator <: OperandRole end

"""
    MMAAtom{(m,n,k),TA,TB,TC}()
    MMAAtom((m,n,k), TA, TC=Float32)

A warp-collective `mma.sync` instruction described by its shape and element
types. The atom contributes exactly one instruction-specific fact: the
thread/value ownership of its A, B and accumulator operands, from
[`operand_layout`](@ref). Fragments, loads, stores, reductions, tiling and
conversions derive from those ownerships. `Float32` inputs select the TF32
instruction of that shape. The PTX extension supplies the instruction for
each atom it implements; an atom without one is still a valid description.
"""
struct MMAAtom{S,TA,TB,TC}
    function MMAAtom{S,TA,TB,TC}() where {S,TA,TB,TC}
        _check_atom(S,TA,TB,TC)
        new{S,TA,TB,TC}()
    end
end
Base.@constprop :aggressive @inline MMAAtom(shape::NTuple{3,Int},::Type{TA},::Type{TC}=_default_accumulator(TA)) where {TA,TC} =
    MMAAtom{shape,TA,TA,TC}()
Base.@constprop :aggressive @inline MMAAtom(shape::NTuple{3,Int},::Type{TA},::Type{TB},::Type{TC}) where {TA,TB,TC} =
    MMAAtom{shape,TA,TB,TC}()
_default_accumulator(::Type{T}) where T = T <: Integer ? Int32 : Float32
Base.size(::MMAAtom{S}) where S = S
threads(::MMAAtom) = 32
Base.eltype(::MMAAtom{S,TA,TB,TC},::OperandA) where {S,TA,TB,TC} = TA
Base.eltype(::MMAAtom{S,TA,TB,TC},::OperandB) where {S,TA,TB,TC} = TB
Base.eltype(::MMAAtom{S,TA,TB,TC},::Accumulator) where {S,TA,TB,TC} = TC

_warp_shapes(::Val{8}) = ((16,8,16),(16,8,32))
_warp_shapes(::Val{16}) = ((16,8,8),(16,8,16))
_warp_shapes(::Val{32}) = ((16,8,4),(16,8,8))
_warp_shapes(::Val) = ()
_input_family(::Type{T}) where T<:Union{Float8E4M3,Float8E5M2} = :fp8
_input_family(::Type{T}) where T<:Union{Int8,UInt8} = :int8
_input_family(::Type{T}) where T<:Union{BFloat16,Float16} = :fp16
_input_family(::Type{Float32}) = :tf32
_input_family(::Type) = :none
# Pure and inlinable: the constructor may run inside a kernel.
@inline function _check_atom(S,TA,TB,TC)
    S isa NTuple{3,Int} || throw(ArgumentError("MMA shape must be an (m,n,k) tuple"))
    family = _input_family(TA)
    family !== :none && family === _input_family(TB) ||
        throw(ArgumentError("A and B must share an input family"))
    family === :fp16 && TA !== TB && throw(ArgumentError("16-bit A and B must share a type"))
    S in _warp_shapes(Val(_element_bits(TA))) ||
        throw(ArgumentError("unsupported warp MMA shape for these inputs"))
    TC === (family === :int8 ? Int32 : Float32) || (TC === Float16 && TA === Float16) ||
        throw(ArgumentError("unsupported accumulator type for these inputs"))
    nothing
end

# The ownerships follow the PTX ISA figures for mma.sync m16n8kK. Each
# thread's A word covers `per_word` consecutive K columns; lane group t%4
# selects the column block and t÷4 the row. Logical indices are column-major
# within the operand's own shape. Modes of extent one are omitted.
_modes(pairs) = (Tuple(Layouts.static(n) for (n,_) in pairs if n > 1),
                 Tuple(Layouts.static(d) for (n,d) in pairs if n > 1))
function _ownership(shape,lanes,values)
    ls, ld = _modes(lanes)
    vs, vd = _modes(values)
    if isempty(vs)
        vs, vd = (Layouts.static(1),), (Layouts.static(0),)
    end
    Layouts.Ownership(Val(shape),Layouts.Layout((ls,vs),(ld,vd)))
end
function _operand_layout(a::MMAAtom{S,TA},::OperandA) where {S,TA}
    m,n,k = S
    per_word = 32 ÷ _element_bits(TA)
    block = 4per_word
    _ownership((m,k),((4,m*per_word),(8,1)),((per_word,m),(2,8),(k÷block,m*block)))
end
function _operand_layout(a::MMAAtom{S,TA},::OperandB) where {S,TA}
    m,n,k = S
    per_word = 32 ÷ _element_bits(TA)
    block = 4per_word
    _ownership((k,n),((4,per_word),(8,k)),((per_word,1),(k÷block,block)))
end
function _operand_layout(a::MMAAtom{S},::Accumulator) where S
    m,n,k = S
    _ownership((m,n),((4,2m),(8,1)),((2,m),(2,8)))
end
"""
    operand_layout(atom, role)

The thread/value ownership of an atom's A, B or accumulator operand: an
explicit `Layouts.Ownership` following the PTX ISA figures. Evaluated while
generating, so it is a constant inside kernels.
"""
@generated operand_layout(a::A,::R) where {A<:MMAAtom,R<:OperandRole} = :($(_operand_layout(A(),R())))
_words(atom::MMAAtom,role::OperandRole) =
    _register_count(operand_layout(atom,role)) * _element_bits(eltype(atom,role)) ÷ 32

"""
    load_a(atom, tile, lane)
    load_b(atom, tile, lane)

Load an atom's A (m×k) or B (k×n) operand from a shared tile through
[`load_fragment`](@ref) with the operand's ownership.
"""
function load_a end
@doc (@doc load_a) load_b
function load_b end
"Collectively multiply operands and return the updated immutable accumulator."
function mma end
"Atoms with an instruction binding."
instruction_atoms() = copy(_INSTRUCTION_ATOMS)
const _INSTRUCTION_ATOMS = MMAAtom[]
"""
    load_fragment(ownership, tile, thread)

Load this thread's values of a memory tile at the coordinates the ownership
assigns to it. Scalar loads are correct for any static ownership. A shared
tile of 8- or 16-bit elements whose ownership decomposes into
[`CopyAtom`](@ref) blocks along the layout's static unit-stride axis loads
through `ldmatrix` instead, with the same result. Packed element types
return a `PackedFragment`.
"""
function load_fragment end
"""
    store!(plan, destination, accumulator, thread)
    store!(destination, fragment, thread)
    store!(pointer, packed::PackedFragment)

Store an accumulator to a logical output view using its ownership. The
tile form stores any fragment at its ownership's coordinates: scalar stores
in general, `stmatrix` for a packed fragment whose ownership decomposes
into [`CopyAtom`](@ref) blocks over a shared tile when compiling for
sm_90 or later. The packed payload overload writes this thread's local
words contiguously to a global typed BF16/FP16 pointer or a UInt16 bit
pointer. It writes complete words, using 4-, 8- or 16-byte alignment for
one, two/three or at least four words, and a sufficiently large distinct
destination region for each thread.
"""
function store! end

@inline function zero_accumulator(a::MMAAtom{S,TA,TB,TC}) where {S,TA,TB,TC}
    values = Fragment(ntuple(_ -> zero(TC),Val(4)),operand_layout(a,Accumulator()))
    _element_bits(TC) == 32 ? values : pack(values)
end

"""
    TiledMMA(atom, Val((warps_m,warps_n)), Val((repeat_m,repeat_n)), Val(k))

A CTA's MMA decomposition. Each warp computes repeat_m×repeat_n atoms; K is
a positive multiple of the atom's K. Warp numbering is M-fastest. This plan
describes ownership and instruction repetition, with no allocation or hidden
barriers. Its accumulator is one flat `Fragment` in `TiledMMAOwnership`.
"""
struct TiledMMA{A,W,R,K}
    atom::A
    function TiledMMA(a::MMAAtom,::Val{W},::Val{R},::Val{K}) where {W,R,K}
        W isa Tuple && R isa Tuple && length(W) == length(R) == 2 &&
            all(n -> n isa Int && n > 0,(W...,R...)) && prod(W) <= 32 &&
            K isa Int && K > 0 && K % size(a)[3] == 0 || throw(ArgumentError("invalid MMA tiling"))
        new{typeof(a),W,R,K}(a)
    end
end
Base.size(p::TiledMMA{A,W,R,K}) where {A,W,R,K} = (size(p.atom)[1]*W[1]*R[1],size(p.atom)[2]*W[2]*R[2],K)
threads(::TiledMMA{A,W}) where {A,W} = 32prod(W)
_acc_count(::TiledMMA{A,W,R}) where {A,W,R} = prod(R)
@generated _plan(::Type{TiledMMA{A,W,R,K}}) where {A,W,R,K} = :($(TiledMMA(A(),Val(W),Val(R),Val(K))))

# The tiled accumulator's flattened thread/value mapping. It is useful for
# masks and epilogues as well as stores; physical memory layouts are separate.
struct TiledMMAOwnership{P} end
operand_layout(p::TiledMMA,::Accumulator) = TiledMMAOwnership{typeof(p)}()
Base.size(::TiledMMAOwnership{P}) where P = size(_plan(P))[1:2]
@inline function Layouts.coordinate(::TiledMMAOwnership{TiledMMA{A,W,R,K}},
                                    tid::Integer,::Val{E}) where {A,W,R,K,E}
    plan = _plan(TiledMMA{A,W,R,K})
    m,n = size(plan.atom)[1],size(plan.atom)[2]
    0 <= E < 4prod(R) || throw(BoundsError())
    atom,word=E÷4,E%4
    rm,rn=atom%R[1],atom÷R[1]
    warp,lane=tid÷oftype(tid,32),tid%oftype(tid,32)
    wm,wn=warp%oftype(tid,W[1]),warp÷oftype(tid,W[1])
    r,c=Layouts.coordinate(operand_layout(plan.atom,Accumulator()),lane,Val(word))
    (r+wm*oftype(tid,m*R[1])+oftype(tid,m*rm),c+wn*oftype(tid,n*R[2])+oftype(tid,n*rn))
end
_register_count(::TiledMMAOwnership{TiledMMA{A,W,R,K}}) where {A,W,R,K} = 4prod(R)
@inline function zero_accumulator(p::TiledMMA{A,W,R,K}) where {A,W,R,K}
    Fragment(ntuple(_ -> zero(eltype(p.atom,Accumulator())),Val(4prod(R))),TiledMMAOwnership{typeof(p)}())
end
const TiledAccumulator{P,T,N} = Fragment{T,N,TiledMMAOwnership{P}}

# One atom's accumulator, as a fragment in the atom's own ownership.
@inline function _atom_accumulator(acc::TiledAccumulator{P},::Val{I}) where {P,I}
    Fragment(@rtuple(j -> acc.data[4*(I-1)+j], 1:4),operand_layout(_plan(P).atom,Accumulator()))
end

"""
    pack_operand_a(atom, left, right)
    pack_operand_a(atom, accumulator, Val(m), Val(k))

Conversion from two horizontally adjacent FP32 16×8 accumulator fragments to
one 16×16 A operand, without lane communication. Round to the atom's
BF16/FP16 type, nearest even, and pack low element first. Signed zero and
infinities survive; NaNs remain NaNs but their payload/sign are unspecified.
No finite saturation or flush-to-zero modifier is applied. Finite overflow
follows the destination format. This is numerical conversion, not an FP32
bit reinterpretation.

The tiled overload selects zero-based M repetition `m` and 16-column pair `k`.
It requires one N warp and complete pairs of N atoms. Logical axis-1 ownership must match
that of the consuming MMA; the surrounding kernel owns this correspondence.
"""
@inline function pack_operand_a(atom::MMAAtom{S,TA},left::Fragment{Float32,4},
        right::Fragment{Float32,4}) where {S,TA}
    S == (16,8,16) && _element_bits(TA) == 16 || throw(ArgumentError("operand conversion targets the 16-bit m16n8k16 A operand"))
    pair = TiledMMAOwnership{TiledMMA{typeof(atom),(1,1),(1,2),16}}()
    pack(TA,relayout(operand_layout(atom,OperandA()),Fragment((left.data...,right.data...),pair)))
end
@inline function pack_operand_a(atom::MMAAtom,a::TiledAccumulator{TiledMMA{A,W,R,K}},
        ::Val{M},::Val{J}) where {A,W,R,K,M,J}
    W[2] == 1 && iseven(R[2]) || throw(ArgumentError("conversion requires one N warp and paired N atoms"))
    0 <= M < R[1] && 0 <= J < R[2]÷2 || throw(BoundsError())
    pack_operand_a(atom,_atom_accumulator(a,Val(M+1+2J*R[1])),_atom_accumulator(a,Val(M+1+(2J+1)*R[1])))
end
