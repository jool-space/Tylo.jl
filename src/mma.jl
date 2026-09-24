"""
    Atom

One hardware instruction, described as data. Every atom answers `size`,
`threads` and `eltype(atom, role)`, and [`operand_layout`](@ref)`(atom, role)`
for each operand it holds in registers: the thread/value ownership the
instruction prescribes. Loads, stores, reductions, tilings and conversions
derive from those ownerships; the instruction bindings live in the PTX layer.
"""
abstract type Atom end

"An operand of an atom: `OperandA`, `OperandB`, `Accumulator`, `ScaleA`, `ScaleB`, `Registers` or `Addresses`."
abstract type OperandRole end
"The A (m×k) operand of a matrix multiply."
struct OperandA <: OperandRole end
"The B (k×n) operand of a matrix multiply."
struct OperandB <: OperandRole end
"The m×n accumulator of a matrix multiply."
struct Accumulator <: OperandRole end
threads(a::Atom) = 32

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
struct MMAAtom{S,TA,TB,TC} <: Atom
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
Base.eltype(::MMAAtom{S,TA,TB,TC},::OperandA) where {S,TA,TB,TC} = TA
Base.eltype(::MMAAtom{S,TA,TB,TC},::OperandB) where {S,TA,TB,TC} = TB
Base.eltype(::MMAAtom{S,TA,TB,TC},::Accumulator) where {S,TA,TB,TC} = TC

_warp_shapes(::Val{8}) = ((16,8,16),(16,8,32))
_warp_shapes(::Val{16}) = ((16,8,8),(16,8,16))
_warp_shapes(::Val{32}) = ((16,8,4),(16,8,8))
_warp_shapes(::Val) = ()
_input_family(::Type{T}) where T<:Union{Float8_E4M3FN,Float8_E5M2} = :fp8
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
    Layouts.Ownership(shape,Layouts.Layout((ls,vs),(ld,vd)))
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
@generated operand_layout(a::A,::R) where {A<:Atom,R<:OperandRole} = :($(_operand_layout(A(),R())))
_words(atom::MMAAtom,role::OperandRole) =
    _register_count(operand_layout(atom,role)) * _element_bits(eltype(atom,role)) ÷ 32

"The row scales of a [`ScaledMMAAtom`](@ref)'s A operand: `m×4`, (row, K block)."
struct ScaleA <: OperandRole end
"The column scales of a [`ScaledMMAAtom`](@ref)'s B operand: `4×n`, (K block, column)."
struct ScaleB <: OperandRole end

"""
    ScaledMMAAtom{(m,n,k),TA,TB,TS}()
    ScaledMMAAtom((16,8,32), TA, TB=TA, TS=Float8_E8M0FNU)

A block-scaled warp `mma.sync`: `D = (A .* scale_A) * (B .* scale_B) + C` with
one scale per row of A and one per column of B. With K = 32 one instruction
multiplies one microscaling block, so a K loop over a matrix quantized in
blocks of 32 along K (MXFP8: `Float8_E4M3FN` or `Float8_E5M2` elements with
`Float8_E8M0FNU` power-of-two scales) accumulates the full product with no
dequantization. The accumulator is `Float32`.

A, B and the accumulator have the ownerships of the dense
`MMAAtom{(m,n,k),TA,TB,Float32}`. The atom adds two operands,
[`ScaleA`](@ref) and [`ScaleB`](@ref): each thread holds four scales, those
of its row (or column) for four consecutive K blocks, and the `block`
argument of [`mma`](@ref) selects one of the four. Their ownerships
were measured on a GB10: lanes `4q` and `4q+1` hold the scales of rows `q` and
`q+m÷2`, the instruction can equally read them from lanes `4q+2` and `4q+3`,
and every lane of quad `q` may supply the scale of column `q`. The layouts
therefore replicate: every lane loads the scales of a row and of a column,
and no lane is special.
"""
struct ScaledMMAAtom{S,TA,TB,TS} <: Atom
    function ScaledMMAAtom{S,TA,TB,TS}() where {S,TA,TB,TS}
        _check_scaled_atom(S,TA,TB,TS)
        new{S,TA,TB,TS}()
    end
end
Base.@constprop :aggressive @inline ScaledMMAAtom(shape::NTuple{3,Int},::Type{TA},::Type{TB}=TA,
        ::Type{TS}=Float8_E8M0FNU) where {TA,TB,TS} = ScaledMMAAtom{shape,TA,TB,TS}()
@inline function _check_scaled_atom(S,TA,TB,TS)
    S === (16,8,32) || throw(ArgumentError("block-scaled MMA multiplies one 32-element block: shape (16,8,32)"))
    _input_family(TA) === :fp8 && _input_family(TB) === :fp8 ||
        throw(ArgumentError("block-scaled MMA takes FP8 operands"))
    TS === Float8_E8M0FNU || throw(ArgumentError("block scales are Float8_E8M0FNU"))
    nothing
end
"The dense atom with the same A, B and accumulator operands."
dense_atom(::ScaledMMAAtom{S,TA,TB}) where {S,TA,TB} = MMAAtom{S,TA,TB,Float32}()
Base.size(::ScaledMMAAtom{S}) where S = S
Base.eltype(a::ScaledMMAAtom,role::Union{OperandA,OperandB,Accumulator}) = eltype(dense_atom(a),role)
Base.eltype(::ScaledMMAAtom{S,TA,TB,TS},::Union{ScaleA,ScaleB}) where {S,TA,TB,TS} = TS
"Scales held in one scale register: those of four consecutive K blocks."
const SCALE_BLOCKS = 4

_operand_layout(a::ScaledMMAAtom,role::Union{OperandA,OperandB,Accumulator}) = _operand_layout(dense_atom(a),role)
# (m, 4): scale of row r and block s at r + m*s. Lane bit 0 selects the upper
# half of the rows, lane bit 1 replicates, the quad is the row within a half.
function _operand_layout(a::ScaledMMAAtom{S},::ScaleA) where S
    m,n,k = S
    _ownership((m,SCALE_BLOCKS),((2,m÷2),(2,0),(8,1)),((SCALE_BLOCKS,m),))
end
# (4, n): scale of block s and column c at s + 4c. All four lanes of quad c
# replicate column c.
function _operand_layout(a::ScaledMMAAtom{S},::ScaleB) where S
    m,n,k = S
    _ownership((SCALE_BLOCKS,n),((4,0),(8,SCALE_BLOCKS)),((SCALE_BLOCKS,1),))
end
@inline zero_accumulator(a::ScaledMMAAtom) = zero_accumulator(dense_atom(a))

"""
    load_fragment(atom, role, tile, thread)

Load an atom operand from a memory tile of the operand's shape and element
type: [`load_fragment`](@ref) with `operand_layout(atom, role)`. For a
[`ScaledMMAAtom`](@ref) the scale operands are `m×4` (row, K block) for
`ScaleA` and `4×n` (K block, column) for `ScaleB`; a scale matrix over more
blocks is windowed four blocks at a time.
"""
Base.@propagate_inbounds function load_fragment(a::Atom,role::OperandRole,t::MemoryTile,thread::Integer)
    o = operand_layout(a,role)
    size(t) == size(o) || throw(DimensionMismatch("the operand tile must be $(size(o)[1])×$(size(o)[2])"))
    eltype(t) === eltype(a,role) || throw(ArgumentError("tile element type differs from the atom operand"))
    load_fragment(o,t,thread)
end
"""
    mma(atom, a, b, accumulator)
    mma(scaled_atom, a, b, accumulator, scale_a, scale_b, block)

Collectively multiply operands and return the updated immutable accumulator.
A [`ScaledMMAAtom`](@ref) also takes its two scale registers and the zero-based
`block` (0 to 3) that selects which of their four scales applies.
"""
function mma end
"Warp MMA atoms, dense and block-scaled, with an instruction binding."
instruction_atoms() = copy(_INSTRUCTION_ATOMS)
const _INSTRUCTION_ATOMS = Atom[]
"""
    load_fragment(ownership, tile, thread)

Load this thread's values of a memory tile at the coordinates the ownership
assigns to it. Scalar loads are correct for any static ownership. A shared
tile of 8- or 16-bit elements whose ownership decomposes into
[`CopyAtom`](@ref) blocks along the layout's static unit-stride axis loads
through `ldmatrix` instead, with the same result.
"""
function load_fragment end
"""
    store!(destination, fragment, thread)
    store!(plan::TiledMMA, destination, accumulator, thread)
    store!(plan::TiledMMA, destination, accumulator, origin, thread)

Store a fragment at its ownership's coordinates in a memory tile: scalar
stores in general, aligned vectors where the ownership and the tile's
declared alignment allow them, and `stmatrix` for 8- or 16-bit values whose
ownership decomposes into [`CopyAtom`](@ref) blocks over a shared tile when
compiling for sm_90 or later. The tiled forms store a tiled accumulator;
with an `origin` they select an edge tile of a bounded destination and
convert to its element type, skipping coordinates outside it.
"""
function store! end

"""
    zero_accumulator(atom_or_plan)

A zero accumulator fragment in the operand ownership of an MMA atom or a
[`TiledMMA`](@ref).
"""
@inline zero_accumulator(a::MMAAtom{S,TA,TB,TC}) where {S,TA,TB,TC} =
    Fragment(ntuple(_ -> zero(TC),Val(4)),operand_layout(a,Accumulator()))

"""
    TiledMMA(atom, (warps_m, warps_n), (repeat_m, repeat_n), k)

A CTA's MMA decomposition. Each warp computes `repeat_m×repeat_n` atoms; `k`
is a positive multiple of the atom's K. Warp numbering is M-fastest. All
arguments must be constants. This plan describes ownership and instruction
repetition, with no allocation or hidden barriers. Its accumulator is one
flat `Fragment` in `operand_layout(plan, Accumulator())`: the atom's
accumulator ownership embedded in the CTA tile, then repeated over warps and
atoms, M-fastest.
"""
struct TiledMMA{A,W,R,K}
    atom::A
    function TiledMMA{A,W,R,K}(a::A) where {A<:MMAAtom,W,R,K}
        W isa Tuple{Int,Int} && R isa Tuple{Int,Int} && all(>(0),(W...,R...)) && prod(W) <= 32 &&
            K isa Int && K > 0 && K % size(a)[3] == 0 || throw(ArgumentError("invalid MMA tiling"))
        new{A,W,R,K}(a)
    end
end
Base.@constprop :aggressive @inline TiledMMA(a::MMAAtom,warps::Tuple{Integer,Integer},repeats::Tuple{Integer,Integer},k::Integer) =
    TiledMMA{typeof(a),map(Int,warps),map(Int,repeats),Int(k)}(a)
Base.size(p::TiledMMA{A,W,R,K}) where {A,W,R,K} = (size(p.atom)[1]*W[1]*R[1],size(p.atom)[2]*W[2]*R[2],K)
threads(::TiledMMA{A,W}) where {A,W} = 32prod(W)
_plan(::Type{TiledMMA{A,W,R,K}}) where {A,W,R,K} = TiledMMA{A,W,R,K}(A())

function _operand_layout(p::TiledMMA{A,W,R,K},::Accumulator) where {A,W,R,K}
    atom = _operand_layout(p.atom,Accumulator())
    m, n = size(atom)
    M = m*W[1]*R[1]
    L, s = Layouts.Layout, Layouts.static
    embedded = Layouts.compose(L((s(m),s(n)),(s(1),s(M))),atom.mapping)
    threads = Layouts.concat(Layouts.mode(embedded,1),L((s(W[1]),s(W[2])),(s(m*R[1]),s(n*R[2]*M))))
    values = Layouts.concat(Layouts.mode(embedded,2),L((s(R[1]),s(R[2])),(s(m),s(n*M))))
    Layouts.Ownership(size(p)[1:2],Layouts.concat(threads,values))
end
@generated operand_layout(p::P,::Accumulator) where P<:TiledMMA = :($(_operand_layout(_plan(P),Accumulator())))
@inline zero_accumulator(p::TiledMMA{A,W,R,K}) where {A,W,R,K} =
    Fragment(ntuple(_ -> zero(eltype(p.atom,Accumulator())),Val(4prod(R))),operand_layout(p,Accumulator()))
# Whether a fragment type holds a plan's accumulator, decided while generating.
_is_accumulator(::Type{P},::Type{F}) where {P,F} = F <: Fragment &&
    same_distribution(_static_instance(F.parameters[3]),_operand_layout(_plan(P),Accumulator()))

# One atom's accumulator, as a fragment in the atom's own ownership.
@inline function _atom_accumulator(p::TiledMMA,acc::Fragment,::Val{I}) where I
    Fragment(@rtuple(j -> acc.data[4*(I-1)+j], 1:4),operand_layout(p.atom,Accumulator()))
end

"""
    pack_operand_a(atom, left, right)
    pack_operand_a(plan::TiledMMA, accumulator, m, k)

Conversion from two horizontally adjacent FP32 16×8 accumulator fragments to
one 16×16 A operand of `atom`, without lane communication. Round to the
atom's BF16/FP16 type, nearest even. Signed zero and infinities survive; NaNs
remain NaNs but their payload/sign are unspecified. No finite saturation or
flush-to-zero modifier is applied. This is numerical conversion, not an FP32
bit reinterpretation.

The tiled form converts the pair of 16-column blocks `k` in M repetition `m`
(both zero-based constants) of a `plan` accumulator into an A operand of the
plan's own atom. It requires one N warp and complete pairs of N atoms.
Logical axis-1 ownership must match that of the consuming MMA; the
surrounding kernel owns this correspondence.
"""
@inline function pack_operand_a(atom::MMAAtom{S,TA},left::Fragment{Float32,4},
        right::Fragment{Float32,4}) where {S,TA}
    S == (16,8,16) && _element_bits(TA) == 16 || throw(ArgumentError("operand conversion targets the 16-bit m16n8k16 A operand"))
    pair = operand_layout(TiledMMA(atom,(1,1),(1,2),16),Accumulator())
    Fragment{TA}(relayout(operand_layout(atom,OperandA()),Fragment((left.data...,right.data...),pair)))
end
Base.@constprop :aggressive @inline pack_operand_a(p::TiledMMA,a::Fragment,m::Integer,k::Integer) =
    _pack_operand_a(p,a,Val(Int(m)),Val(Int(k)))
@inline function _pack_operand_a(p::TiledMMA{A,W,R,K},a::Fragment,::Val{M},::Val{J}) where {A,W,R,K,M,J}
    W[2] == 1 && iseven(R[2]) || throw(ArgumentError("conversion requires one N warp and paired N atoms"))
    0 <= M < R[1] && 0 <= J < R[2]÷2 || throw(BoundsError())
    _same_distribution(Layouts.layout(a),operand_layout(p,Accumulator())) ||
        throw(ArgumentError("the fragment is not this plan's accumulator"))
    pack_operand_a(p.atom,_atom_accumulator(p,a,Val(M+1+2J*R[1])),_atom_accumulator(p,a,Val(M+1+(2J+1)*R[1])))
end
