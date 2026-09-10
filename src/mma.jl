abstract type OperandRole end
struct OperandA <: OperandRole end
struct OperandB <: OperandRole end
struct Accumulator <: OperandRole end

"A warp-collective 16×8×16 MMA atom with FP16/BF16 inputs and FP32 accumulation."
struct MMA16x8x16{T}
    function MMA16x8x16(::Type{T}) where T
        T === BFloat16 || T === Float16 || throw(ArgumentError("FP16 or BF16 required"))
        new{T}()
    end
end
Base.size(::MMA16x8x16) = (16,8,16)

# Thread/value distributions are ISA properties, independent of shared
# storage layout. B uses the mathematical (K,N) convention, not a Bᵀ API.
function operand_layout(::MMA16x8x16,::OperandA)
    Layouts.Ownership(Val((16,16)),Layouts.Layout(
        ((Layouts.static(4),Layouts.static(8)),(Layouts.static(2),Layouts.static(2),Layouts.static(2))),
        ((Layouts.static(32),Layouts.static(1)),(Layouts.static(16),Layouts.static(8),Layouts.static(128)))))
end
function operand_layout(::MMA16x8x16,::OperandB)
    Layouts.Ownership(Val((16,8)),Layouts.Layout(
        ((Layouts.static(4),Layouts.static(8)),(Layouts.static(2),Layouts.static(2))),
        ((Layouts.static(2),Layouts.static(16)),(Layouts.static(1),Layouts.static(8)))))
end
function operand_layout(::MMA16x8x16,::Accumulator)
    Layouts.Ownership(Val((16,8)),Layouts.Layout(
        ((Layouts.static(4),Layouts.static(8)),(Layouts.static(2),Layouts.static(2))),
        ((Layouts.static(32),Layouts.static(1)),(Layouts.static(16),Layouts.static(8)))))
end

# Operand packing and local value counts come from the atom role.
# Elementwise accumulator results may change scalar type (e.g. Bool masks);
# only the instruction's prescribed FP32 format can be passed back to MMA.
_word_type(::Type{T},::Union{OperandA,OperandB}) where T = UInt32
_word_type(::Type{T},::Accumulator) where T = T
_word_count(::OperandA) = 4
_word_count(::OperandB) = 2
_word_count(::Accumulator) = 4
struct MMAFragment{T,Role<:OperandRole,N,R}
    data::NTuple{N,R}
    function MMAFragment(::Type{T},role::Role,data::NTuple{N,R}) where {T,Role<:OperandRole,N,R}
        if role isa Accumulator
            isbitstype(T) || throw(ArgumentError("register values must be isbits"))
        else
            MMA16x8x16(T)
        end
        R === _word_type(T,role) && N == _word_count(role) ||
            throw(ArgumentError("register payload does not match MMA role"))
        new{T,Role,N,R}(data)
    end
end
Layouts.layout(::MMAFragment{T,Role}) where {T,Role} =
    operand_layout(MMA16x8x16(T),Role())
Layouts.layout(::MMAFragment{T,Accumulator}) where T =
    operand_layout(MMA16x8x16(BFloat16),Accumulator())
zero_accumulator(::MMA16x8x16) = MMAFragment(Float32,Accumulator(),(0f0,0f0,0f0,0f0))
@inline function Base.map(f::F,a::MMAFragment{T,Accumulator}) where {F,T}
    data = (f(a.data[1]),f(a.data[2]),f(a.data[3]),f(a.data[4]))
    MMAFragment(eltype(data),Accumulator(),data)
end
@inline scale(a::MMAFragment{Float32,Accumulator},x) = map(Base.Fix2(*,x),a)

"Load a 16×16 A operand from shared memory; K contiguous in 16-byte groups."
function load_a end
"Load a 16×8 B operand from shared memory; K contiguous in 16-byte groups."
function load_b end
"Collectively multiply operands and return the updated immutable accumulator."
function mma end
"""
    store!(plan, destination, accumulator, thread)
    store!(pointer, packed::PackedBF16)

Store an accumulator to a logical output view using its ownership. The packed
payload overload writes this thread's local words contiguously to a global
UInt16 pointer. It requires 16-byte alignment, a multiple of eight BF16 values,
and a sufficiently large distinct destination region for each thread.
"""
function store! end

"""
    TiledMMA(atom, Val((warps_m,warps_n)), Val((repeat_m,repeat_n)), Val(k))

A CTA's MMA decomposition. Each warp computes repeat_m×repeat_n atoms; K is
a positive multiple of 16. Warp numbering is M-fastest. This plan describes
ownership and instruction repetition, with no allocation or hidden barriers.
"""
struct TiledMMA{A,W,R,K}
    atom::A
    function TiledMMA(a::MMA16x8x16,::Val{W},::Val{R},::Val{K}) where {W,R,K}
        W isa Tuple && R isa Tuple && length(W) == length(R) == 2 &&
            all(n -> n isa Int && n > 0,(W...,R...)) && prod(W) <= 32 &&
            K isa Int && K > 0 && K % 16 == 0 || throw(ArgumentError("invalid MMA tiling"))
        new{typeof(a),W,R,K}(a)
    end
end
Base.size(::TiledMMA{A,W,R,K}) where {A,W,R,K} = (16W[1]*R[1],8W[2]*R[2],K)
threads(::TiledMMA{A,W}) where {A,W} = 32prod(W)
struct MMAAccumulator{P,N,T}
    data::NTuple{N,MMAFragment{T,Accumulator,4,T}}
    function MMAAccumulator(p::P,data::NTuple{N,MMAFragment{T,Accumulator,4,T}}) where {P<:TiledMMA,N,T}
        N == _acc_count(p) || throw(ArgumentError("accumulator shape does not match plan"))
        new{P,N,T}(data)
    end
end
_acc_count(::TiledMMA{A,W,R}) where {A,W,R} = prod(R)
@inline zero_accumulator(p::TiledMMA) = MMAAccumulator(p,ntuple(_ -> zero_accumulator(p.atom),Val(_acc_count(p))))
# Static scalar calls avoid outlining nested tuple-map callbacks into device
# functions, which would materialize the accumulator in local memory.
@generated function Base.map(f::F,a::MMAAccumulator{P,N}) where {F,P,N}
    fragments=[:(_mapped_mma_fragment(f,a.data[$i])) for i in 1:N]
    quote
        Base.@inline
        MMAAccumulator(_plan($P),($(fragments...),))
    end
end
@inline _mapped_mma_fragment(f::F,a::MMAFragment{T,Accumulator}) where {F,T} = map(f,a)
@inline scale(a::MMAAccumulator,x) = map(Base.Fix2(*,x),a)
_plan(::Type{TiledMMA{MMA16x8x16{T},W,R,K}}) where {T,W,R,K} =
    TiledMMA(MMA16x8x16(T),Val(W),Val(R),Val(K))

# The tiled accumulator's flattened thread/value mapping. It is useful for
# masks and epilogues as well as stores; physical memory layouts are separate.
struct TiledMMAOwnership{P} end
operand_layout(p::TiledMMA,::Accumulator) = TiledMMAOwnership{typeof(p)}()
Layouts.layout(::MMAAccumulator{P}) where P = TiledMMAOwnership{P}()
Base.size(::TiledMMAOwnership{P}) where P = size(_plan(P))[1:2]
@inline function Layouts.coordinate(::TiledMMAOwnership{TiledMMA{A,W,R,K}},
                                    tid::Integer,::Val{E}) where {A,W,R,K,E}
    0 <= E < 4prod(R) || throw(BoundsError())
    atom,word=E÷4,E%4
    rm,rn=atom%R[1],atom÷R[1]
    warp,lane=tid÷oftype(tid,32),tid%oftype(tid,32)
    wm,wn=warp%oftype(tid,W[1]),warp÷oftype(tid,W[1])
    r,c=Layouts.coordinate(operand_layout(_plan(TiledMMA{A,W,R,K}).atom,Accumulator()),lane,Val(word))
    (r+wm*oftype(tid,16R[1])+oftype(tid,16rm),c+wn*oftype(tid,8R[2])+oftype(tid,8rn))
end

"""
    pack_operand_a(atom, left, right)
    pack_operand_a(atom, accumulator, Val(m), Val(k))

GPU conversion from two horizontally adjacent FP32 16×8 C atoms to one 16×16
A operand, without lane communication. Round to the atom's BF16/FP16 type,
nearest even, and pack low element first. Signed zero and infinities survive;
NaNs remain NaNs but their payload/sign are unspecified. No finite saturation
or flush-to-zero modifier is applied. Finite overflow follows the destination
format. This is numerical conversion, not an FP32 bit reinterpretation.

The tiled overload selects zero-based M repetition `m` and 16-column pair `k`.
It requires one N warp and complete pairs of N atoms. Row ownership must match
that of the consuming MMA; the surrounding kernel owns this correspondence.
"""
function pack_operand_a end
@inline function pack_operand_a(atom::MMA16x8x16,
        a::MMAAccumulator{TiledMMA{A,W,R,K}},::Val{M},::Val{J}) where {A,W,R,K,M,J}
    W[2] == 1 && iseven(R[2]) || throw(ArgumentError("conversion requires one N warp and paired N atoms"))
    0 <= M < R[1] && 0 <= J < R[2]÷2 || throw(BoundsError())
    pack_operand_a(atom,a.data[M+1+2J*R[1]],a.data[M+1+(2J+1)*R[1]])
end
