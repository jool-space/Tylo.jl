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

# Packing and register counts come from the atom role, never tile-area/32.
_word_type(::Type{T},::Union{OperandA,OperandB}) where T = UInt32
_word_type(::Type{Float32},::Accumulator) = Float32
_word_count(::OperandA) = 4
_word_count(::OperandB) = 2
_word_count(::Accumulator) = 4
struct MMAFragment{T,Role<:OperandRole,N,R}
    data::NTuple{N,R}
    function MMAFragment(::Type{T},role::Role,data::NTuple{N,R}) where {T,Role<:OperandRole,N,R}
        if role isa Accumulator
            T === Float32 || throw(ArgumentError("FP32 accumulator required"))
        else
            MMA16x8x16(T)
        end
        R === _word_type(T,role) && N == _word_count(role) ||
            throw(ArgumentError("register payload does not match MMA role"))
        new{T,Role,N,R}(data)
    end
end
Layouts.layout(::MMAFragment{T,Role}) where {T,Role} =
    operand_layout(MMA16x8x16(T === Float32 ? BFloat16 : T),Role())
zero_accumulator(::MMA16x8x16) = MMAFragment(Float32,Accumulator(),(0f0,0f0,0f0,0f0))
@inline Base.map(f,a::MMAFragment{Float32,Accumulator}) =
    MMAFragment(Float32,Accumulator(),map(f,a.data))
@inline scale(a::MMAFragment{Float32,Accumulator},x) = map(Base.Fix2(*,x),a)

"Load a 16×16 A operand from shared memory; K contiguous in 16-byte groups."
function load_a end
"Load a 16×8 B operand from shared memory; K contiguous in 16-byte groups."
function load_b end
"Collectively multiply operands and return the updated immutable accumulator."
function mma end
"Store an accumulator to a logical output view, using its lane/value ownership."
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
struct MMAAccumulator{P,N}
    data::NTuple{N,MMAFragment{Float32,Accumulator,4,Float32}}
    function MMAAccumulator(p::P,data::NTuple{N,MMAFragment{Float32,Accumulator,4,Float32}}) where {P<:TiledMMA,N}
        N == _acc_count(p) || throw(ArgumentError("accumulator shape does not match plan"))
        new{P,N}(data)
    end
end
_acc_count(::TiledMMA{A,W,R}) where {A,W,R} = prod(R)
@inline zero_accumulator(p::TiledMMA) = MMAAccumulator(p,ntuple(_ -> zero_accumulator(p.atom),Val(_acc_count(p))))
@inline Base.map(f,a::MMAAccumulator{P,N}) where {P,N} =
    MMAAccumulator(_plan(P),map(x -> map(f,x),a.data))
@inline scale(a::MMAAccumulator,x) = map(Base.Fix2(*,x),a)
_plan(::Type{TiledMMA{MMA16x8x16{T},W,R,K}}) where {T,W,R,K} =
    TiledMMA(MMA16x8x16(T),Val(W),Val(R),Val(K))
