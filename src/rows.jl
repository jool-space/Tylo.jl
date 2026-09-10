struct WarpRowLayout{N} end
Base.size(::WarpRowLayout{N}) where N = (1,32N)
_register_count(::WarpRowLayout{N}) where N = N
@inline function Layouts.coordinate(::WarpRowLayout{N},lane::Integer,::Val{E}) where {N,E}
    0 <= E < N || throw(BoundsError())
    (zero(lane),lane+oftype(lane,32E))
end

"""
    WarpRowFragment(values::NTuple{N,T})

A `Fragment` with one logical row striped over a full warp: lane t owns
columns t + 32e. The local values are scattered within that row. All 32 lanes
must participate in its collective reductions.
"""
const WarpRowFragment{N,T} = Fragment{T,N,WarpRowLayout{N}}
WarpRowFragment(data::NTuple{N,T}) where {N,T} = Fragment(data,WarpRowLayout{N}())

abstract type RowOwnership end
"One row per lane; no result replication between lanes."
struct LaneRowOwnership <: RowOwnership end
"One row per warp; its result is replicated over all 32 lanes."
struct WarpRowOwnership <: RowOwnership end
"Two rows per atom repetition and lane group; results replicated over four lanes."
struct MMARowOwnership{WarpsM,RepeatM} <: RowOwnership end
_row_count(::LaneRowOwnership) = 1
_row_count(::WarpRowOwnership) = 1
_row_count(::MMARowOwnership{W,R}) where {W,R} = 2R

# A reduction retains its two logical axes, with extent one on axis 2.
struct ReducedFragmentLayout{O<:RowOwnership} end
Base.size(::ReducedFragmentLayout{LaneRowOwnership}) = (32,1)
Base.size(::ReducedFragmentLayout{WarpRowOwnership}) = (1,1)
Base.size(::ReducedFragmentLayout{MMARowOwnership{W,R}}) where {W,R} = (16W*R,1)
_register_count(::ReducedFragmentLayout{O}) where O = _row_count(O())
@inline Layouts.coordinate(::ReducedFragmentLayout{O},t::Integer,e::Val) where O =
    (row_coordinate(O(),t,e),zero(t))

"""
    RowValues(ownership, values::NTuple{N,T})

A `Fragment` of reduction results with singleton axis 2 and explicit
replication between lanes. Use dotted arithmetic to broadcast it back onto
a compatible fragment. `only(results)` extracts this thread's scalar when
it holds one result; it does not claim that the whole tile has one element.
"""
const RowValues{O,N,T} = Fragment{T,N,ReducedFragmentLayout{O}}
RowValues(o::O,data::NTuple{N,T}) where {O<:RowOwnership,N,T} =
    Fragment(data,ReducedFragmentLayout{O}())
"Describe the logical rows held by each thread and the replication of row results."
row_ownership(::RowFragment) = LaneRowOwnership()
row_ownership(::WarpRowFragment) = WarpRowOwnership()
row_ownership(::MMAFragment{T,Accumulator}) where T = MMARowOwnership{1,1}()
function row_ownership(::TiledMMA{A,W,R}) where {A,W,R}
    W[2] == 1 || throw(ArgumentError("row reductions across multiple N warps require explicit shared communication"))
    MMARowOwnership{W[1],R[1]}()
end
row_ownership(::MMAAccumulator{P}) where P = row_ownership(_plan(P))
row_ownership(::RowValues{O}) where O = O()
@inline Base.only(x::RowValues{O,1}) where O = x.data[1]

"Identify result slot E's zero-based logical row at a zero-based thread index."
@inline function row_coordinate(x::RowValues,tid::Integer,::Val{E}) where E
    0 <= E < length(x.data) || throw(BoundsError())
    row_coordinate(row_ownership(x),tid,Val(E))
end
@inline row_coordinate(::LaneRowOwnership,t::Integer,::Val{0}) = t
@inline row_coordinate(::WarpRowOwnership,t::Integer,::Val{0}) = t÷oftype(t,32)
@inline function row_coordinate(::MMARowOwnership{W,R},t::Integer,::Val{E}) where {W,R,E}
    0 <= E < 2R || throw(BoundsError())
    (t÷oftype(t,32))*oftype(t,16R)+(t%oftype(t,32))÷oftype(t,4)+oftype(t,8(E%2)+16(E÷2))
end

# Balanced local tree with static tuple access. Device communication lives in
# the PTX extension, separate from these ordinary register operations.
@generated function _local_reduce(op::F,x::NTuple{N,Float32}) where {F,N}
    N > 0 || error("empty register reduction")
    function tree(first,last)
        first == last && return :(x[$first])
        mid = (first+last)÷2
        :(op($(tree(first,mid)),$(tree(mid+1,last))))
    end
    quote
        Base.@inline
        $(tree(1,N))
    end
end

"""
    row_sum(fragment)
    row_max(fragment)

FP32 reductions over logical columns. Lane-local rows require no communication;
warp-striped rows and MMA accumulators require all 32 lanes of each warp.
Results are replicated within each owning lane group. Tiled MMA currently
requires one warp along N: cross-warp reductions are explicitly caller-owned.
The operation's floating-point reduction order is not a sequential fold.
"""
@inline row_sum(f) = _row_reduce(+,f)
@inline row_max(f) = _row_reduce(max,f)

"Apply `op(value, row_result)` to every value, preserving register ownership."
@inline row_map(op,f::RowFragment,r::RowValues{LaneRowOwnership}) =
    map(x -> op(x,only(r)),f)
@inline row_map(op,f::WarpRowFragment,r::RowValues{WarpRowOwnership}) =
    map(x -> op(x,only(r)),f)
@inline function row_map(op,f::MMAFragment{T,Accumulator},r::RowValues{MMARowOwnership{1,1}}) where T
    data = ntuple(i -> op(f.data[i],r.data[(i-1)÷2+1]),Val(4))
    MMAFragment(eltype(data),Accumulator(),data)
end
@generated function row_map(op::F,a::MMAAccumulator{TiledMMA{A,W,R,K}},
                            r::RowValues{MMARowOwnership{WM,RM}}) where {F,A,W,R,K,WM,RM}
    W == (WM,1) && R[1] == RM || return :(throw(DimensionMismatch("row ownership differs from accumulator")))
    values = [:(row_map(op,a.data[$(i+(j-1)*R[1])],
        RowValues(MMARowOwnership{1,1}(),(r.data[$(2i-1)],r.data[$(2i)]))))
        for j in 1:R[2] for i in 1:R[1]]
    quote
        Base.@inline
        MMAAccumulator(_plan($(TiledMMA{A,W,R,K})),($(values...),))
    end
end
