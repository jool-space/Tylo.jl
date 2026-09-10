"""
    WarpRowFragment(values::NTuple{N,Float32})

One row striped over a full warp: lane t owns columns t + 32e, e = 0:N-1.
Unlike `RowFragment`, a reduction communicates between lanes. Supply identity
values for invalid columns; all 32 lanes must execute collective operations.
"""
struct WarpRowFragment{N}
    data::NTuple{N,Float32}
    function WarpRowFragment(data::NTuple{N,Float32}) where N
        N > 0 || throw(ArgumentError("a row fragment must contain values"))
        new{N}(data)
    end
end
struct WarpRowLayout{N} end
Base.size(::WarpRowLayout{N}) where N = (1,32N)
Layouts.layout(::WarpRowFragment{N}) where N = WarpRowLayout{N}()
@inline function Layouts.coordinate(::WarpRowLayout{N},lane::Integer,::Val{E}) where {N,E}
    0 <= E < N || throw(BoundsError())
    (zero(lane),lane+oftype(lane,32E))
end
@inline Base.map(f,a::WarpRowFragment) = WarpRowFragment(map(f,a.data))

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

"""
    RowValues(ownership, values::NTuple{N,Float32})

Row results in the distribution described by `row_ownership(fragment)`.
`row_coordinate(results, thread, Val(e))` identifies each result's logical row.
Use `row_map(op, fragment, results)` to apply results without redistributing
registers. `only(results)` extracts a scalar when each thread holds one result.
These types describe ownership; they do not prove collective participation.
"""
struct RowValues{O<:RowOwnership,N}
    data::NTuple{N,Float32}
    function RowValues(o::O,data::NTuple{N,Float32}) where {O<:RowOwnership,N}
        N == _row_count(o) || throw(DimensionMismatch("row result count differs from ownership"))
        new{O,N}(data)
    end
end
"Describe the logical rows held by each thread and the replication of row results."
row_ownership(::RowFragment) = LaneRowOwnership()
row_ownership(::WarpRowFragment) = WarpRowOwnership()
row_ownership(::MMAFragment{Float32,Accumulator}) = MMARowOwnership{1,1}()
function row_ownership(::TiledMMA{A,W,R}) where {A,W,R}
    W[2] == 1 || throw(ArgumentError("row reductions across multiple N warps require explicit shared communication"))
    MMARowOwnership{W[1],R[1]}()
end
row_ownership(::MMAAccumulator{P}) where P = row_ownership(_plan(P))
row_ownership(::RowValues{O}) where O = O()
@inline Base.only(x::RowValues{O,1}) where O = x.data[1]
@inline Base.map(f,x::RowValues) = RowValues(row_ownership(x),map(f,x.data))

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
@inline _row_reduce(op,f::RowFragment{Float32}) =
    RowValues(row_ownership(f),(_local_reduce(op,f.data),))

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
@inline row_map(op,f::RowFragment{Float32},r::RowValues{LaneRowOwnership}) =
    map(x -> op(x,only(r)),f)
@inline row_map(op,f::WarpRowFragment,r::RowValues{WarpRowOwnership}) =
    map(x -> op(x,only(r)),f)
@inline function row_map(op,f::MMAFragment{Float32,Accumulator},r::RowValues{MMARowOwnership{1,1}})
    MMAFragment(Float32,Accumulator(),ntuple(i -> op(f.data[i],r.data[(i-1)÷2+1]),Val(4)))
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
