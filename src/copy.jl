"""
    CopyPlan{Shape,Threads,Axis}()

Distribute a matrix's 16-byte vectors over `Threads` threads, with vectors
running along logical `Axis` (1 or 2). Shape is static; source and destination
may have different memory layouts. This plan issues cp.async copies only.
Call `commit_copies()` and `wait_copies(Val(groups_remaining))` explicitly,
then synchronize the consumers before reading shared memory.

Call `validate_copy(plan, T, destination_layout, source_layout)` on the host
to check vector contiguity/alignment and unique destination ownership. Raw
pointers must additionally be 16-byte aligned. Reusing a shared allocation
requires synchronization with its readers, separately from copy completion.
"""
struct CopyPlan{S,Threads,Axis}
    function CopyPlan{S,Threads,Axis}() where {S,Threads,Axis}
        S isa Tuple && length(S) == 2 && all(n -> n isa Int && n > 0,S) &&
            Threads isa Int && 0 < Threads <= 1024 && Threads % 32 == 0 &&
            Axis isa Int && Axis in (1,2) || throw(ArgumentError("invalid collective copy shape"))
        new{S,Threads,Axis}()
    end
end
Base.size(::CopyPlan{S}) where S = S
function _copy_vectors(::CopyPlan{S,Threads,Axis},::Type{T}) where {S,Threads,Axis,T}
    isbitstype(T) && sizeof(T) in (1,2,4,8,16) || throw(ArgumentError("unsupported copy element"))
    v = 16 ÷ sizeof(T)
    S[Axis] % v == 0 || throw(ArgumentError("copy axis must contain whole 16-byte vectors"))
    v,prod(S) ÷ v
end
@inline function _copy_coordinate(::CopyPlan{S,Threads,Axis},i,::Val{V}) where {S,Threads,Axis,V}
    n = oftype(i,S[Axis] ÷ V)
    inner,outer = (i % n) * oftype(i,V), i ÷ n
    Axis == 1 ? (inner,outer) : (outer,inner)
end

function validate_copy(p::CopyPlan{S,Threads,Axis},::Type{T},dst,src) where {S,Threads,Axis,T}
    size(dst) == size(src) == S || throw(DimensionMismatch("copy plan/view shapes differ"))
    v,n = _copy_vectors(p,T)
    written = Set{Int}()
    for i in 0:n-1
        c = _copy_coordinate(p,i,Val(v))
        for l in (dst,src)
            off = l(c)
            off*sizeof(T) % 16 == 0 || throw(ArgumentError("unaligned copy vector"))
            for j in 0:v-1
                q = ntuple(k -> c[k]+(k == Axis ? j : 0),2)
                l(q) == off+j || throw(ArgumentError("copy vector is not contiguous"))
            end
        end
        for j in 0:v-1
            off = Int(dst(c))+j
            off in written && throw(ArgumentError("copy destination aliases itself"))
            push!(written,off)
        end
    end
    nothing
end
function copy_async! end
"Close this thread's uncommitted cp.async operations as one group."
function commit_copies end
"Wait until at most N committed cp.async groups remain for this thread; no CTA barrier."
function wait_copies end

"""
    copy_async!(plan, destination, source, origin::Tuple, thread)

Copy a fixed-capacity tile from logical `origin` in a bounded global source.
Valid aligned contiguous 16-byte vectors use cp.async. Boundary or unaligned
vectors use scalar loads/stores, zero-filling every invalid element. Source
coordinates are checked before a pointer is formed. Destination shape and
vector alignment must satisfy the same contract as the unmasked form.

Both paths require explicit commit/wait and consumer synchronization. A copy
wait alone does not publish scalar shared stores to other threads. This form
uses scalar zero-fill because the pinned PTX wrapper exposes full-vector copies.
"""
copy_async!

@inline _valid_coordinate(t::MemoryTile,c::Tuple) =
    0 <= c[1] < size(t)[1] && 0 <= c[2] < size(t)[2]
@generated function _contiguous_vector(l,c,::Val{Axis},::Val{V}) where {Axis,V}
    conditions=[:(l((c[1]+$(Axis==1 ? j : 0),c[2]+$(Axis==2 ? j : 0))) == offset+$j) for j in 1:V-1]
    condition=isempty(conditions) ? true : foldl((a,b)->:($a && $b),conditions)
    quote
        Base.@inline
        offset=l(c)
        $condition
    end
end
