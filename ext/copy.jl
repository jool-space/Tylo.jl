@inline function _check_vector(l,c,axis,::Val{V},::Type{T}) where {V,T}
    offset = l(c)
    offset*sizeof(T) % 16 == 0 || throw(ArgumentError("unaligned 16-byte vector"))
    for i in 1:V-1
        q = ntuple(j -> c[j] + (j == axis ? oftype(c[j],i) : zero(c[j])),2)
        l(q) == offset+i || throw(ArgumentError("noncontiguous 16-byte vector"))
    end
    nothing
end

@generated function Tylo.copy_async!(p::CopyPlan{S,Threads,Axis},
        dst::SharedTile{T},src::GlobalTile{T},tid::Integer) where {S,Threads,Axis,T}
    v,n = Tylo._copy_vectors(CopyPlan{S,Threads,Axis}(),T)
    copies = Expr[]
    for pass in 0:cld(n,Threads)-1
        push!(copies,quote
            index = tid + oftype(tid,$(pass*Threads))
            if index < oftype(tid,$n)
                c = Tylo._copy_coordinate(p,index,Val($v))
                @boundscheck begin
                    _check_vector(dst.layout,c,$Axis,Val($v),T)
                    _check_vector(src.layout,c,$Axis,Val($v),T)
                end
                ptx"cp.async.cg.shared.global"(pointer(dst,c),pointer(src,c),Val(16))
            end
        end)
    end
    quote
        Base.@inline
        size(dst) == size(src) == $S || throw(DimensionMismatch("copy plan/view shapes differ"))
        @boundscheck 0 <= tid < $Threads || throw(BoundsError())
        $(copies...)
        nothing
    end
end
@inline Tylo.commit_copies() = ptx"cp.async.commit_group"()
@inline Tylo.wait_copies(::Val{N}) where N = ptx"cp.async.wait_group"(Val(N))
