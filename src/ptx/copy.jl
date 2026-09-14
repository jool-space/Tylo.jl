@inline function _check_vector(l,c,axis,::Val{V},::Type{T}) where {V,T}
    offset = l(c)
    offset*sizeof(T) % 16 == 0 || throw(ArgumentError("unaligned 16-byte vector"))
    for i in 1:V-1
        q = ntuple(j -> c[j] + (j == axis ? oftype(c[j],i) : zero(c[j])),2)
        l(q) == offset+i || throw(ArgumentError("noncontiguous 16-byte vector"))
    end
    nothing
end

@generated function copy_async!(p::CopyPlan{S,Threads,Axis},
        dst::SharedTile{T},src::GlobalTile{T},tid::Integer) where {S,Threads,Axis,T}
    v,n = _copy_vectors(CopyPlan{S,Threads,Axis}(),T)
    copies = Expr[]
    for pass in 0:cld(n,Threads)-1
        push!(copies,quote
            index = tid + oftype(tid,$(pass*Threads))
            if index < oftype(tid,$n)
                c = _copy_coordinate(p,index,Val($v))
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
@inline commit_copies() = ptx"cp.async.commit_group"()
@inline wait_copies(::Val{N}) where N = ptx"cp.async.wait_group"(Val(N))

@generated function _scalar_copy_vector!(dst::SharedTile{T},src::GlobalTile{T},c,q,
                                         ::Val{Axis},::Val{V}) where {T,Axis,V}
    stores=Expr[]
    for j in 0:V-1
        push!(stores,quote
            d=(c[1]+oftype(c[1],$(Axis==1 ? j : 0)),c[2]+oftype(c[2],$(Axis==2 ? j : 0)))
            s=(q[1]+$(Axis==1 ? j : 0),q[2]+$(Axis==2 ? j : 0))
            x=_valid_coordinate(src,s) ? unsafe_load(pointer(src,s)) : zero(T)
            unsafe_store!(pointer(dst,d),x)
        end)
    end
    quote
        Base.@inline
        $(stores...)
        nothing
    end
end
@generated function copy_async!(p::CopyPlan{S,Threads,Axis},dst::SharedTile{T},
        src::GlobalTile{T},origin::Tuple,tid::Integer) where {S,Threads,Axis,T}
    v,n=_copy_vectors(CopyPlan{S,Threads,Axis}(),T)
    copies=Expr[]
    for pass in 0:cld(n,Threads)-1
        push!(copies,quote
            index=tid+oftype(tid,$(pass*Threads))
            if index < oftype(tid,$n)
                c=_copy_coordinate(p,index,Val($v))
                @boundscheck _check_vector(dst.layout,c,$Axis,Val($v),T)
                # Widen before adding an origin or multiplying global strides.
                q=(Int(origin[1])+Int(c[1]),Int(origin[2])+Int(c[2]))
                full=_valid_coordinate(src,q) && q[$Axis] <= size(src)[$Axis]-$v
                copied=false
                if full && _contiguous_vector(src.layout,q,Val($Axis),Val($v))
                    source=pointer(src,q)
                    if reinterpret(UInt64,source)%UInt64(16)==UInt64(0)
                        ptx"cp.async.cg.shared.global"(pointer(dst,c),source,Val(16))
                        copied=true
                    end
                end
                if !copied
                    # A wholly invalid floating-point vector has all-zero
                    # bits. Four word stores avoid eight BF16 address/value
                    # predicates while preserving the scalar fallback for a
                    # vector that straddles a boundary or another dtype.
                    empty=q[$(3-Axis)]<0 || q[$(3-Axis)]>=size(src)[$(3-Axis)] ||
                          q[$Axis]>=size(src)[$Axis] || q[$Axis]+$v<=0
                    if $(T in (BFloat16,Float16,Float32)) && empty
                        target=reinterpret(Core.LLVMPtr{UInt32,PTX.AS.Shared},pointer(dst,c))
                        unsafe_store!(target,UInt32(0),1)
                        unsafe_store!(target,UInt32(0),2)
                        unsafe_store!(target,UInt32(0),3)
                        unsafe_store!(target,UInt32(0),4)
                    else
                        _scalar_copy_vector!(dst,src,c,q,Val($Axis),Val($v))
                    end
                end
            end
        end)
    end
    quote
        Base.@inline
        size(dst)==$S || throw(DimensionMismatch("copy destination does not match plan"))
        length(origin)==2 || throw(ArgumentError("copy origin must have two coordinates"))
        @boundscheck 0 <= tid < $Threads || throw(BoundsError())
        $(copies...)
        nothing
    end
end
