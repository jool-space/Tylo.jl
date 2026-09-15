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

# Matrix copies derived from a `MatrixCopyPlan`. Each instruction takes up
# to four consecutive blocks; lane t addresses row t%8 of block (t÷8) within
# the instruction, so every lane forms a valid address even for .x1/.x2.
@inline function _matrix_address(::Val{First},::Val{N},::Val{Rows},::Val{Axis},::Val{Width},
                                 lane::Integer) where {First,N,Rows,Axis,Width}
    block = oftype(lane,First) + ((lane >> 3) & oftype(lane,N-1))
    br = Rows == 1 ? zero(lane) : block % oftype(lane,Rows)
    bc = Rows == 1 ? block : block ÷ oftype(lane,Rows)
    row = oftype(lane,8)*br + (lane & oftype(lane,7))
    col = oftype(lane,Width)*bc
    Axis == 2 ? (row,col) : (col,row)
end
function _matrix_copy_expr(plan::MatrixCopyPlan,::Type{T},op::Symbol) where T
    groups = _matrix_groups(length(plan.words))
    width = 8plan.per_unit
    statements = Expr[]
    results = Dict{Int,Any}()
    for (g,group) in enumerate(groups)
        n = length(group)
        name = "$(op === :load ? "ldmatrix" : "stmatrix").sync.aligned.m8n8.x$(n)$(plan.trans ? ".trans" : "").shared.b16"
        instruction = Meta.parse("ptx\"$name\"")
        c = Symbol(:c_,g)
        push!(statements,:($c = _matrix_address(Val($(first(group)-1)),Val($n),Val($(plan.grid[1])),
                                                Val($(plan.axis)),Val($width),thread)))
        push!(statements,:(@boundscheck _check_vector(tile.layout,$c,$(plan.axis),Val($width),$T)))
        if op === :load
            d = Symbol(:d_,g)
            push!(statements,:($d = $instruction(pointer(tile,$c))))
            for (j,b) in enumerate(group)
                results[plan.words[b]] = n == 1 ? d : :($d[$j])
            end
        else
            args = [:(f.data[$(plan.words[b])]) for b in group]
            push!(statements,:($instruction(pointer(tile,$c),$(n == 1 ? args[1] : Expr(:tuple,args...)))))
        end
    end
    statements, op === :load ? [results[w] for w in 1:length(plan.words)] : Any[]
end
# stmatrix needs sm_90; the CUDACore extension answers from the compile target.
_stmatrix_available() = false

# Loads and stores at the ownership's coordinates. Scalar accesses are
# correct for every static ownership and serve as the oracle. Shared tiles
# of 8- or 16-bit elements whose ownership decomposes into copy-atom blocks
# along the layout's unit-stride axis use ldmatrix and stmatrix; otherwise
# slots that form contiguous, aligned vectors within the tile's declared
# alignment use vector accesses, as whole words for packed element types.
_vector_type(::Type{T},k) where T = k == 1 ? T : NTuple{k,VecElement{T}}
function _vector_access_exprs(plan,::Type{T},AS,packed::Bool,op::Symbol,data) where T
    width = plan.width
    statements = Expr[]
    values = Any[]
    for (g,slots) in enumerate(plan.groups)
        k = packed ? width ÷ 4 : length(slots)          # words or elements per vector
        U = packed ? UInt32 : T
        V = _vector_type(U,k)
        p = Symbol(:p_,g)
        push!(statements,:($p = reinterpret(Core.LLVMPtr{$V,$AS},
            pointer(tile,Layouts.coordinate(ownership,thread,Val($(slots[1]-1)))))))
        if op === :load
            v = Symbol(:v_,g)
            push!(statements,:($v = unsafe_load($p,1,Val($width))))
            append!(values,k == 1 ? [v] : [:($v[$j].value) for j in 1:k])
        else
            first = packed ? (slots[1]-1) ÷ _per_word(T) + 1 : slots[1]
            payload = k == 1 ? :($data[$first]) : Expr(:tuple,[:(VecElement($data[$(first+j-1)])) for j in 1:k]...)
            push!(statements,:(unsafe_store!($p,$payload,1,Val($width))))
        end
    end
    statements, values
end
@generated function load_fragment(::L,tile::MemoryTile{T,AS,ML,A},thread::Integer) where {L,T,AS,ML,A}
    o = _static_instance(L)
    o === nothing && return :(throw(ArgumentError("loads require a static ownership")))
    packed = _element_bits(T) < 32
    axis = _contiguous_axis(ML)
    if AS == 3 && packed && axis !== nothing
        plan = matrix_copy_plan(o,T,axis)
        if plan !== nothing
            statements, words = _matrix_copy_expr(plan,T,:load)
            return quote
                Base.@inline
                @boundscheck 0 <= thread < 32 || throw(BoundsError())
                $(statements...)
                PackedFragment($T,($(words...),),$o)
            end
        end
    end
    vplan = axis === nothing ? nothing : vector_plan(o,T,axis,A)
    if vplan !== nothing
        statements, values = _vector_access_exprs(vplan,T,AS,packed,:load,nothing)
        return quote
            Base.@inline
            ownership = $o
            $(statements...)
            $(packed ? :(PackedFragment($T,($(values...),),ownership)) : :(Fragment(($(values...),),ownership)))
        end
    end
    n = _register_count(o)
    loads = [:(unsafe_load(pointer(tile,Layouts.coordinate(ownership,thread,Val($e))),1,Val($(sizeof(T))))) for e in 0:n-1]
    quote
        Base.@inline
        ownership = $o
        values = Fragment(($(loads...),),ownership)
        $(packed ? :(pack(values)) : :(values))
    end
end
@generated function store!(tile::MemoryTile{T,AS,ML,A},f::Fragment{T,N,L},thread::Integer) where {T,AS,ML,A,N,L}
    o = _static_instance(L)
    o === nothing && return :(throw(ArgumentError("stores require a static ownership")))
    axis = _contiguous_axis(ML)
    vplan = axis === nothing || _element_bits(T) < 32 ? nothing : vector_plan(o,T,axis,A)
    if vplan !== nothing
        statements, _ = _vector_access_exprs(vplan,T,AS,false,:store,:(f.data))
        return quote
            Base.@inline
            ownership = Layouts.layout(f)
            $(statements...)
            nothing
        end
    end
    stores = [:(unsafe_store!(pointer(tile,Layouts.coordinate(ownership,thread,Val($e))),f.data[$(e+1)],1,Val($(sizeof(T))))) for e in 0:N-1]
    quote
        Base.@inline
        ownership = Layouts.layout(f)
        $(stores...)
        nothing
    end
end
@generated function store!(tile::MemoryTile{T,AS,ML,A},f::PackedFragment{T,W,L},thread::Integer) where {T,AS,ML,A,W,L}
    fallback = quote
        Base.@inline
        store!(tile,unpack(f),thread)
    end
    o = _static_instance(L)
    o === nothing && return fallback
    axis = _contiguous_axis(ML)
    axis === nothing && return fallback
    vplan = vector_plan(o,T,axis,A)
    words = if vplan === nothing
        :(store!(tile,unpack(f),thread))
    else
        statements, _ = _vector_access_exprs(vplan,T,AS,true,:store,:(f.data))
        quote
            ownership = Layouts.layout(f)
            $(statements...)
        end
    end
    plan = AS == 3 ? matrix_copy_plan(o,T,axis) : nothing
    plan === nothing && return quote
        Base.@inline
        $words
        nothing
    end
    statements, _ = _matrix_copy_expr(plan,T,:store)
    quote
        Base.@inline
        if _stmatrix_available()
            @boundscheck 0 <= thread < 32 || throw(BoundsError())
            $(statements...)
        else
            $words
        end
        nothing
    end
end
