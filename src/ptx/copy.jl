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
Base.@constprop :aggressive @inline wait_copies(n::Integer) = ptx"cp.async.wait_group"(Val(Int(n)))

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
            args = [:(words[$(plan.words[b])]) for b in group]
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
# The address of slot `e`: the thread's base plus the slot's constant offset
# when the tile's offsets split that way (`base` must be bound), the full
# coordinate otherwise.
_slot_pointer_expr(split::Bool,o,e) = split ?
    :(_slot_address(tile,base,$(Layouts._slot_coordinate(o,e)))) :
    :(pointer(tile,Layouts._coordinate(ownership,thread,Val($e))))
_split_addresses(o,ML) = !(ML <: Layouts.Composition) && tile_offsets(o,ML) !== nothing
_prologue(split::Bool,bounded::Bool) = quote
    $(split || bounded ? :(tc = Layouts._thread_coordinate(ownership,thread)) : nothing)
    $(split ? :(base = pointer(tile,tc)) : nothing)
end
# A bounded tile's slot is valid when its coordinate lies inside the extents.
_valid_expr(o,e,B) = (c = Layouts._slot_coordinate(o,e);
    checks = [c[a] == 0 ? :(tc[$a] < tile.bounds[$a]) : :((tc[$a] + oftype(tc[$a],$(c[a]))) < tile.bounds[$a])
              for a in 1:2 if B.parameters[a] !== Nothing];
    isempty(checks) ? true : foldl((x,y) -> :($x && $y),checks))
function _vector_access_exprs(plan,o,split,bounded,B,::Type{T},AS,packed::Bool,op::Symbol,data) where T
    width = plan.width
    per = packed ? _per_word(T) : 1
    statements = Expr[]
    units = Dict{Int,Any}()            # word (packed) or slot -> loaded value
    for (g,slots) in enumerate(plan.groups)
        k = packed ? width ÷ 4 : length(slots)          # words or elements per vector
        U = packed ? UInt32 : T
        V = _vector_type(U,k)
        members = packed ? [(slots[per*(j-1)+1]-1)÷per+1 for j in 1:k] : slots
        p = Symbol(:p_,g)
        push!(statements,:($p = reinterpret(Core.LLVMPtr{$V,$AS},$(_slot_pointer_expr(split,o,slots[1]-1)))))
        valid = bounded ? _valid_expr(o,slots[1]-1,B) : true
        if op === :load
            v = Symbol(:v_,g)
            load = :(unsafe_load($p,1,Val($width)))
            zeros = k == 1 ? :(zero($U)) : Expr(:tuple,[:(VecElement(zero($U))) for _ in 1:k]...)
            push!(statements,bounded ? :($v = $valid ? $load : $zeros) : :($v = $load))
            for (j,m) in enumerate(members)
                units[m] = k == 1 ? v : :($v[$j].value)
            end
        else
            payload = k == 1 ? :($data[$(members[1])]) : Expr(:tuple,[:(VecElement($data[$m])) for m in members]...)
            store = :(unsafe_store!($p,$payload,1,Val($width)))
            push!(statements,bounded ? :($valid && $store) : store)
        end
    end
    statements, [units[i] for i in sort(collect(keys(units)))]
end
@generated function load_fragment(::L,tile::MemoryTile{T,AS,ML,A,B},thread::Integer) where {L,T,AS,ML,A,B}
    o = _static_instance(L)
    o === nothing && return :(throw(ArgumentError("loads require a static ownership")))
    packed = _element_bits(T) < 32
    bounded = B !== Nothing
    axis = _contiguous_axis(ML)
    if AS == 3 && packed && axis !== nothing && !bounded
        plan = matrix_copy_plan(o,T,axis)
        if plan !== nothing
            statements, words = _matrix_copy_expr(plan,T,:load)
            return quote
                Base.@inline
                @boundscheck 0 <= thread < 32 || throw(BoundsError())
                $(statements...)
                _from_words($T,($(words...),),$o)
            end
        end
    end
    split = _split_addresses(o,ML)
    vplan = vector_plan(o,T,ML,A)
    if vplan !== nothing
        statements, values = _vector_access_exprs(vplan,o,split,bounded,B,T,AS,packed,:load,nothing)
        return quote
            Base.@inline
            ownership = $o
            $(_prologue(split,bounded))
            $(statements...)
            $(packed ? :(_from_words($T,($(values...),),ownership)) : :(Fragment(($(values...),),ownership)))
        end
    end
    n = _register_count(o)
    loads = map(0:n-1) do e
        load = :(unsafe_load($(_slot_pointer_expr(split,o,e)),1,Val($(sizeof(T)))))
        bounded ? :($(_valid_expr(o,e,B)) ? $load : zero($T)) : load
    end
    quote
        Base.@inline
        ownership = $o
        $(_prologue(split,bounded))
        Fragment(($(loads...),),ownership)
    end
end
# An ownership held by one thread needs no thread index: it is zero.
@generated function _only_thread(::O) where O
    o = _static_instance(O)
    o !== nothing && _thread_count(o) == 1 ? :(Int32(0)) :
        :(throw(ArgumentError("several threads hold this ownership; pass the thread index")))
end
Base.@propagate_inbounds load_fragment(o::Layouts.Ownership,tile::MemoryTile) = load_fragment(o,tile,_only_thread(o))
Base.@propagate_inbounds store!(tile::MemoryTile,f::Fragment) = store!(tile,f,_only_thread(Layouts.layout(f)))
@generated function store!(tile::MemoryTile{T,AS,ML,A,B},f::Fragment{T,N,L},thread::Integer) where {T,AS,ML,A,B,N,L}
    o = _static_instance(L)
    bounded = B !== Nothing
    o === nothing && return quote
        Base.@inline
        ownership = Layouts.layout(f)
        $(bounded ? :(throw(ArgumentError("bounded stores require a static ownership"))) : nothing)
        $([:(unsafe_store!(pointer(tile,Layouts._coordinate(ownership,thread,Val($e))),_value(f,Val($(e+1))),1,Val($(sizeof(T))))) for e in 0:N-1]...)
        nothing
    end
    split = _split_addresses(o,ML)
    # A replicated value is written once, by the replica whose replicating bits are zero.
    mask = replica_mask(o)
    writer = mask == 0 ? true : :((thread & oftype(thread,$mask)) == zero(thread))
    scalar = quote
        ownership = Layouts.layout(f)
        $(_prologue(split,bounded))
        $(map(0:N-1) do e
            store = :(unsafe_store!($(_slot_pointer_expr(split,o,e)),_value(f,Val($(e+1))),1,Val($(sizeof(T)))))
            bounded ? :($(_valid_expr(o,e,B)) && $store) : store
        end...)
    end
    packed = _element_bits(T) < 32
    vplan = vector_plan(o,T,ML,A)
    words = if vplan === nothing
        scalar
    else
        statements, _ = _vector_access_exprs(vplan,o,split,bounded,B,T,AS,packed,:store,packed ? :words : :(f.data))
        quote
            ownership = Layouts.layout(f)
            $(_prologue(split,bounded))
            $(packed ? :(words = _pack_words(f)) : nothing)
            $(statements...)
        end
    end
    axis = _contiguous_axis(ML)
    plan = AS == 3 && packed && axis !== nothing && !bounded ? matrix_copy_plan(o,T,axis) : nothing
    plan === nothing && return quote
        Base.@inline
        if $writer
            $words
        end
        nothing
    end
    statements, _ = _matrix_copy_expr(plan,T,:store)
    quote
        Base.@inline
        if _stmatrix_available()
            @boundscheck 0 <= thread < 32 || throw(BoundsError())
            words = _pack_words(f)
            $(statements...)
        else
            $words
        end
        nothing
    end
end
