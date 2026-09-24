@inline function _check_vector(l,c,axis,::Val{V},::Type{T}) where {V,T}
    offset = l(c)
    offset*sizeof(T) % 16 == 0 || throw(ArgumentError("unaligned 16-byte vector"))
    for i in 1:V-1
        q = ntuple(j -> c[j] + (j == axis ? oftype(c[j],i) : zero(c[j])),2)
        l(q) == offset+i || throw(ArgumentError("noncontiguous 16-byte vector"))
    end
    nothing
end

@inline commit_copies() = ptx"cp.async.commit_group"()
Base.@constprop :aggressive @inline wait_copies(n::Integer) = ptx"cp.async.wait_group"(Val(Int(n)))

# One cp.async of `W` bytes: `.cg` (L2 only) for 16 bytes, `.ca` otherwise.
# With `bytes` below `W`, only that many are read and the rest are zero.
@inline _cp_async(dst,src,::Val{16}) = ptx"cp.async.cg.shared.global"(dst,src,Val(16))
@inline _cp_async(dst,src,::Val{W}) where W = ptx"cp.async.ca.shared.global"(dst,src,Val(W))
@generated function _cp_async(dst::Core.LLVMPtr{T,3},src::Core.LLVMPtr{S,1},::Val{W},bytes::UInt32) where {T,S,W}
    W in (4,8,16) || return :(throw(ArgumentError("cp.async copies 4, 8 or 16 bytes")))
    asm = "cp.async.$(W == 16 ? "cg" : "ca").shared.global [\$0], [\$1], $W, \$2;"
    quote
        Base.@inline
        PTX.@asmcall($asm,"r,l,r,~{memory}",true,Nothing,
                     Tuple{Core.LLVMPtr{$T,3},Core.LLVMPtr{$S,1},UInt32},dst,src,bytes)
        nothing
    end
end

"""
    copy_async!(ownership, destination::SharedTile, source::GlobalTile, thread)

Copy each thread's values of `ownership` from a global tile to a shared tile
of the same shape and element type, without passing through registers. The
slots group into the widest vectors, 16, 8 or 4 bytes, contiguous and
aligned in both tiles within their declared alignments, each one
`cp.async`; values that form no such vector are copied by ordinary loads
and stores. A replicated value is copied once. A bounded source zero-fills
the elements outside it, a vector at a time (see [`bound`](@ref)).

Close the copies with `commit_copies()`, wait with `wait_copies(n)`, then
synchronize the threads that read the destination: neither the copy nor the
wait is a barrier, and ordinary stores become visible only through it.
[`distribute`](@ref) gives the usual coalesced distribution.
"""
@generated function copy_async!(::O,dst::MemoryTile{T,3,DL,DA,DB},src::MemoryTile{T,1,SL,SA,SB},
        thread::Integer) where {O,T,DL,DA,DB,SL,SA,SB}
    o = _static_instance(O)
    o === nothing && return :(throw(ArgumentError("copies require a static ownership")))
    DB === Nothing || return :(throw(ArgumentError("bound the source of a copy, not its destination")))
    bounded = SB !== Nothing
    shape = map(Int,size(o))
    dsplit, ssplit = _split_addresses(o,DL), _split_addresses(o,SL)
    plan = copy_plan(o,T,DL,DA,SL,SA)
    statements = Expr[]
    if plan !== nothing
        W = plan.width
        for slots in plan.groups
            e = slots[1]-1
            d, s = _slot_pointer_expr(dsplit,o,e,:dst,:dbase), _slot_pointer_expr(ssplit,o,e,:src,:sbase)
            if bounded
                # Elements outside the source's bound are not read: the copy
                # reads a prefix of the vector, or nothing, and zero-fills.
                vb = _vector_bounds(o,slots,SB,:src)
                vb.check === nothing || push!(statements,vb.check)
                inside = vb.cut === nothing ? W : :($(_inside_expr(vb.cut,length(slots),:src))*$(sizeof(T)))
                push!(statements,:(_cp_async($d,$s,Val($W),ifelse($(vb.valid),$inside % UInt32,UInt32(0)))))
            else
                push!(statements,:(_cp_async($d,$s,Val($W))))
            end
        end
    else
        for e in 0:_register_count(o)-1
            d, s = _slot_pointer_expr(dsplit,o,e,:dst,:dbase), _slot_pointer_expr(ssplit,o,e,:src,:sbase)
            load = :(unsafe_load($s,1,Val($(sizeof(T)))))
            value = bounded ? :($(_valid_expr(o,e,SB,:src)) ? $load : zero($T)) : load
            push!(statements,:(unsafe_store!($d,$value,1,Val($(sizeof(T))))))
        end
    end
    mask = replica_mask(o)
    quote
        Base.@inline
        size(dst) == size(src) == $(Tuple(shape)) || throw(DimensionMismatch("the tiles must have the ownership's shape"))
        @boundscheck 0 <= thread < $(_thread_count(o)) || throw(BoundsError())
        ownership = $o
        $(dsplit || ssplit || bounded ? :(tc = Layouts._thread_coordinate(ownership,thread)) : nothing)
        $(dsplit ? :(dbase = pointer(dst,tc)) : nothing)
        $(ssplit ? :(sbase = pointer(src,tc)) : nothing)
        if $(mask == 0 ? true : :((thread & oftype(thread,$mask)) == zero(thread)))
            $(statements...)
        end
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
_slot_pointer_expr(split::Bool,o,e,tile=:tile,base=:base) = split ?
    :(_slot_address($tile,$base,$(Layouts._slot_coordinate(o,e)))) :
    :(pointer($tile,Layouts._coordinate(ownership,thread,Val($e))))
# A swizzle does not add, so swizzled tiles address every slot in full.
_split_addresses(o,ML) = !_swizzled(ML) && tile_offsets(o,ML) !== nothing
_swizzled(::Type) = false
_swizzled(::Type{<:Layouts.Composition}) = true
_swizzled(::Type{Layouts.Window{S,L,O}}) where {S,L,O} = _swizzled(L)
_prologue(split::Bool,bounded::Bool,tile=:tile,base=:base) = quote
    $(split || bounded ? :(tc = Layouts._thread_coordinate(ownership,thread)) : nothing)
    $(split ? :($base = pointer($tile,tc)) : nothing)
end
# A bounded tile's slot is valid when its coordinate lies inside the extents.
_valid_expr(o,e,B,tile=:tile) = (c = Layouts._slot_coordinate(o,e);
    checks = [c[a] == 0 ? :(tc[$a] < $tile.bounds[$a]) : :((tc[$a] + oftype(tc[$a],$(c[a]))) < $tile.bounds[$a])
              for a in 1:2 if B.parameters[a] !== Nothing];
    isempty(checks) ? true : foldl((x,y) -> :($x && $y),checks))
# How the bounds cut a vector, from its slots in storage order. Along a
# bounded axis its elements share one coordinate (the vector is inside or
# outside as a whole), or take consecutive coordinates (the bound keeps a
# prefix: `cut` names that axis and the first coordinate), or neither: then
# the bound must not fall inside it, checked, and the first element decides.
function _vector_bounds(o,slots,B,tile=:tile)
    whole, cut, checks = Expr[], nothing, Expr[]
    for a in 1:2
        B.parameters[a] === Nothing && continue
        cs = [Layouts._slot_coordinate(o,e-1)[a] for e in slots]
        lo, hi = extrema(cs)
        if cut === nothing && lo != hi && cs == cs[1] .+ (0:length(cs)-1)
            cut = (a,cs[1])
            continue
        end
        push!(whole,_below(a,cs[1],tile))
        lo == hi || push!(checks,:((tc[$a] + oftype(tc[$a],$hi) < $tile.bounds[$a]) | (tc[$a] + oftype(tc[$a],$lo) >= $tile.bounds[$a])))
    end
    valid = isempty(whole) ? true : foldl((x,y) -> :($x && $y),whole)
    check = isempty(checks) ? nothing : :(@boundscheck $(foldl((x,y) -> :($x & $y),checks)) ||
        throw(ArgumentError("a bound falls inside a vector of scattered coordinates; bound at a multiple of its extent")))
    (; valid, cut, check)
end
_below(a,c,tile) = c == 0 ? :(tc[$a] < $tile.bounds[$a]) : :((tc[$a] + oftype(tc[$a],$c)) < $tile.bounds[$a])
# Elements of a cut vector inside the bound: from none to all `k`.
_inside_expr((a,c),k,tile) = :(clamp(Int($tile.bounds[$a]) - (Int(tc[$a]) + $c),0,$k))
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
        vb = bounded ? _vector_bounds(o,slots,B) : (; valid=true, cut=nothing, check=nothing)
        vb.check === nothing || push!(statements,vb.check)
        n = Symbol(:n_,g)
        vb.cut === nothing || push!(statements,:($n = $(_inside_expr(vb.cut,length(slots),:tile))))
        full = vb.cut === nothing ? vb.valid : :($(vb.valid) & ($n == $(length(slots))))
        some = vb.cut === nothing ? false : :($(vb.valid) & ($n > 0))
        elements = :(reinterpret(Core.LLVMPtr{$T,$AS},$p))
        if op === :load
            v = Symbol(:v_,g)
            load = :(unsafe_load($p,1,Val($width)))
            zeros = k == 1 ? :(zero($U)) : Expr(:tuple,[:(VecElement(zero($U))) for _ in 1:k]...)
            if vb.cut !== nothing
                # The elements inside the bound one by one, the rest zero, as the vector.
                xs = Expr(:tuple,[:($(j-1) < $n ? unsafe_load($elements,$j,Val($(sizeof(T)))) : zero($T)) for j in eachindex(slots)]...)
                words = packed ? :(_to_storage(NTuple{$k,UInt32},$xs)) : xs
                partial = k == 1 ? :($words[1]) : :(map(VecElement,$words))
                push!(statements,:($v = $full ? $load : $some ? $partial : $zeros))
            else
                push!(statements,bounded ? :($v = $full ? $load : $zeros) : :($v = $load))
            end
            for (j,m) in enumerate(members)
                units[m] = k == 1 ? v : :($v[$j].value)
            end
        else
            payload = k == 1 ? :($data[$(members[1])]) : Expr(:tuple,[:(VecElement($data[$m])) for m in members]...)
            store = :(unsafe_store!($p,$payload,1,Val($width)))
            if vb.cut !== nothing
                partial = Expr(:block,[:($(j-1) < $n && unsafe_store!($elements,_value(f,Val($e)),$j,Val($(sizeof(T))))) for (j,e) in enumerate(slots)]...)
                push!(statements,:(if $full; $store; elseif $some; $partial; end))
            else
                push!(statements,bounded ? :($full && $store) : store)
            end
        end
    end
    statements, [units[i] for i in sort(collect(keys(units)))]
end
# The tile has the ownership's shape and the thread is one of its threads,
# checked unless elided; the generated bodies below then address freely.
Base.@propagate_inbounds function load_fragment(o::Layouts.Ownership,tile::MemoryTile,thread::Integer)
    @boundscheck _check_access(o,tile,thread)
    _load_fragment(o,tile,thread)
end
Base.@propagate_inbounds function store!(tile::MemoryTile,f::Fragment,thread::Integer)
    eltype(f) === eltype(tile) ||
        throw(ArgumentError("storing $(eltype(f)) values to a $(eltype(tile)) tile; convert with Fragment{$(eltype(tile))}(f)"))
    @boundscheck _check_access(Layouts.layout(f),tile,thread)
    _store!(tile,f,thread)
end
@inline function _check_access(o,tile,thread)
    size(tile) == size(o) || throw(DimensionMismatch("the tile must have the ownership's shape"))
    0 <= thread < threads(o) || throw(BoundsError())
    nothing
end
@generated function _load_fragment(::L,tile::MemoryTile{T,AS,ML,A,B},thread::Integer) where {L,T,AS,ML,A,B}
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
@generated function _store!(tile::MemoryTile{T,AS,ML,A,B},f::Fragment{T,N,L},thread::Integer) where {T,AS,ML,A,B,N,L}
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
