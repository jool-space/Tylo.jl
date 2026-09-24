# Symbolic layout algebra.
#
# Every operation returns a layout whose function is a stated expression of
# its inputs' functions on a stated domain. Leaf staticness is preserved:
# arithmetic between static leaves stays static, and anything touching a
# runtime leaf becomes a runtime value of that leaf's type. Decisions about
# structure (which modes merge, drop or split) depend on types only, never
# on runtime values, so wholly static inputs give wholly static results and
# mixed inputs keep one stable type. The structural operations are generated
# functions: they walk the layout types once and emit straight-line code, so
# nothing here depends on inference following a recursion. Every operation
# is exact; one that cannot represent its result as a layout throws.

# ── Leaf arithmetic ─────────────────────────────────────────────────────
# Static.jl folds operations between static leaves. A runtime leaf keeps
# its own integer type, so Int32 strides stay Int32.
@inline _op(op,a::StaticInt,b::StaticInt) = op(a,b)
@inline _op(op,a::StaticInt,b::Integer) = op(oftype(b,Int(a)),b)
@inline _op(op,a::Integer,b::StaticInt) = op(a,oftype(a,Int(b)))
@inline _op(op,a::Integer,b::Integer) = op(a,b)
_isstatic(::StaticInt) = true
_isstatic(::Integer) = false
_isstatic(t::Tuple) = all(map(_isstatic,t))
_isstatic(l::Layout) = _isstatic(shape(l)) && _isstatic(strides(l))

# ── Generation-time leaves ──────────────────────────────────────────────
# While generating, a leaf is an `Int` when static or an `Expr` reading the
# runtime value from an argument. Arithmetic folds static pairs and
# otherwise emits `_op`, so the emitted code preserves leaf types.
const Leaf = Union{Int,Expr,Symbol}
_lit(x::Int) = :(static($x))
_lit(x::Union{Expr,Symbol}) = x
_sop(op::Symbol,a::Leaf,b::Leaf) =
    a isa Int && b isa Int ? getfield(Base,op)(a,b) : :(_op($op,$(_lit(a)),$(_lit(b))))
_sis(x::Leaf,n::Int) = x isa Int && x == n
_seq(a::Leaf,b::Leaf) = a isa Int && b isa Int && a == b
struct Inexact <: Exception
    msg::String
end
# Layout types as trees: a tuple node is a Vector, a leaf is `(shape, stride)`.
_leaf(::Type{StaticInt{N}},access) where N = N
_leaf(::Type,access) = access
function _tree(::Type{S},::Type{D},sacc,dacc) where {S,D}
    S <: Tuple || return (_leaf(S,sacc),_leaf(D,dacc))
    [_tree(S.parameters[i],D.parameters[i],:($sacc[$i]),:($dacc[$i])) for i in 1:length(S.parameters)]
end
_tree(::Type{Layout{S,D}},var::Symbol) where {S,D} = _tree(S,D,:($var.shape),:($var.strides))
_leaves(node::Vector) = reduce(vcat,map(_leaves,node); init=Tuple{Leaf,Leaf}[])
_leaves(leaf::Tuple) = [leaf]
_leaves(::Type{L},var::Symbol) where L<:Layout = _leaves(_tree(L,var))
_shape_expr(node::Vector) = Expr(:tuple,map(_shape_expr,node)...)
_shape_expr(leaf::Tuple) = _lit(leaf[1])
_stride_expr(node::Vector) = Expr(:tuple,map(_stride_expr,node)...)
_stride_expr(leaf::Tuple) = _lit(leaf[2])
_layout_expr(node::Vector) = isempty(node) ? :(trivial()) : :(Layout($(_shape_expr(node)),$(_stride_expr(node))))
_unwrap(node::Vector) = length(node) == 1 ? node[1] : node
# Generate a body, turning an inexact algebra into a thrown ArgumentError.
function _generate(f)
    try
        f()
    catch e
        e isa Inexact ? :(throw(ArgumentError($(e.msg)))) : rethrow()
    end
end

"The trivial layout `(1):(0)`: one element at offset zero."
trivial() = Layout((static(1),),(static(0),))
"The number of top-level modes."
rank(l::Layout) = length(shape(l))
"Mode `i` of `l` as a layout of its own."
@inline mode(l::Layout,i::Int) = _mode(shape(l)[i],strides(l)[i])
@inline _mode(s::Tuple,d::Tuple) = Layout(s,d)
@inline _mode(s::IntLike,d::IntLike) = Layout((s,),(d,))
"Drop the hierarchy: one flat mode per leaf, same function on the linearized domain."
@generated flatten(l::L) where L<:Layout = _layout_expr(_leaves(L,:l))
"Drop modes with a static zero stride; the function is unchanged on the domain."
@generated filter_zeros(l::L) where L<:Layout = _layout_expr([p for p in _leaves(L,:l) if !_sis(p[2],0)])

"""
    concat(a::Layout, b::Layout...)

The layout with one mode per input: `concat(a, b)((i, j)) == a(i) + b(j)`.
Domains multiply; images add.
"""
@inline concat(ls::Layout...) = Layout(map(shape,ls),map(strides,ls))

function _coalesce(pairs)
    out = Tuple{Leaf,Leaf}[]
    for (s,d) in pairs
        _sis(s,1) && continue
        if !isempty(out)
            ps, pd = out[end]
            if _seq(d,_sop(:*,ps,pd))
                out[end] = (_sop(:*,ps,s),pd)
                continue
            end
        end
        push!(out,(s,d))
    end
    out
end
# The same merge from the right, treating the last mode as unbounded: a
# composition may read past the last extent, so the function must be kept
# on all natural numbers, not only on the domain.
function _coalesce_x(pairs)
    s, d = pairs[end]
    out = Tuple{Leaf,Leaf}[(_sis(s,1) ? 2 : s,d)]
    for (s,d) in reverse(pairs[1:end-1])
        _sis(s,1) && continue
        hs, hd = out[1]
        _seq(_sop(:*,s,d),hd) ? (out[1] = (_sop(:*,s,hs),d)) : pushfirst!(out,(s,d))
    end
    out
end

"""
    coalesce(l)

Flatten `l`, drop unit modes and merge adjacent modes whose strides continue
one another, without changing the function on the linearized domain. A mode
drops only when its extent is statically one and merges only when the
strides deciding it are static, so the structure of the result depends on
types alone.
"""
@generated Base.coalesce(l::L) where L<:Layout = _layout_expr(_coalesce(_leaves(L,:l)))

# ── Composition ─────────────────────────────────────────────────────────
_compose(ls,ld,node::Vector,checks) = map(n -> _unwrap(_compose(ls,ld,n,checks)),node)
function _compose(ls,ld,leaf::Tuple,checks)
    s, d = leaf
    _sis(d,0) && return [(s,d)]
    length(ls) == 1 && return [(s,_sop(:*,d,ld[1]))]
    out = Tuple{Leaf,Leaf}[]
    rest_s, rest_d = s, d
    for i in 1:length(ls)-1
        cs, cd = ls[i], ld[i]
        if rest_d isa Int && cs isa Int
            rest_d % cs == 0 || cs % rest_d == 0 ||
                throw(Inexact("composition is not exact: stride $rest_d neither divides nor is a multiple of the mode extent $cs"))
        else
            push!(checks,:(_check_stride($(_lit(rest_d)),$(_lit(cs)))))
        end
        next_s, next_d = _sop(:cld,cs,rest_d), _sop(:cld,rest_d,cs)
        if _sis(next_s,1) || _sis(rest_s,1)
            rest_d = next_d
        else
            new_s = _sop(:min,next_s,rest_s)
            if rest_s isa Int && new_s isa Int
                rest_s % new_s == 0 ||
                    throw(Inexact("composition is not exact: extent $rest_s does not split evenly at the mode boundary $new_s"))
            else
                push!(checks,:(_check_extent($(_lit(rest_s)),$(_lit(new_s)))))
            end
            push!(out,(new_s,_sop(:*,rest_d,cd)))
            rest_s, rest_d = _sop(:÷,rest_s,new_s), next_d
        end
    end
    last_d = _sop(:*,rest_d,ld[end])
    isempty(out) && return [(rest_s,last_d)]
    _sis(rest_s,1) && return out
    push!(out,(rest_s,last_d))
end
@inline _check_stride(d,n) = _op(%,d,n) == 0 || _op(%,n,d) == 0 ||
    throw(ArgumentError("composition is not exact: stride $(Int(d)) neither divides nor is a multiple of the mode extent $(Int(n))"))
@inline _check_extent(s,n) = _op(%,s,n) == 0 ||
    throw(ArgumentError("composition is not exact: extent $(Int(s)) does not split evenly at the mode boundary $(Int(n))"))

"""
    compose(a::Layout, b::Layout)

The layout of `a(b(c))` for every `c` in `b`'s domain, in `b`'s profile:
each mode of `b` is threaded through the flat modes of `a`. The result is
exact. Two divisibility conditions make it representable, and both are
checked, at generation time for static leaves and at run time otherwise: a
mode of `b` must either start on a multiple of an `a` mode or step evenly
inside it, and where a mode of `b` runs past an `a` mode it must do so a
whole number of times. The last mode of `a` is unbounded, as in
`coalesce`, so `b` may exceed `a`'s extent there. A third condition makes
the modes of `b` add: their offsets must not carry across a mode boundary
of `a`, which holds for injective `b` and is checked for static leaves.
Violations throw an `ArgumentError`. A mode of `b` with a static zero
stride passes through.
"""
@generated function compose(a::A,b::B) where {A<:Layout,B<:Layout}
    _generate() do
        checks = Expr[]
        flat = _coalesce_x(_leaves(A,:a))
        _check_carries(map(first,flat),_leaves(B,:b))
        result = _compose(map(first,flat),map(last,flat),_tree(B,:b),checks)
        quote
            $(checks...)
            $(_layout_expr(result))
        end
    end
end
compose(c::Composition,b::Layout) = Composition(c.outer,compose(c.inner,b))
# Composition threads every leaf of `b` through `a` separately and adds the
# results, which is `a(b(c))` only if the leaves' offsets never carry across
# a mode boundary of `a` when added: at every boundary `P` (a product of
# leading extents of `a`), the largest residues modulo `P` of the leaves
# must sum to less than `P`. Injective tilers and ownerships satisfy it;
# overlapping modes may not. Checked for static leaves.
function _check_carries(extents,leaves)
    prefix = 1
    for e in extents[1:end-1]
        e isa Int || return
        prefix *= e
        total = 0
        for (s,d) in leaves
            _sis(d,0) && continue
            s isa Int && d isa Int || return
            total += maximum((d*i) % prefix for i in 0:s-1)
        end
        total < prefix || throw(Inexact("composition is not exact: modes of the inner layout carry across a boundary of $prefix elements of the outer layout"))
    end
end

# ── Complement and inverses ─────────────────────────────────────────────
"""
    complement(l, cotarget=cosize(l))

A layout whose image meets `l`'s only at zero and whose concatenation with
`l` is injective and reaches at least `cotarget`: the modes `l` leaves out
of `[0, cotarget)`, in ascending stride order. The complement of a tile is
the layout of tile positions. `l` needs static strides, or a single mode;
shapes and `cotarget` may be runtime values. Overlapping modes, or a
stride that is not a multiple of the extent covered so far, have no
complement and throw.
"""
complement(l::Layout) = complement(l,cosize(l))
@generated function complement(l::L,cotarget::C) where {L<:Layout,C<:IntLike}
    _generate() do
        target = _leaf(C,:cotarget)
        leaves = [(s,d) for (s,d) in _leaves(L,:l) if !_sis(d,0) && !_sis(s,1)]
        isempty(leaves) && return :(coalesce(Layout((cotarget,),(static(1),))))
        length(leaves) > 1 && any(p -> !(p[2] isa Int),leaves) &&
            throw(Inexact("complement needs static strides in a layout with several modes"))
        sort!(leaves; by=p -> p[2] isa Int ? p[2] : 0)
        out = Tuple{Leaf,Leaf}[]
        running = 1
        for (s,d) in leaves
            if d isa Int && running isa Int
                d % running == 0 || throw(Inexact("no complement: stride $d is not a multiple of the extent $running covered by the smaller strides"))
                d ÷ running >= 1 || throw(Inexact("no complement: a mode with stride $d overlaps the extent $running covered by the smaller strides"))
            end
            push!(out,(_sop(:÷,d,running),running))
            running = _sop(:*,d,s)
        end
        push!(out,(_sop(:cld,target,running),running))
        _layout_expr(_coalesce(out))
    end
end

"""
    right_inverse(l)

For a static `l`, the layout `r` with `l(r(i)) == i` for every `i` in the
longest run `0, 1, 2, …` inside the image of `l`. Composing `l` with it
gives the identity on that run. A layout whose image does not continue
past zero has the trivial inverse.
"""
@generated function right_inverse(l::L) where L<:Layout
    leaves = _leaves(L,:l)
    all(p -> p[1] isa Int && p[2] isa Int,leaves) ||
        return :(throw(ArgumentError("right_inverse needs a static layout")))
    flat = _coalesce(leaves)
    prefix = cumprod([1; first.(flat)])
    out = Tuple{Leaf,Leaf}[]
    current = 1
    for i in sortperm(flat; by=last)
        s, d = flat[i]
        d == current || continue
        push!(out,(s,prefix[i]))
        current = s*d
    end
    _layout_expr(_coalesce(out))
end

"""
    left_inverse(l)

For a static injective `l`, the layout `r` with `r(l(i)) == i` for every
`i` in `l`'s domain. Modes with a zero stride are ignored, so for a
broadcasting layout the result recovers one preimage.
"""
@generated function left_inverse(l::L) where L<:Layout
    _generate() do
        leaves = _leaves(L,:l)
        all(p -> p[1] isa Int && p[2] isa Int,leaves) ||
            throw(Inexact("left_inverse needs a static layout"))
        flat = _coalesce(leaves)
        prefix = cumprod([1; first.(flat)])
        order = [i for i in sortperm(flat; by=last) if flat[i][2] != 0]
        isempty(order) && return :(trivial())
        out = Tuple{Leaf,Leaf}[]
        extent, stride = 1, 0
        for i in order
            s, d = flat[i]
            d % extent == 0 || throw(Inexact("no left inverse: stride $d is not a multiple of the extent $extent covered by the smaller strides"))
            push!(out,(d ÷ extent,stride))
            extent, stride = d, prefix[i]
        end
        push!(out,(flat[order[end]][1],stride))
        _layout_expr(_coalesce(out))
    end
end

# ── Division and products ───────────────────────────────────────────────
_tiler(t::Layout) = t
_tiler(t::IntLike) = Layout((t,),(static(1),))
"""
    logical_divide(l, tile)

Split `l` into `((within tile), (tile index))`: the composition of `l` with
`concat(tile, complement(tile, length(l)))`. A tuple of tiles applies one
per leading top-level mode of `l`, `:` leaving a mode untouched; an integer
tile is the contiguous tile of that extent, static or runtime like any
leaf. A tile that does not divide the layout exactly throws; nothing is
padded.
"""
function logical_divide(l::Layout,t::Union{Layout,IntLike})
    tiles = concat(_tiler(t),complement(_tiler(t),length(l)))
    length(tiles) == length(l) ||
        throw(ArgumentError("the tile does not divide the layout: $(Int(length(_tiler(t)))) elements per tile, $(Int(length(l))) in the layout"))
    compose(l,tiles)
end
logical_divide(l::Layout,::Colon) = l
function logical_divide(l::Layout,t::Tuple)
    length(t) <= rank(l) || throw(ArgumentError("more tiles than modes"))
    Layout(_by_mode(logical_divide,shape(l),strides(l),t)...)
end
logical_divide(c::Composition,t) = Composition(c.outer,logical_divide(c.inner,t))
# Apply a tile per leading mode; the remaining modes pass through.
@inline _by_mode(f,s::Tuple,d::Tuple,::Tuple{}) = (s,d)
@inline function _by_mode(f,s::Tuple,d::Tuple,t::Tuple)
    m = f(_mode(first(s),first(d)),first(t))
    rs, rd = _by_mode(f,Base.tail(s),Base.tail(d),Base.tail(t))
    ((_single(shape(m)),rs...),(_single(strides(m)),rd...))
end
@inline _single(t::Tuple) = length(t) == 1 ? t[1] : t
@inline _single(x) = x
"""
    zipped_divide(l, tiles)

`logical_divide` regrouped as `((tile modes...), (tile indices..., rest...))`.
Every entry of `tiles` must be a layout or an extent.
"""
zipped_divide(l::Layout,t::Union{Layout,IntLike}) = logical_divide(l,t)
function zipped_divide(l::Layout,t::Tuple)
    d = logical_divide(l,t)
    ts, td, is, id = _unzip(shape(d),strides(d),t)
    Layout((ts,is),(td,id))
end
zipped_divide(c::Composition,t) = Composition(c.outer,zipped_divide(c.inner,t))
@inline _unzip(s::Tuple,d::Tuple,::Tuple{}) = ((),(),s,d)
@inline function _unzip(s::Tuple,d::Tuple,t::Tuple)
    first(t) isa Colon && throw(ArgumentError("zipped_divide needs a tile in every listed mode"))
    ts, td, is, id = _unzip(Base.tail(s),Base.tail(d),Base.tail(t))
    ((_single(first(s)[1]),ts...),(_single(first(d)[1]),td...),
     (_single(first(s)[2]),is...),(_single(first(d)[2]),id...))
end
"""
    tiled_divide(l, tiles)

`zipped_divide` with the index modes unpacked: `((tile modes...), index..., rest...)`.
"""
function tiled_divide(l::Layout,t::Tuple)
    z = zipped_divide(l,t)
    Layout((shape(z)[1],shape(z)[2]...),(strides(z)[1],strides(z)[2]...))
end
tiled_divide(c::Composition,t) = Composition(c.outer,tiled_divide(c.inner,t))

"""
    logical_product(block, tile)

Repeat `block` at the positions `tile` enumerates: `((block), (tile))`, the
second mode being `tile` composed with the complement of `block`. With a
compact block this places copies of the block `length(block)` apart, in
`tile`'s order. A tuple of tiles applies one per leading top-level mode.
"""
logical_product(b::Layout,t::Union{Layout,IntLike}) =
    concat(b,compose(complement(b,_op(*,length(b),cosize(_tiler(t)))),_tiler(t)))
logical_product(b::Layout,::Colon) = b
function logical_product(b::Layout,t::Tuple)
    length(t) <= rank(b) || throw(ArgumentError("more tiles than modes"))
    Layout(_by_mode(logical_product,shape(b),strides(b),t)...)
end
"""
    blocked_product(block, tile)

Repeat `block` over `tile` mode by mode: mode `i` of the result is
`(block mode i, tile mode i)`, the block being the faster index. Ranks are
padded with unit modes.
"""
function blocked_product(b::Layout,t::Layout)
    r = max(rank(b),rank(t))
    p = logical_product(_pad(b,Val(r)),_pad(t,Val(r)))
    Layout(map(tuple,shape(p)[1],shape(p)[2]),map(tuple,strides(p)[1],strides(p)[2]))
end
"""
    raked_product(block, tile)

Like `blocked_product`, with the tile as the faster index in every mode, so
the copies of the block interleave.
"""
function raked_product(b::Layout,t::Layout)
    r = max(rank(b),rank(t))
    p = logical_product(_pad(b,Val(r)),_pad(t,Val(r)))
    Layout(map(tuple,shape(p)[2],shape(p)[1]),map(tuple,strides(p)[2],strides(p)[1]))
end
@inline _pad(l::Layout,::Val{R}) where R = rank(l) == R ? l :
    Layout((shape(l)...,ntuple(_ -> static(1),Val(R-rank(l)))...),(strides(l)...,ntuple(_ -> static(0),Val(R-rank(l)))...))

# ── Slicing ─────────────────────────────────────────────────────────────
"""
    slice(l, coord) -> (layout, offset)

Fix the modes where `coord` holds an integer and keep those where it holds
`:`. The returned layout maps the kept coordinates and `offset` is the
contribution of the fixed ones, so `l(c) == layout(kept) + offset`.
"""
function slice(l::Layout,coord::Tuple)
    ss, ds, offset = _slice(shape(l),strides(l),coord)
    _kept(ss,ds), offset
end
_kept(::Tuple{},::Tuple{}) = trivial()
_kept(s::Tuple,d::Tuple) = Layout(s,d)
_slice(s,d,::Colon) = ((s,),(d,),static(0))
_slice(s::IntLike,d::IntLike,c::IntLike) = ((),(),_op(*,c,d))
_slice(s::Tuple,d::Tuple,c::IntLike) = ((),(),_eval(s,d,c))
function _slice(s::Tuple,d::Tuple,c::Tuple)
    length(s) == length(c) || throw(ArgumentError("coordinate rank mismatch"))
    pieces = map(_slice,s,d,c)
    _cat(map(p -> p[1],pieces)...), _cat(map(p -> p[2],pieces)...), +(map(p -> p[3],pieces)...)
end
@inline _cat() = ()
@inline _cat(t::Tuple,rest::Tuple...) = (t...,_cat(rest...)...)

# ── Exact host queries ──────────────────────────────────────────────────
# Enumeration over a layout's domain: the oracle for the algebra above and
# the check for properties the algebra does not decide.
"Whether two layouts have the same function on the same domain size."
agree(a::AbstractLayout,b::AbstractLayout) =
    Int(length(a)) == Int(length(b)) && all(Int(a(i)) == Int(b(i)) for i in 0:Int(length(a))-1)
"Whether distinct coordinates map to distinct offsets."
is_injective(l::AbstractLayout) = allunique(Int(l(i)) for i in 0:Int(length(l))-1)
"Whether the image is exactly `[0, length(l))`."
is_compact(l::AbstractLayout) = Set(Int(l(i)) for i in 0:Int(length(l))-1) == Set(0:Int(length(l))-1)
