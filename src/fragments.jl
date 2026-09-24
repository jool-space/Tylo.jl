"""
    Fragment(values::NTuple, ownership)

This thread's immutable share of a logical tile: its values and the
[`Ownership`](@ref Layouts.Ownership) that says which logical coordinate
each value holds. `f.data` is always the tuple of logical values, whatever
their width. Internally, 8- and 16-bit values live packed in the 32-bit
register words instructions consume, adjacent slots lowest first, so loads
produce and instructions consume them without repacking, and a fragment
carried through a loop stays in its words. Arithmetic preserves ownership.
"""
struct Fragment{T,N,L,D}
    storage::D
    ownership::L
    function Fragment{T,N,L,D}(storage::D,ownership::L) where {T,N,L,D}
        N isa Int && N > 0 || throw(ArgumentError("a fragment must contain values"))
        isbitstype(T) && isbitstype(L) || throw(ArgumentError("register values and ownership must be isbits"))
        N == _register_count(ownership) || throw(DimensionMismatch("value count differs from ownership"))
        new{T,N,L,D}(storage,ownership)
    end
end
_register_count(l::Layouts.Ownership) = Int(size(l.mapping)[2])
Layouts.layout(f::Fragment) = getfield(f,:ownership)
Layouts.layout(o::Layouts.Ownership) = o
Base.eltype(::Type{<:Fragment{T}}) where T = T
Base.eltype(f::Fragment) = eltype(typeof(f))

# Register words: `32 ÷ bits` consecutive values per 32-bit word, lowest
# first, as instructions consume 8- and 16-bit operands. A fragment of such
# values in complete words is stored as its words.
_per_word(::Type{T}) where T = 32 ÷ _element_bits(T)
_packed(::Type{T},N) where T = _element_bits(T) in (8,16) && N % _per_word(T) == 0
_storage_type(::Type{T},N) where T = _packed(T,N) ? NTuple{N÷_per_word(T),UInt32} : NTuple{N,T}

@inline function Fragment(data::NTuple{N,T},ownership::L) where {N,T,L}
    D = _storage_type(T,N)
    Fragment{T,N,L,D}(_to_storage(D,data),ownership)
end
@generated function _to_storage(::Type{D},data::NTuple{N,T}) where {D,N,T}
    D === NTuple{N,T} && return :data
    W = N ÷ _per_word(T); bits = _element_bits(T); per = 32 ÷ bits; U = _carrier(T)
    # Adjacent values join into carriers of twice the width, then those: a
    # value pair produced as one 16-bit register (a paired conversion) then
    # folds back to that register instead of being split into bytes.
    function join(values,width)
        length(values) == 1 && return only(values)
        wide = width == 8 ? UInt16 : UInt32
        pairs = [:($wide($(values[2j-1])) | ($wide($(values[2j])) << $width)) for j in 1:length(values)÷2]
        join(pairs,2width)
    end
    words = [join([:(reinterpret($U, data[$(per*(i-1)+j)])) for j in 1:per],bits) for i in 1:W]
    quote
        Base.@inline
        ($(words...),)
    end
end
@generated function _values(f::Fragment{T,N,L,D}) where {T,N,L,D}
    D === NTuple{N,T} && return :(getfield(f,:storage))
    bits = _element_bits(T); per = 32 ÷ bits; U = _carrier(T)
    values = [:(reinterpret($T,(w[$(i÷per+1)] >> $(bits*(i%per))) % $U)) for i in 0:N-1]
    quote
        Base.@inline
        w = getfield(f,:storage)
        ($(values...),)
    end
end
# The `I`-th value alone, unpacking only its word.
@generated function _value(f::Fragment{T,N,L,D},::Val{I}) where {T,N,L,D,I}
    D === NTuple{N,T} && return :(getfield(f,:storage)[$I])
    bits = _element_bits(T); per = 32 ÷ bits; U = _carrier(T)
    :(reinterpret($T,(getfield(f,:storage)[$((I-1)÷per+1)] >> $(bits*((I-1)%per))) % $U))
end
@inline Base.getproperty(f::Fragment,s::Symbol) = s === :data ? _values(f) : getfield(f,s)
Base.propertynames(::Fragment) = (:data,:ownership)

"The register words of a fragment: its storage for 8- and 16-bit values, reinterpreted 32-bit values otherwise."
@generated function _pack_words(f::Fragment{T,N,L,D}) where {T,N,L,D}
    D <: NTuple{<:Any,UInt32} && _packed(T,N) && return :(getfield(f,:storage))
    _element_bits(T) == 32 && return :(($([:(reinterpret(UInt32,getfield(f,:storage)[$i])) for i in 1:N]...),))
    :(throw(ArgumentError("register words require complete groups of 8- or 16-bit values, or 32-bit values")))
end
"A fragment of `T` values from register words, without conversion."
@generated function _from_words(::Type{T},words::NTuple{W,UInt32},ownership::L) where {T,W,L}
    bits = _element_bits(T)
    bits == 32 && return :(Fragment(($([:(reinterpret($T,words[$i])) for i in 1:W]...),),ownership))
    N = W*(32 ÷ bits)
    :(Fragment{$T,$N,L,NTuple{$W,UInt32}}(words,ownership))
end
"The same storage under another ownership with as many slots."
@inline _with_ownership(f::Fragment{T,N,L,D},o) where {T,N,L,D} = Fragment{T,N,typeof(o),D}(getfield(f,:storage),o)

"""
    fill(x, ownership)
    zeros(T, ownership)

A fragment holding `x` (or `zero(T)`) in every slot of `ownership`.
"""
Base.@constprop :aggressive @inline Base.fill(x,o::Layouts.Ownership) = Fragment(ntuple(_ -> x,Val(_register_count(o))),o)
Base.@constprop :aggressive @inline Base.zeros(::Type{T},o::Layouts.Ownership) where T = fill(zero(T),o)
"""
    Fragment(fragment, ownership)

The same values, slot for slot, under another ownership with as many slots:
registers are untouched. For example, a thread's running per-column
statistics become one row of a warp-wide matrix to reduce across lanes.
"""
@inline function Fragment(f::Fragment{T,N},ownership::Layouts.Ownership) where {T,N}
    N == _register_count(ownership) || throw(DimensionMismatch("value count differs from ownership"))
    _with_ownership(f,ownership)
end
"""
    reinterpret(T, fragment)

The same bits as values of `T`, which must have the element size of the
fragment's values, under the same ownership.
"""
@inline function Base.reinterpret(::Type{T},f::Fragment{S,N,L,D}) where {T,S,N,L,D}
    isbitstype(T) && sizeof(T) == sizeof(S) || throw(ArgumentError("reinterpret requires element types of one size"))
    D === NTuple{N,S} ? Fragment(map(x -> reinterpret(T,x),getfield(f,:storage)),Layouts.layout(f)) :
        Fragment{T,N,L,D}(getfield(f,:storage),Layouts.layout(f))
end

"""
    Fragment{T}(fragment)

Convert every value to `T`, preserving ownership. Conversion follows
[`convert_element`](@ref): Microfloat destinations round to nearest even and
saturate. Conversions between Microfloats and wide floats go through
Microfloats' vector funnel, and FP32 to BF16/FP16 through paired
conversions, so a kernel gets one native `cvt` per two values where its
target has the instruction.
"""
@inline Fragment{T}(f::Fragment) where T = _convert_fragment(T,f)
@inline _convert_fragment(::Type{T},f::Fragment{T}) where T = f
@inline _convert_fragment(::Type{T},f::Fragment) where T = map(Base.Fix1(convert_element,T),f)
@inline _convert_fragment(::Type{T},f::Fragment{T}) where T<:Microfloat = f
@inline function _convert_fragment(::Type{T},f::Fragment{S,N}) where {T<:Microfloat,S,N}
    Fragment(Tuple(Microfloats.cvt(Microfloats.SVector{N,T},f.data,RoundNearest,SAT)),Layouts.layout(f))
end
@inline function _convert_fragment(::Type{F},f::Fragment{T,N}) where {F<:Microfloats.WideFloat,T<:Microfloat,N}
    Fragment(Tuple(Microfloats.cvt(Microfloats.SVector{N,F},f.data)),Layouts.layout(f))
end

@inline function Base.map(op::F, f::Fragment{T,N}) where {F,T,N}
    Fragment(@rtuple(i -> op(f.data[i]), 1:N), Layouts.layout(f))
end


function Base.show(io::IO,f::Fragment)
    print(io,"Fragment(")
    show(io,f.data)
    print(io,", ")
    show(io,Layouts.layout(f))
    print(io,')')
end
