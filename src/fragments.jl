"""
    Fragment(values::NTuple, ownership)

This thread's immutable share of a logical tile. The ownership layout maps
`(thread, value_slot)` to logical coordinates; it does not describe memory
strides, and the values need not form a row or be contiguous. Arithmetic
preserves ownership. Collective support depends on that ownership layout.
"""
struct Fragment{T,N,L}
    data::NTuple{N,T}
    ownership::L
    function Fragment(data::NTuple{N,T},ownership::L) where {N,T,L}
        N > 0 || throw(ArgumentError("a fragment must contain values"))
        isbitstype(T) && isbitstype(L) || throw(ArgumentError("register values and ownership must be isbits"))
        N == _register_count(ownership) || throw(DimensionMismatch("value count differs from ownership"))
        new{T,N,L}(data,ownership)
    end
end
_register_count(::Union{Layouts.LocalOwnership{N},Layouts.StripedOwnership{N}}) where N = N
_register_count(l::Layouts.Ownership) = Int(size(l.mapping)[2])
Layouts.layout(f::Fragment) = f.ownership

"""
    PackedFragment(T, words::NTuple{W,UInt32}, ownership)

Register representation of logical 8- or 16-bit elements of type `T`. Each
word holds `32 ÷ bits` elements, lowest element first. Ownership counts
logical elements, not words. This constructor interprets bits; `pack(T, f)`
numerically converts values and `unpack(p)` exposes ordinary typed values.
"""
struct PackedFragment{T,W,L}
    data::NTuple{W,UInt32}
    ownership::L
    function PackedFragment(::Type{T}, data::NTuple{W,UInt32}, ownership::L) where {T,W,L}
        _element_bits(T) in (8,16) || throw(ArgumentError("packing supports 8- and 16-bit element types"))
        W > 0 || throw(ArgumentError("a packed fragment must contain values"))
        isbitstype(L) && _register_count(ownership) == W * (32 ÷ _element_bits(T)) ||
            throw(DimensionMismatch("packed payload differs from ownership"))
        new{T,W,L}(data, ownership)
    end
end
_per_word(::Type{T}) where T = 32 ÷ _element_bits(T)
Layouts.layout(f::PackedFragment) = f.ownership
Base.eltype(::Type{<:PackedFragment{T}}) where T = T
Base.eltype(f::PackedFragment) = eltype(typeof(f))

"""
    pack(T, fragment)
    pack(fragment)

Convert logical values to `T` and pack adjacent local slots into register words.
The one-argument form preserves the element type and all bits. Neither form
moves data between threads or changes logical ownership. Supported element
types are 8- and 16-bit: `BFloat16`, `Float16`, `Float8E4M3`, `Float8E5M2`,
`Int8` and `UInt8`. A fragment must hold complete words.
"""
@inline function pack(::Type{T}, f::Fragment) where T
    pack(map(T, f))
end
@generated function pack(f::Fragment{T,N}) where {T,N}
    _element_bits(T) in (8,16) || return :(throw(ArgumentError("packing supports 8- and 16-bit element types")))
    per = 32 ÷ _element_bits(T)
    N % per == 0 || return :(throw(ArgumentError("packing requires complete words")))
    U = _carrier(T)
    words = [foldl((acc,j) -> :($acc | (UInt32(reinterpret($U, f.data[$(per*(i-1)+j)])) << $(_element_bits(T)*(j-1)))),
                   2:per; init=:(UInt32(reinterpret($U, f.data[$(per*(i-1)+1)])))) for i in 1:N÷per]
    quote
        Base.@inline
        PackedFragment(T, ($(words...),), Layouts.layout(f))
    end
end

"Expose the logical typed values of a packed fragment without numerical conversion."
@generated function unpack(f::PackedFragment{T,W}) where {T,W}
    per = 32 ÷ _element_bits(T)
    U = _carrier(T)
    values = [:(reinterpret(T, (f.data[$(i÷per+1)] >> $(_element_bits(T)*(i%per))) % $U)) for i in 0:per*W-1]
    quote
        Base.@inline
        Fragment(($(values...),), Layouts.layout(f))
    end
end

@inline function Base.map(op::F, f::Fragment{T,N}) where {F,T,N}
    Fragment(@rtuple(i -> op(f.data[i]), 1:N), Layouts.layout(f))
end

"Multiply every register value by a scalar, preserving ownership."
@inline scale(f::Fragment, a) = map(Base.Fix2(*, a), f)
