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

Register representation of logical elements of type `T`, currently `BFloat16`
or `Float16`. Each word holds two elements, low element first. Ownership counts
logical elements, not words. This constructor interprets bits; `pack(T, f)`
numerically converts values and `unpack(p)` exposes ordinary typed values.
"""
struct PackedFragment{T,W,L}
    data::NTuple{W,UInt32}
    ownership::L
    function PackedFragment(::Type{T}, data::NTuple{W,UInt32}, ownership::L) where {T,W,L}
        (T === BFloat16 || T === Float16) || throw(ArgumentError("packing supports BFloat16 and Float16"))
        W > 0 || throw(ArgumentError("a packed fragment must contain values"))
        isbitstype(L) && _register_count(ownership) == 2W ||
            throw(DimensionMismatch("packed payload differs from ownership"))
        new{T,W,L}(data, ownership)
    end
end
Layouts.layout(f::PackedFragment) = f.ownership
Base.eltype(::Type{<:PackedFragment{T}}) where T = T
Base.eltype(f::PackedFragment) = eltype(typeof(f))

"""
    pack(T, fragment)
    pack(fragment)

Convert logical values to `T` and pack adjacent local slots into register words.
The one-argument form preserves the element type and all bits. Neither form
moves data between threads or changes logical ownership. Supported types are
`BFloat16` and `Float16`. A fragment must hold complete pairs.
"""
@inline function pack(::Type{T}, f::Fragment) where T
    pack(map(T, f))
end
@inline function pack(f::Fragment{T,N}) where {T,N}
    (T === BFloat16 || T === Float16) || throw(ArgumentError("packing supports BFloat16 and Float16"))
    iseven(N) || throw(ArgumentError("packing requires complete pairs"))
    words = @rtuple(1:N÷2) do i
        UInt32(reinterpret(UInt16, f.data[2i-1])) | (UInt32(reinterpret(UInt16, f.data[2i])) << 16)
    end
    PackedFragment(T, words, Layouts.layout(f))
end

"Expose the logical typed values of a packed fragment without numerical conversion."
@inline function unpack(f::PackedFragment{T,W}) where {T,W}
    values = @rtuple(0:2W-1) do i
        reinterpret(T, (f.data[i÷2+1] >> (16*(i%2))) % UInt16)
    end
    Fragment(values, Layouts.layout(f))
end

@inline function Base.map(op::F, f::Fragment{T,N}) where {F,T,N}
    Fragment(@rtuple(i -> op(f.data[i]), 1:N), Layouts.layout(f))
end

"Multiply every register value by a scalar, preserving ownership."
@inline scale(f::Fragment, a) = map(Base.Fix2(*, a), f)
