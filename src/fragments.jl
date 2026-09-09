"""
    RowFragment(values::NTuple{N,T})

This thread's N values in a tile distributed as `Layouts.LaneRows{N}`.
The payload is immutable. Column slices take zero-based static offsets;
dynamic register indexing is deliberately absent from the interface.
"""
struct RowFragment{T,N}
    data::NTuple{N,T}
    function RowFragment(data::NTuple{N,T}) where {N,T}
        Layouts.LaneRows{N}()
        isbitstype(T) || throw(ArgumentError("register values must be isbits"))
        new{T,N}(data)
    end
end

"""
    PackedBF16(words::NTuple{W,UInt32})

A row fragment of 2W BF16 elements, two per word, low element first.
Packing changes representation, while preserving lane/row ownership.
"""
struct PackedBF16{W}
    data::NTuple{W,UInt32}
    function PackedBF16(data::NTuple{W,UInt32}) where W
        W > 0 || throw(ArgumentError("a packed fragment must contain values"))
        new{W}(data)
    end
end

Layouts.layout(::RowFragment{T,N}) where {T,N} = Layouts.LaneRows{N}()
Layouts.layout(::PackedBF16{W}) where W = Layouts.LaneRows{2W}()

"""
    columns(fragment_or_tile, Val(first), Val(width))

Select consecutive logical columns at a zero-based, compile-time offset.
TMEM views also accept a runtime UInt32 offset; their width remains static.
Packed representations require boundaries between complete packed words.
"""
@generated function columns(f::RowFragment{T,N}, ::Val{F}, ::Val{W}) where {T,N,F,W}
    Layouts.check_columns(N,F,W)
    values = [:(f.data[$i]) for i in F+1:F+W]
    quote
        Base.@inline
        RowFragment(($(values...),))
    end
end
@generated function columns(f::PackedBF16{N}, ::Val{F}, ::Val{W}) where {N,F,W}
    Layouts.check_columns(2N,F,W)
    iseven(F) && iseven(W) || error("packed BF16 slices require even boundaries")
    values = [:(f.data[$i]) for i in F÷2+1:(F+W)÷2]
    quote
        Base.@inline
        PackedBF16(($(values...),))
    end
end

# Explicit scalar calls make the indexing independent of LLVM loop unrolling.
@generated function Base.map(op::F, f::RowFragment{T,N}) where {F,T,N}
    values = [:(op(f.data[$i])) for i in 1:N]
    quote
        Base.@inline
        RowFragment(($(values...),))
    end
end

"Multiply every register value by a scalar, preserving the row distribution."
@inline scale(f::RowFragment, a) = map(Base.Fix2(*, a), f)
