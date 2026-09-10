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
_register_count(::Layouts.LaneRows{N}) where N = N
_register_count(l::Layouts.Ownership) = Int(size(l.mapping)[2])
Layouts.layout(f::Fragment) = f.ownership

"""
    RowFragment(values::NTuple{N,T})

Convenience constructor for `Fragment(values, Layouts.LaneRows{N}())`:
each lane owns a complete logical row. This is one ownership arrangement,
not the general fragment abstraction or a memory contiguity guarantee.
"""
const RowFragment{T,N} = Fragment{T,N,Layouts.LaneRows{N}}
RowFragment(data::NTuple{N,T}) where {N,T} = Fragment(data,Layouts.LaneRows{N}())

"""
    PackedBF16(words::NTuple{W,UInt32})

A packed fragment of 2W BF16 elements, two per word, low element first.
The two-argument constructor accepts explicit ownership; the one-argument
convenience constructor uses `Layouts.LaneRows`. Packing preserves logical
coordinates and local slot order.
"""
struct PackedBF16{W,L}
    data::NTuple{W,UInt32}
    ownership::L
    function PackedBF16(data::NTuple{W,UInt32},ownership::L) where {W,L}
        W > 0 || throw(ArgumentError("a packed fragment must contain values"))
        isbitstype(L) && _register_count(ownership) == 2W ||
            throw(DimensionMismatch("packed payload differs from ownership"))
        new{W,L}(data,ownership)
    end
end
PackedBF16(data::NTuple{W,UInt32}) where W = PackedBF16(data,Layouts.LaneRows{2W}())
Layouts.layout(f::PackedBF16) = f.ownership

"""
    columns(fragment, Val(first), Val(width))

Select consecutive logical columns at a zero-based, compile-time offset.
Use `window` for axis-explicit register or memory partitions.
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
@inline function columns(f::PackedBF16{N},::Val{F},::Val{W}) where {N,F,W}
    _register_window_axis(typeof(Layouts.layout(f))) == 2 ||
        throw(ArgumentError("use window with explicit logical axes"))
    iseven(F) && iseven(W) || error("packed BF16 slices require even boundaries")
    window(f,Val((0,F)),Val((32,W)))
end

# Explicit scalar calls make the indexing independent of LLVM loop unrolling.
@generated function Base.map(op::F, f::Fragment{T,N}) where {F,T,N}
    values = [:(op(f.data[$i])) for i in 1:N]
    quote
        Base.@inline
        Fragment(($(values...),),Layouts.layout(f))
    end
end

"Multiply every register value by a scalar, preserving ownership."
@inline scale(f::Fragment, a) = map(Base.Fix2(*, a), f)
