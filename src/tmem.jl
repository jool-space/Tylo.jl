# TMEM uses 32-bit columns, not byte offsets. BF16 packs two logical
# columns into each physical column. This relationship is independent of
# the warp-band address field in bits 31:16.
_column_words(::Type{Float32}, n) = n
_column_shift(::Type{Float32}) = 0
_column_shift(::Type{BFloat16}) = 1
function _column_words(::Type{BFloat16}, n)
    iseven(n) || throw(ArgumentError("TMEM BF16 views require an even width"))
    n ÷ 2
end

"""
    TmemTile{T,N}(address::UInt32)

Borrow a 128-row TMEM tile of N logical columns. T is Float32 or BFloat16.
The address must name lane zero and an allocation large enough for the view.
This is an unsafe borrowed view: it neither allocates nor owns TMEM, and
does not establish readiness or permission to overwrite aliased storage.
"""
struct TmemTile{T,N}
    address::UInt32
    function TmemTile{T,N}(address::UInt32) where {T,N}
        N isa Int && N > 0 || throw(ArgumentError("positive static width required"))
        _column_words(T,N) <= 512 || throw(ArgumentError("TMEM view exceeds 512 columns"))
        new{T,N}(address)
    end
end
Base.size(::TmemTile{T,N}) where {T,N} = (128,N)

"""
    TmemRows{T,N}(address::UInt32)

Borrow one warp's 32-row band of a TMEM tile. The address includes the
warp-band bits; all lanes in the warp pass the same address.
Prefer `warp_rows(tile, warp_in_group)` when starting from an allocation.
"""
struct TmemRows{T,N}
    address::UInt32
    function TmemRows{T,N}(address::UInt32) where {T,N}
        N isa Int && N > 0 || throw(ArgumentError("positive static width required"))
        _column_words(T,N) <= 512 || throw(ArgumentError("TMEM view exceeds 512 columns"))
        new{T,N}(address)
    end
end
Base.size(::TmemRows{T,N}) where {T,N} = (32,N)

"""
    warp_rows(tile, warp_in_group::UInt32)

Select warp band 0, 1, 2, or 3. Caller must pass a warp-uniform value in 0:3.
The lane's own row within this band is selected by the TMEM instruction.
"""
@inline warp_rows(t::TmemTile{T,N}, warp::UInt32) where {T,N} =
    TmemRows{T,N}(t.address + (warp << UInt32(21)))

for View in (:TmemTile, :TmemRows)
    @eval begin
        @inline function columns(t::$View{T,N}, ::Val{F}, ::Val{W}) where {T,N,F,W}
            Layouts.check_columns(N,F,W)
            off = _column_words(T,F)
            $View{T,W}(t.address + UInt32(off))
        end

        Base.@propagate_inbounds function columns(t::$View{T,N}, first::UInt32,
                                                   ::Val{W}) where {T,N,W}
            Layouts.check_columns(N,0,W)
            @boundscheck begin
                first <= UInt32(N-W) || throw(BoundsError(t,first))
                first & UInt32((1 << _column_shift(T))-1) == 0 ||
                    throw(ArgumentError("packed BF16 slices require even boundaries"))
            end
            $View{T,W}(t.address + (first >> _column_shift(T)))
        end

        """
            reinterpret_tile(T, tile)

        View the same physical TMEM footprint with a different element format.
        This does not convert values or synchronize access. For example an FP32
        score tile becomes a BF16 view with twice as many logical columns.
        """
        @inline function reinterpret_tile(::Type{BFloat16}, t::$View{Float32,N}) where N
            $View{BFloat16,2N}(t.address)
        end
        @inline function reinterpret_tile(::Type{Float32}, t::$View{BFloat16,N}) where N
            $View{Float32,N÷2}(t.address)
        end
    end
end

# No fragment interface: using pending registers requires wait_load.
struct PendingLoad{N}
    words::NTuple{N,UInt32}
end
