using cuTile: TileArray, Tile

const TileVector{T} = TileArray{T,1}
const TileMatrix{T} = TileArray{T,2}
const TileArray3{T} = TileArray{T,3}
const TileArray4{T} = TileArray{T,4}
const TileArray5{T} = TileArray{T,5}

const Optional{T} = Union{T,Nothing}

x → T::Type = T.(x)

struct TransposePostfix end

const ᵀ = TransposePostfix()

Base.:(*)(x, ::TransposePostfix) = transpose(x)

# Per-dim element index tiles for a tile of `shape` placed at 1-based tile
# coordinates `index` — the `ct.store` addressing convention spelled out as
# index tiles (each broadcast along its own dim). The tile spans the leading
# `length(shape)` array dims; trailing coords are scalar element positions.
@inline function element_indices(shape::Tuple, index::NTuple{N, Integer}) where {N}
    ntuple(Val(N)) do d
        ext = d <= length(shape) ? shape[d] : 1
        v = (Int32(index[d]) - 1i32) * Int32(ext) .+ ct.arange(ext)
        d == 1 ? v : reshape(v, ntuple(k -> k == d ? ext : 1, Val(d)))
    end
end

# `ct.store`-shaped atomic accumulation, built on cuTile's tile-indexed
# `atomic_add`, which bounds-masks out-of-range elements just like `ct.store`
# clips partial tiles.
@inline function atomic_add_tile(A::TileArray{T}, index::Tuple, tile::Tile{T};
                                 memory_order=ct.MemoryOrder.Relaxed) where {T}
    ct.atomic_add(A, element_indices(size(tile), index), tile; memory_order)
    return
end

arithmetic_type(T::Type) = T
arithmetic_type(::Type{TFloat32}) = Float32

tensorcore_type(T::Type) = T
tensorcore_type(::Type{Float32}) = TFloat32
