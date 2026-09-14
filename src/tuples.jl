# Specialize on the whole range so every callback receives a literal index,
# including when the bounds come from a caller's type parameters.
@generated function _rtuple(f::F, ::Val{R}) where {F,R}
    R isa AbstractRange{<:Integer} ||
        return :(throw(ArgumentError("@rtuple expects an integer range")))
    values = [:(Base.@inline f($i)) for i in R]
    quote
        Base.@inline
        ($(values...),)
    end
end

"""
    @rtuple(f, range)
    @rtuple(range) do i
        ...
    end

Internal tuple construction for small, statically known integer ranges. Call
`f` once per index, in range order, with inlining requested at each call site.
Indices keep their integer type; empty ranges return `()`.

The callable and range expressions are evaluated once in the caller's scope.
Range values become type parameters: literals and type-derived bounds work,
but wrapping an unknown runtime range does not make it known to inference.
This helper does not guarantee register residency.
"""
macro rtuple(f, range)
    :(_rtuple($(esc(f)), Val($(esc(range)))))
end
