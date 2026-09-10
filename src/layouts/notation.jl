# Rewrite tuple syntax, but leave ordinary Julia expressions opaque. In
# particular, quotes and nested macros own any dollar syntax inside them.
function _layout_expr(x)
    if x isa Expr
        if x.head === :$
            value = only(x.args)
            value isa Expr && value.head === :... &&
                throw(ArgumentError("@Layout does not support tuple splatting"))
            return esc(value)
        elseif x.head === :tuple
            return Expr(:tuple, map(_layout_expr, x.args)...)
        elseif x.head === :...
            throw(ArgumentError("@Layout does not support tuple splatting"))
        end
    end
    :(static($(esc(x))))
end

raw"""
    @Layout(shape, strides)
    @Layout shape strides

Construct a `Layout` with static shape and stride leaves by default.
Tuple syntax is traversed recursively. Other unmarked expressions become
`static(expr)`, which also converts tuple values recursively. `$expr` or
`$(expr)` preserves the supplied value and type, including whole subtrees.

```julia
using Tylo.Layouts: @Layout
@Layout (64, 64) (64, 1)          # wholly static
@Layout ((8, 4), $n) ((1, 8), $ld) # runtime n/ld if supplied as ordinary integers
@Layout ($subshape, 64) ($substrides, 1) # preserve mixed tuple values
```

Expressions are evaluated once per occurrence in the caller's scope, when
execution reaches the layout construction. Requesting `static(expr)` does
not make a runtime value known to inference. Interpolation preserves an
already-static value; it does not force conversion to an ordinary integer.
Shape and stride trees must match, as for [`Layout`](@ref). Tuple splatting
and inferred strides are not supported. Interpolate a whole expression as
`$(2*n)`, rather than placing `$` inside arithmetic.

An enclosing Julia quote consumes ordinary dollar interpolation first.
When generating a macro call, insert `Expr(:$, :n)` to preserve a marker
for this macro, or emit the explicit `Layout` constructor instead.
"""
macro Layout(shape, strides)
    :(Layout($(_layout_expr(shape)), $(_layout_expr(strides))))
end
