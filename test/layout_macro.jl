using Tylo.Layouts: @Layout, Layout, shape, static

# Importing only the macro must suffice; caller bindings must not capture
# the Layout constructor or static converter emitted by the macro.
module LayoutMacroCaller
using Tylo.Layouts: @Layout
const Layout = nothing
const static = nothing
function make(n, ld)
    @Layout (64, $n) ($ld, 1)
end
function make_static(::Val{K}) where K
    @Layout (64, K) (K, 1)
end
end

# A nested macro consumes its own dollar expression, not @Layout's.
macro layout_test_identity(x)
    @assert x isa Expr && x.head === :$
    esc(only(x.args))
end

@testset "Static layout notation" begin
    @test (@Layout (64, 32) (32, 1)) ===
        Layout((static(64), static(32)), (static(32), static(1)))
    @test (@Layout((64, 32), (32, 1))) === (@Layout (64, 32) (32, 1))
    @test (@Layout (8,) (1,)) === Layout((static(8),), (static(1),))
    @test (@Layout (8, 4) (1, 0)) === Layout((static(8), static(4)), (static(1), static(0)))

    n, ld = Int32(5), Int32(128)
    mixed = @inferred LayoutMacroCaller.make(n, ld)
    @test shape(mixed) === (static(64), n)
    @test strides(mixed) === (ld, static(1))
    @test typeof(LayoutMacroCaller.make(Int32(7), Int32(256))) === typeof(mixed)
    @test (@inferred LayoutMacroCaller.make_static(Val(32))) === (@Layout (64, 32) (32, 1))
    @test (@inferred LayoutMacroCaller.make(static(32), static(32))) === (@Layout (64, 32) (32, 1))

    nested = @Layout ((8, 4), $n) ((1, 8), $ld)
    @test shape(nested) === ((static(8), static(4)), n)
    @test strides(nested) === ((static(1), static(8)), ld)
    @test size(nested) == (32, 5)
    @test nested(((Int32(3), Int32(2)), Int32(1))) === Int32(147)
    @test nested((Int32(19), Int32(1))) === Int32(147)

    subshape, substrides = ((static(2), Int32(4)), Int32(3)), ((static(1), Int32(2)), Int32(8))
    preserved = @Layout ($subshape, 64) ($substrides, 32)
    @test shape(preserved) === (subshape, static(64))
    @test strides(preserved) === (substrides, static(32))
    converted = @Layout (subshape, 64) (substrides, 32)
    @test shape(converted) === (static(subshape), static(64))
    @test strides(converted) === (static(substrides), static(32))
    @test (@Layout $subshape $substrides) === Layout(subshape, substrides)
    @test (@Layout subshape substrides) === Layout(static(subshape), static(substrides))
    @test (@Layout (8, $(2*n)) ($(ld+8), 1)) ===
        Layout((static(8), 2*n), (ld+8, static(1)))

    # Escaped and unmarked expressions both execute exactly once per occurrence.
    calls = Int[]
    record(i, value) = (push!(calls, i); value)
    made = @Layout (record(1, 8), $(record(2, n))) ($(record(3, ld)), record(4, 1))
    @test calls == [1, 2, 3, 4]
    @test made === Layout((static(8), n), (ld, static(1)))
    @test (@Layout ((@layout_test_identity $n), 8) (1, 8)) ===
        Layout((static(n), static(8)), (static(1), static(8)))

    # Preserve a dollar node across a surrounding quote, without looking up
    # the runtime variable at expression-construction time.
    marker = Expr(:$, :layout_runtime_size)
    expr = :(let layout_runtime_size = Int32(7)
        @Layout (8, $marker) (1, 8)
    end)
    @test Core.eval(@__MODULE__, expr) === Layout((static(8), Int32(7)), (static(1), static(8)))

    @test_throws ArgumentError (@Layout ((4, 4), 8) (1, 16))
    @test_throws ArgumentError (@Layout (16, 0) (1, 16))
    @test_throws ArgumentError (@Layout (16, 8) (1, -16))
    @test_throws ArgumentError (@Layout (16, 8) (1, $(Int32(-1))))
    for code in (raw"@Layout (dims..., 8) (1, 8)", raw"@Layout ($(dims...), 8) (1, 8)")
        @test_throws ArgumentError macroexpand(@__MODULE__, Meta.parse(code))
    end
end
