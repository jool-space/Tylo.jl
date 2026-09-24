# Layout algebra against the enumeration oracle: every operation's result
# must agree pointwise with the function it claims to represent, infer to a
# static layout from static inputs, and reject inexact cases.
using Tylo.Layouts: @Layout, Layout, static, StaticInt, shape, cosize, compose, Swizzle
using Tylo.Layouts: coalesce, complement, right_inverse, left_inverse, logical_divide, zipped_divide,
    tiled_divide, logical_product, blocked_product, raked_product, concat, agree, is_injective,
    is_compact, slice, flatten, filter_zeros, mode, trivial
const TL = Tylo.Layouts

L(s, d) = Layout(map(static, s), map(static, d))
pointwise(a, b) = agree(a, b)

@testset "coalesce" begin
    c = @inferred coalesce(@Layout(((4, 2), 1, 3), ((1, 4), 0, 8)))
    @test shape(c) === (static(24),)
    m = coalesce(Layout((static(4), 2, static(3)), (static(1), static(4), 8)))
    @test shape(m) == (8, static(3)) && TL.strides(m) === (static(1), 8)   # static strides decide; a runtime stride blocks merging
    @test shape(@inferred coalesce(L((1, 1), (0, 0)))) === (static(1),)
end

@testset "compose oracle" begin
    cases = [
        (L((4, 4), (1, 4)), L((8,), (2,))),
        (L((16, 32), (32, 1)), L(((8, 2), (16, 2)), ((1, 8), (16, 32)))),
        (L((2, 4, 8), (1, 2, 8)), L((4, 16), (1, 4))),
        (L((64,), (1,)), L((8, 8), (8, 1))),
        (L((8, 8), (8, 1)), L((8, 8), (8, 1))),
        (L(((2, 2), 3), ((1, 2), 4)), L((12,), (1,))),
        (L((4, 4), (1, 8)), L((4,), (2,))),
        (L((6, 2), (2, 1)), L((3, 2), (2, 6))),
    ]
    for (a, b) in cases
        c = @inferred compose(a, b)
        @test pointwise(c, TL.Composition(a, b))
        @test TL._isstatic(c)
    end
    # Inexact compositions throw instead of approximating.
    @test_throws ArgumentError compose(L((4, 4), (1, 8)), L((8,), (3,)))
    @test_throws ArgumentError compose(L((4, 4), (1, 8)), L((3,), (2,)))
    # Mixed leaves keep their staticness.
    n = 24
    c = compose(Layout((static(8), n), (static(1), static(8))), L((8, 3), (1, 8)))
    @test shape(c) === (static(8), static(3)) && TL.strides(c) === (static(1), static(8))
    @test pointwise(c, TL.Composition(Layout((static(8), n), (static(1), static(8))), L((8, 3), (1, 8))))
    g = Layout((static(128), n), (Int32(1), Int32(256)))
    c = @inferred compose(g, L((8, 8), (1, 8)))
    @test TL.strides(c) === (Int32(1), Int32(8)) && shape(c) === (static(8), static(8))
    # Swizzled outer stays outside.
    sw = compose(Swizzle{2,3,2}(), L((32, 32), (32, 1)))
    t = compose(sw, L((8, 8), (1, 32)))
    @test t isa TL.Composition && all(t(i) == sw(L((8, 8), (1, 32))(i)) for i in 0:63)
end

@testset "complement" begin
    for (l, n) in [(L((4,), (1,)), 16), (L((4,), (2,)), 16), (L((2, 2), (1, 4)), 16),
                   (L((8,), (1,)), 24), (L((2, 4), (4, 1)), 32), (L((3,), (1,)), 5)]
        c = @inferred complement(l, static(n))
        both = concat(l, c)
        @test is_injective(both)
        @test isdisjoint(Set(l(i) for i in 0:length(l)-1), Set(c(i) for i in 1:length(c)-1))
        @test cosize(both) >= n
        @test TL._isstatic(c)
    end
    @test complement(L((4,), (2,)), static(16)) == L((2, 2), (1, 8))
    @test complement(L((2, 2), (1, 4)), static(16)) == L((2, 2), (2, 8))
    @test_throws ArgumentError complement(L((2, 2), (1, 1)), static(8))
    @test_throws ArgumentError complement(L((2, 2), (2, 3)), static(16))
    r = complement(L((8,), (1,)), 40)     # runtime cotarget
    @test shape(r) == (5,) && TL.strides(r) === (static(8),)
    z = complement(L((4,), (0,)), static(16))
    @test z == L((16,), (1,))
end

@testset "inverses" begin
    for l in [L((4,), (1,)), L((2, 2), (1, 2)), L((4, 2), (2, 1)), L((2, 2), (1, 4)),
              L((2, 3, 4), (12, 1, 3)), L((8, 8), (8, 1))]
        r = @inferred right_inverse(l)
        @test all(l(r(i)) == i for i in 0:length(r)-1)
        li = @inferred left_inverse(l)
        @test all(li(l(i)) == i for i in 0:length(l)-1)
    end
    @test right_inverse(L((4, 2), (2, 1))) == L((2, 4), (4, 1))
    @test right_inverse(L((2, 2), (1, 4))) == L((2,), (1,))
    @test right_inverse(L((4,), (2,))) == trivial()
    @test left_inverse(L((4,), (2,))) == L((2, 4), (0, 1))
    @test_throws ArgumentError right_inverse(Layout((4,), (1,)))
end

@testset "divide" begin
    l = L((16, 32), (32, 1))
    d = @inferred logical_divide(l, (L((8,), (1,)), L((16,), (1,))))
    @test shape(d) == ((8, 2), (16, 2)) && TL.strides(d) == ((32, 256), (1, 16))
    z = @inferred zipped_divide(l, (static(8), static(16)))
    @test shape(z) == ((8, 16), (2, 2)) && TL.strides(z) == ((32, 1), (256, 16))
    t = @inferred tiled_divide(l, (static(8), static(16)))
    @test shape(t) == ((8, 16), 2, 2)
    @test all(z(((i, j), (bi, bj))) == l((i + 8bi, j + 16bj)) for i in 0:7, j in 0:15, bi in 0:1, bj in 0:1)
    # A strided tile selects every other column.
    s = logical_divide(L((8,), (1,)), L((4,), (2,)))
    @test shape(s) == (4, 2) && TL.strides(s) == (2, 1)
    # Runtime rows keep a static tile mode.
    m = 40
    g = logical_divide(Layout((m, static(64)), (static(1), m)), (static(8), static(16)))
    @test shape(g)[1][1] === static(8) && shape(g)[1][2] == 5
    @test shape(g)[2] === (static(16), static(4))
    # Unaligned tiling is rejected (composition inexact), never padded.
    @test_throws ArgumentError logical_divide(L((16,), (1,)), L((3,), (1,)))
    @test_throws ArgumentError logical_divide(Layout((41,), (static(1),)), static(8))
    # Plain integers are runtime tiles, as everywhere in Layout.
    @test shape(logical_divide(l, (8, 16)))[1][1] == 8 && !TL._isstatic(logical_divide(l, (8, 16)))
    sw = compose(Swizzle{2,3,2}(), L((32, 32), (32, 1)))
    @test zipped_divide(sw, (static(8), static(8))) isa TL.Composition
end

@testset "product" begin
    b = L((2, 2), (1, 2))
    t = L((3, 3), (1, 3))
    p = @inferred blocked_product(b, t)
    @test shape(p) == ((2, 3), (2, 3)) && TL.strides(p) == ((1, 4), (2, 12))
    @test all(p(((i1, j1), (i2, j2))) == b((i1, i2)) + 4t((j1, j2)) for i1 in 0:1, j1 in 0:2, i2 in 0:1, j2 in 0:2)
    r = @inferred raked_product(b, t)
    @test shape(r) == ((3, 2), (3, 2)) && TL.strides(r) == ((4, 1), (12, 2))
    lp = @inferred logical_product(L((4,), (1,)), L((3,), (1,)))
    @test lp == L(((4,), (3,)), ((1,), (4,)))
    # A strided block: the tile fills the gaps of the block first.
    lp = logical_product(L((4,), (2,)), L((4,), (1,)))
    @test lp == L(((4,), ((2, 2),)), ((2,), ((1, 8),)))   # the split mode keeps the tile profile
    @test is_injective(lp)
    @test_throws ArgumentError logical_product(L((4,), (2,)), L((3,), (1,)))
end

@testset "slice, mode, flatten" begin
    l = L(((2, 2), 3), ((1, 2), 4))
    sub, off = @inferred slice(l, ((1, :), 2))
    @test sub == L((2,), (2,)) && off == static(9)
    @test mode(l, 1) == L((2, 2), (1, 2)) && mode(l, 2) == L((3,), (4,))
    @test flatten(l) == L((2, 2, 3), (1, 2, 4))
    @test filter_zeros(L((2, 3), (0, 1))) == L((3,), (1,))
    @test is_compact(L((4, 2), (2, 1))) && !is_compact(L((4,), (2,))) && !is_injective(L((2, 2), (0, 1)))
end

# tile via logical_divide reproduces the existing factorization.
@testset "tile equivalence" begin
    l = L((16, 32), (32, 1))
    old = TL.tile(l, Val((8, 16)))
    new = logical_divide(l, (static(8), static(16)))
    @test shape(old) == shape(new) && TL.strides(old) == TL.strides(new)
    @test pointwise(old, new)
end
