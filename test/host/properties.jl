# Randomized properties: the layout algebra against enumeration, and the
# plans derived from ownerships against a simulation over every thread.
using Tylo.Layouts: Layout, static, compose, concat, agree, is_injective, logical_divide,
    complement, left_inverse, right_inverse, flatten
const TL = Tylo.Layouts
# Each random case compiles its own generated methods; TYLO_PROPERTY_SCALE
# multiplies the number of cases for a longer search.
const SCALE = parse(Int,get(ENV,"TYLO_PROPERTY_SCALE","1"))

# A random static layout: `n` leaves of small extents with strides that may
# overlap, skip or broadcast.
function random_layout(rng; leaves=rand(rng,1:4))
    s = Tuple(static(rand(rng,(1,2,3,4,8))) for _ in 1:leaves)
    d = Tuple(static(rand(rng,(0,1,2,3,4,8,16,32))) for _ in 1:leaves)
    Layout(s,d)
end
# A random bijection of `0:n-1` for power-of-two `n`, as leaves `(extent, stride)`.
function random_bijection(rng,n)
    extents = Int[]
    while prod(extents; init=1) < n
        push!(extents,min(rand(rng,(2,2,4,8)),n ÷ prod(extents; init=1)))
    end
    strides = cumprod([1; extents[1:end-1]])
    order = randperm(rng,length(extents))
    collect(zip(extents[order],strides[order]))
end
# A random ownership of a power-of-two tile: its coordinate digits split
# between threads and slots in random order, with optional replication.
function random_ownership(rng)
    m, n = 2^rand(rng,0:3), 2^rand(rng,0:4)
    threads, slots = Tuple{Int,Int}[], Tuple{Int,Int}[]
    for leaf in random_bijection(rng,m*n)
        push!(rand(rng,Bool) ? threads : slots,leaf)
    end
    rand(rng) < 0.3 && insert!(threads,rand(rng,1:length(threads)+1),(rand(rng,(2,4)),0))
    rand(rng) < 0.15 && insert!(slots,rand(rng,1:length(slots)+1),(2,0))
    while prod(first.(threads); init=1) > 256
        popfirst!(threads)
    end
    isempty(threads) && push!(threads,(1,0))
    isempty(slots) && push!(slots,(1,0))
    mode(leaves) = (Tuple(static(s) for (s,_) in leaves),Tuple(static(d) for (_,d) in leaves))
    (ts,td), (vs,vd) = mode(threads), mode(slots)
    Ownership((m,n),Layout((ts,vs),(td,vd)))
end

@testset "algebra against enumeration" begin
    rng = MersenneTwister(1234)
    for _ in 1:40SCALE
        a, b = random_layout(rng), random_layout(rng)
        @test agree(coalesce(a),a)
        @test agree(flatten(a),a)
        c = try compose(a,b) catch e; e isa ArgumentError || rethrow(); nothing end
        # Compared inside `a`'s domain; past it the last mode of `a` extends.
        inside = all(Int(b(i)) < Int(length(a)) for i in 0:Int(length(b))-1)
        c === nothing || !inside || @test agree(c,TL.Composition(a,b))
        if is_injective(a)
            # Not every injective layout has a layout for an inverse.
            li = try left_inverse(a) catch e; e isa ArgumentError || rethrow(); nothing end
            li === nothing || @test all(Int(li(a(i))) == i for i in 0:Int(length(a))-1)
            k = try complement(a,static(4*Int(TL.cosize(a)))) catch e; e isa ArgumentError || rethrow(); nothing end
            if k !== nothing
                @test is_injective(concat(a,k))
                @test isdisjoint(Set(Int(a(i)) for i in 1:Int(length(a))-1),Set(Int(k(i)) for i in 1:Int(length(k))-1))
            end
        end
        r = right_inverse(a)
        @test all(Int(a(r(i))) == i for i in 0:Int(length(r))-1)
        # A contiguous tile dividing the layout only renames its coordinates.
        for t in (2,4)
            Int(length(a)) % t == 0 || continue
            d = try logical_divide(a,static(t)) catch e; e isa ArgumentError || rethrow(); nothing end
            d === nothing || @test agree(d,a)
        end
    end
end

# Simulate a reduction plan over every thread: each thread's result must
# combine every logical value sharing its kept coordinate exactly once, and
# the result ownership must place it at that coordinate.
function simulate_reduction(o,axis,plan)
    table = Tylo._mapped_table(o)
    kept(c) = axis == 2 ? (c[1],0) : (0,c[2])
    universe = unique(vec(table))
    for t in axes(table,1), (g,group) in enumerate(plan.groups)
        k = kept(table[t,group[1]])
        partners = Set([t-1])
        for b in plan.bits
            union!(partners,[p ⊻ (1 << b) for p in partners])
        end
        combined = [table[p+1,e] for p in partners for e in group]
        sort(combined) == sort([c for c in universe if kept(c) == k]) || return false
        Tylo._mapped_table(plan.result)[t,g] == k || return false
    end
    true
end

@testset "reductions derived from random ownerships" begin
    rng = MersenneTwister(99)
    derived = 0
    for _ in 1:100SCALE
        o = random_ownership(rng)
        for axis in (1,2)
            plan = Tylo.reduction_plan(o,axis)
            plan === nothing && continue
            derived += 1
            @test simulate_reduction(o,axis,plan)
        end
    end
    @test derived > 50SCALE
    # Replicated values are never combined with their copies.
    everywhere = Ownership((1,8);threads=32)
    @test isempty(Tylo.reduction_plan(everywhere,2).bits)
    @test only(sum(fill(1f0,everywhere);dims=2)) == 8f0
    @test Tylo.reduction_plan(Ownership((4,8);threads=128),1) !== nothing
end
