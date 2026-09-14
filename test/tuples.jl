using Tylo: @rtuple

module RtupleCaller
using Tylo: @rtuple
const _rtuple = nothing
const Val = nothing
make(f) = @rtuple(f, 0:3)
end

rtuple_tags(::Val{N}) where N = @rtuple(i -> Val(i), 0:N-1)

@testset "Static range tuples" begin
    @test (@rtuple(i -> i*i, 0:3)) == (0,1,4,9)
    @test (@rtuple(0:3) do i
        i*i
    end) == (0,1,4,9)
    @test (@rtuple(identity, -3:2:3)) == (-3,-1,1,3)
    @test (@rtuple(identity, 4:-2:0)) == (4,2,0)
    @test (@rtuple(identity, Int32(0):Int32(2))) === (Int32(0),Int32(1),Int32(2))
    @test (@rtuple(identity, 2:1)) == ()
    @test (@rtuple(identity, 1:-1:2)) == ()
    @test (@rtuple(identity, 7:7)) == (7,)
    @test (@inferred rtuple_tags(Val(4))) === (Val(0),Val(1),Val(2),Val(3))
    @test RtupleCaller.make(abs) == (0,1,2,3)

    visited = Int[]
    factor = 3
    values = @rtuple(0:3) do i
        push!(visited,i)
        return factor*i
    end
    @test values == (0,3,6,9)
    @test visited == [0,1,2,3]
    @test (@rtuple(2:1) do i
        error("empty range called mapper")
    end) == ()

    events = Symbol[]
    callable() = (push!(events,:callable); identity)
    indices() = (push!(events,:range); 0:2)
    @test (@rtuple(callable(), indices())) == (0,1,2)
    @test events == [:callable,:range]
    empty!(visited)
    @test_throws ErrorException @rtuple(0:3) do i
        push!(visited,i)
        i == 1 && error("stop")
        i
    end
    @test visited == [0,1]
    @test_throws ArgumentError (@rtuple(identity, (1,2)))
    @test_throws ArgumentError (@rtuple(identity, 1.0:2.0))
end
