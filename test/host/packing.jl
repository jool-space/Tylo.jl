@testset "Logical elements and register words" begin
    # Exhaust all 16-bit payloads, including NaN payloads and signed zero.
    for T in (BFloat16,Float16), axis in (1,2)
        for first in 0:32:65535
            bits = ntuple(i -> UInt16(first+i-1),32)
            ownership = lane_local(32,axis)
            f = Fragment(reinterpret.(T,bits),ownership)
            words = @inferred Tylo._pack_words(f)
            @test words == ntuple(i -> UInt32(bits[2i-1]) | UInt32(bits[2i]) << 16,16)
            back = @inferred Tylo._from_words(T,words,ownership)
            @test reinterpret.(UInt16,back.data) == bits
            @test Tylo.Layouts.layout(back) === ownership
        end
        values = (0f0,-0f0,1.00390625f0,1.01171875f0,65504f0,Inf32,-Inf32,1f-40)
        f = Fragment(values,lane_local(8,axis))
        converted = @inferred Fragment{T}(f)
        @test eltype(converted) === T
        @test reinterpret.(UInt16,converted.data) == reinterpret.(UInt16,T.(values))
        @test Fragment{T}(converted) === converted
        @test permutedims(converted).data === converted.data
        @test Tylo.Layouts.coordinate(Tylo.Layouts.layout(permutedims(converted)),7,3) ==
            reverse(Tylo.Layouts.coordinate(Tylo.Layouts.layout(f),7,3))
        origin, shape = axis == 2 ? ((0,2),(32,4)) : ((2,0),(4,32))
        @test window(converted,origin,shape).data === T.(values[3:6])
    end
    @test_throws ArgumentError Tylo._pack_words(Fragment((BFloat16(1),),lane_local(1,2)))
    @test_throws DimensionMismatch Fragment((Float16(0),),lane_local(2,2))
end

@testset "TMEM element types and instruction widths" begin
    for T in (Float32,BFloat16,Float16), words in (1,2,4,8,16,32,64,128), axis in (1,2)
        n = words * (4÷sizeof(T))
        shape = axis == 2 ? (32,n) : (n,32)
        strides = axis == 2 ? (1,128) : (128,1)
        tile = TmemTile(T,UInt32(0),Tylo.Layouts.Layout(shape,strides))
        part = @inferred Tylo._partition(transfer(shape,axis),tile)
        @test eltype(part) === T
        @test size(part) == shape
        @test part.address === UInt32(0)
        for U in (Float32,BFloat16,Float16)
            view = reinterpret_tile(U,tile;dims=axis)
            @test prod(size(view))*sizeof(U) == prod(shape)*sizeof(T)
            @test reinterpret_tile(T,view;dims=axis).address == tile.address
        end
    end
end

@testset "Equal-width reinterpretation preserves subword origins" begin
    tile = TmemTile(BFloat16,UInt32(0),Tylo.Layouts.Layout((32,8),(1,128)))
    odd = window(tile,(0,1),(32,3))
    half = reinterpret_tile(Float16,odd;dims=2)
    @test Tylo.Layouts.layout(half) === Tylo.Layouts.layout(odd)
    @test Tylo.tmem_location(half,(0,0)) == Tylo.tmem_location(odd,(0,0))
    @test_throws ArgumentError reinterpret_tile(Float32,odd;dims=2)
end
