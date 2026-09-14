using Tylo.Layouts: @Layout, Layout, coordinate

@testset "TMEM storage, logical views and transfer ownership" begin
    for width in (32,64,128),base in UInt32.((0,32)),warp in 0:3
        tile=TmemTile(Float32,base,@Layout((128,width),(1,128)))
        plan=TmemTransfer{(32,16),2}()
        @test eltype(tile) === Float32
        for offset in 0:16:width-16
            view=window(tile,(32warp,offset),Val((32,16)))
            part=@inferred partition(plan,view)
            @test part.address == base+UInt32(32warp*65536+offset)
            @test size(part) == (32,16)
            @test size(permutedims(part)) == (16,32)
            @test (@inferred partition(permutedims(plan),permutedims(view))).address == part.address
            @test permutedims(permutedims(plan)) === plan
            for thread in (0,7,31),e in (0,3,15)
                c=coordinate(plan,thread,Val(e))
                @test coordinate(permutedims(plan),thread,Val(e)) == reverse(c)
                loc=Tylo.tmem_location(view,c)
                @test loc.address == base+UInt32((32warp+thread)*65536+offset+e)
                @test loc.bit_offset == 0
                @test Tylo.tmem_location(permutedims(view),reverse(c)) == loc
            end
            @test_throws DimensionMismatch partition(plan,permutedims(view))
        end
    end

    # Runtime offsets retain the allocation-relative origin through nested views.
    tile=TmemTile(Float32,UInt32(16),@Layout((128,256),(1,128)))
    for offset in UInt32(0):UInt32(64)
        a=window(window(tile,(UInt32(64),UInt32(8)),Val((32,192))),
                 (UInt32(0),offset),Val((32,64)))
        @test partition(TmemTransfer{(32,64),2}(),a).address == UInt32(64*65536+24)+offset
    end
    nested=TmemTile(Float32,UInt32(0),@Layout(((4,32),(8,16)),((1,4),(128,1024))))
    @test partition(TmemTransfer{(32,64),2}(),window(nested,(64,32),Val((32,64)))).address == UInt32(64*65536+32)
    badnested=TmemTile(Float32,UInt32(0),@Layout(((4,8),16),((1,8),128)))
    @test_throws ArgumentError partition(TmemTransfer{(32,16),2}(),badnested)
end

# Kernel callers specialize the logical axis; runtime storage extents remain runtime.
as_bf16_axis1(t) = reinterpret_tile(BFloat16,t;dims=1)
as_bf16_axis2(t) = reinterpret_tile(BFloat16,t;dims=2)

@testset "TMEM representation follows the chosen logical axis" begin
    tile=TmemTile(Float32,UInt32(8),@Layout((128,256),(1,128)))
    view=window(tile,(64,32),Val((32,64)))
    for permute in (false,true)
        src=permute ? permutedims(view) : view
        axis=permute ? 1 : 2
        bf=permute ? (@inferred as_bf16_axis1(src)) : (@inferred as_bf16_axis2(src))
        @test size(bf) == (permute ? (128,32) : (32,128))
        back=reinterpret_tile(Float32,bf;dims=axis)
        @test size(back) == size(src)
        for i in (0,13,31),j in (0,1,31,63)
            c=permute ? (j,i) : (i,j)
            @test Tylo.tmem_location(src,c) == Tylo.tmem_location(back,c)
            first=Tylo.tmem_location(bf,permute ? (2j,i) : (i,2j))
            second=Tylo.tmem_location(bf,permute ? (2j+1,i) : (i,2j+1))
            @test first == Tylo.tmem_location(src,c)
            @test second.address == first.address
            @test second.bit_offset == 16
        end
        plan=permute ? TmemTransfer{(128,32),1}() : TmemTransfer{(32,128),2}()
        @test partition(plan,bf).address == Tylo.tmem_location(src,(0,0)).address
        @test_throws ArgumentError reinterpret_tile(BFloat16,src;dims=3-axis)
        odd=window(bf,permute ? (1,0) : (0,1),Val((32,32)))
        @test_throws ArgumentError reinterpret_tile(Float32,odd;dims=axis)
        @test_throws ArgumentError partition(TmemTransfer{(32,32),axis}(),odd)
    end
end

@testset "TMEM transfer contract failures" begin
    p=TmemTransfer{(32,64),2}()
    tile=TmemTile(Float32,UInt32(0),@Layout((128,512),(1,128)))
    @test_throws BoundsError window(tile,(0,500),Val((32,64)))
    @test_throws BoundsError Tylo.tmem_location(tile,(128,0))
    @test_throws BoundsError Tylo.tmem_location(tile,(0,512))
    @test_throws BoundsError partition(p,TmemTile(Float32,UInt32(480),@Layout((32,64),(1,128))))
    @test_throws BoundsError partition(p,TmemTile(Float32,UInt32(128*65536),@Layout((32,64),(1,128))))
    @test_throws ArgumentError partition(p,window(tile,(1,0),Val((32,64))))
    @test_throws ArgumentError partition(p,TmemTile(Float32,UInt32(0),@Layout((32,64),(1,256))))
    @test_throws ArgumentError partition(p,TmemTile(Float32,UInt32(0),@Layout((32,64),(0,128))))
    @test_throws ArgumentError TmemTransfer{(16,64),2}()
    @test_throws ArgumentError TmemTransfer{(32,64),0}()
    @test_throws BoundsError coordinate(p,0,Val(0.5))
    @test_throws BoundsError coordinate(p,0,Val(64))
    @test_throws ArgumentError partition(TmemTransfer{(32,3),2}(),window(tile,(0,0),Val((32,3))))
    @test_throws ArgumentError TmemTile(Float64,UInt32(0),@Layout((32,64),(1,128)))
    swizzled=TmemTile(Float32,UInt32(0),Tylo.Layouts.compose(Tylo.Layouts.Swizzle{1,0,2}(),@Layout((32,64),(1,128))))
    @test_throws ArgumentError partition(p,swizzled)
    @test_throws ArgumentError permutedims(p,(1,1))
    @test_throws ArgumentError permutedims(tile,(2.0,1.0))
end

@testset "Transfer fragments, packed windows and completion" begin
    plan=TmemTransfer{(32,64),2}()
    data=ntuple(Float32,64)
    f=Fragment(data,plan)
    for permute in (false,true)
        g=permute ? permutedims(f) : f
        axis=permute ? 1 : 2
        origin=permute ? (32,0) : (0,32)
        h=@inferred window(g,Val(origin),Val((32,32)))
        @test h.data == data[33:64]
        @test size(Tylo.Layouts.layout(h)) == (32,32)
        @test only(sum(h;dims=axis)) == sum(data[33:64])
        @test (h .- maximum(h;dims=axis)).data == data[33:64] .- 64f0
        @test_throws ArgumentError window(g,Val(permute ? (0,1) : (1,0)),Val((32,32)))
    end
    words=ntuple(UInt32,32)
    packed=PackedFragment(BFloat16,words,plan)
    @test Tylo.Layouts.layout(packed) === plan
    @test window(packed,Val((0,32)),Val((32,32))).data == words[17:32]
    @test window(permutedims(packed),Val((32,0)),Val((32,32))).data == words[17:32]
    @test permutedims(permutedims(packed)) === packed
    @test_throws ArgumentError window(packed,Val((0,1)),Val((32,32)))
    @test_throws DimensionMismatch PackedFragment(BFloat16,words,TmemTransfer{(32,32),2}())
    part=partition(plan,TmemTile(Float32,UInt32(0),@Layout((32,64),(1,128))))
    @test Tylo._check_tmem_store(part,f) === nothing
    @test Tylo._check_tmem_store(permutedims(part),permutedims(f)) === nothing
    @test permutedims(permutedims(f)) === f
    runtime=TmemTile(Float32,UInt32(0),Layout((32,64),(1,128)))
    @test size(reinterpret_tile(BFloat16,runtime;dims=2)) === (32,128)
    @test Tylo._check_tmem_store(part,local_fragment(data)) === nothing
    @test_throws DimensionMismatch Tylo._check_tmem_store(part,permutedims(f))
    pending=Tylo.PendingLoad(Float32,ntuple(UInt32,64),plan)
    @test_throws MethodError map(abs,pending)
    @test_throws MethodError scale(pending,2f0)
    @test_throws MethodError getindex(pending,1)
    @test_throws MethodError window(f,(0,0),Val((32,32)))
    @test isempty(Test.detect_ambiguities(Tylo;recursive=true))
end
