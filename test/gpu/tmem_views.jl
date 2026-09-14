# TEST_TARGET: cc>=8.0
# Exercise TMEM address calculations and transfer-owned register arithmetic
# on any supported GPU, without executing a TMEM instruction.
using Tylo.Layouts: @Layout, coordinate
function tmem_view_kernel!(addresses,bits,part_addresses,out,input,base,offset,::Val{P},::Val{H}) where {P,H}
    tid=UInt32(threadIdx().x)-UInt32(1)
    lane,warp=tid & UInt32(31),tid >> UInt32(5)
    storage=H ? @Layout(((4,32),(8,32)),((1,4),(128,1024))) : @Layout((128,256),(1,128))
    tile=TmemTile(Float32,base,storage)
    view=@inbounds window(tile,(UInt32(32)*warp,offset),Val((32,64)))
    view=P ? permutedims(view) : view
    plan=P ? TmemTransfer{(64,32),1}() : TmemTransfer{(32,64),2}()
    part=@inbounds partition(plan,view)
    axis=P ? 1 : 2
    bf=@inbounds reinterpret_tile(BFloat16,view;dims=axis)
    bfplan=P ? TmemTransfer{(128,32),1}() : TmemTransfer{(32,128),2}()
    bfpart=@inbounds partition(bfplan,bf)
    @inbounds part_addresses[2Int(tid)+1]=part.address
    @inbounds part_addresses[2Int(tid)+2]=bfpart.address
    PTX.Utils.@unroll for e in 0:63
        c=coordinate(plan,lane,Val(e))
        loc=@inbounds Tylo.tmem_location(view,c)
        packed_c=P ? (UInt32(2*e+1),lane) : (lane,UInt32(2*e+1))
        packed_loc=@inbounds Tylo.tmem_location(bf,packed_c)
        @inbounds addresses[128Int(tid)+2*e+1]=loc.address
        @inbounds addresses[128Int(tid)+2*e+2]=packed_loc.address
        @inbounds bits[64Int(tid)+e+1]=packed_loc.bit_offset
    end
    f=Fragment(ntuple(i -> @inbounds(input[64Int(tid)+i]),Val(64)),plan)
    shifted=f .- maximum(f;dims=axis)
    tail=window(shifted,Val(P ? (32,0) : (0,32)),Val((32,32)))
    store!(pointer(out)+64Int(tid),pack(BFloat16,tail))
    nothing
end

@testset "TMEM logical views and ownership on GB10" begin
    tt=Tuple{CuDeviceVector{UInt32,1},CuDeviceVector{UInt32,1},CuDeviceVector{UInt32,1},
             CuDeviceVector{UInt16,1},CuDeviceVector{Float32,1},UInt32,UInt32}
    for p in (false,true),h in (false,true)
        code=compile_kernel(tmem_view_kernel!,Tuple{tt.parameters...,Val{p},Val{h}};
                            arch=CUDACore.SMVersion(12,1,:arch))
        save_code("tmem-views-permuted$p-nested$h",code)
        @test !occursin(r"\bcall",entry_body(code.ptx))
        @test !occursin(".local .",entry_body(code.ptx))
        @test !occursin("tcgen05.",code.ptx)
        @test occursin("cvt.rn.bf16x2.f32",code.ptx)
    end
    if runtime_supported(@__FILE__)
        values=randn(MersenneTwister(194),Float32,64,128)
        input=CuArray(vec(values))
        addresses=CuArray{UInt32}(undef,128*128)
        bits=CuArray{UInt32}(undef,64*128)
        partitions=CuArray{UInt32}(undef,2*128)
        output=CuArray{UInt16}(undef,32*128)
        expected_values=vec(reinterpret.(UInt16,BFloat16.((values .- maximum(values;dims=1))[33:64,:])))
        for p in (false,true),h in (false,true),offset in UInt32.((0,1,63,96))
            base=UInt32(16)
            @cuda threads=128 tmem_view_kernel!(addresses,bits,partitions,output,input,base,offset,Val(p),Val(h))
            # Independent ISA encoding oracle: physical lane is the thread's
            # warpgroup index, and every FP32 value or BF16 pair occupies a word.
            expected=UInt32[base+offset+e+(t<<16) for _ in 1:2,e in 0:63,t in 0:127]
            @test Array(addresses) == vec(expected)
            @test all(==(UInt32(16)),Array(bits))
            expected_parts=UInt32[base+offset+((t÷32*32)<<16) for _ in 1:2,t in 0:127]
            @test Array(partitions) == vec(expected_parts)
            @test Array(output) == expected_values
        end
    else
        @test_skip false
    end
end
