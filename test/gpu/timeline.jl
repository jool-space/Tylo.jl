# TEST_TARGET: cc>=8.0
# `Timeline`: records of the SM clock, from the device, read back per
# recorder and measured between events.
@testset "Timeline records" begin
    # Two records by hand: time counts from the block's first record.
    records = zeros(UInt64,4,2,1)
    records[1,1,1] = (UInt64(1) << 56) | 1000; records[2,1,1] = (UInt64(2) << 56) | 1500
    records[1,2,1] = (UInt64(1) << 56) | 1200
    t = Timeline(records)
    @test [(e.recorder,e.event,e.time) for e in events(t)] == [(0,1,0),(1,1,200),(0,2,500)]
    @test spans(t,1 => 2) == [500]
    @test isempty(spans(t,1 => 2;recorders=(1,)))
end

# Each warp spins for a time proportional to its index, ten times: its
# spans grow with it, one record in two is dropped past the capacity.
function spinning!(timeline)
    tid = Int32(threadIdx().x)-Int32(1)
    warp = tid ÷ Int32(32)
    mine = recorder(timeline,warp)
    for step in Int32(0):Int32(9)
        tid % Int32(32) == Int32(0) && record!(mine,2step,1)
        start = CUDACore.clock(UInt64)
        while CUDACore.clock(UInt64)-start < UInt64(2000)*UInt64(warp+1) end
        tid % Int32(32) == Int32(0) && record!(mine,2step+Int32(1),2)
    end
    tid == Int32(0) && record!(timeline,4,0,9)                 # no such recorder: dropped
    nothing
end

@testset "Timeline on the device" begin
    timeline = Timeline(CUDACore.zeros(UInt64,16,4,2))
    @cuda threads=128 blocks=2 spinning!(timeline)
    synchronize()
    list = events(timeline)
    @test length(list) == 2*4*16                            # 8 of 10 steps fit
    @test all(e -> e.event == (iseven(e.slot) ? 1 : 2),list)   # and none of recorder 4
    for w in 0:3
        s = spans(timeline,1 => 2;recorders=(w,))
        @test length(s) == 2*8 && all(x -> 2000(w+1) <= x < 2000(w+1)+2000,s)
    end
    code = Tylo.machine_code(spinning!,Tuple{Timeline{CuDeviceArray{UInt64,3,1}}})
    @test isempty(audit(code))
end
