# Shared layouts, rings and configurations on the host: offsets, sizes, slots
# and phases, and that they are constants for the compiler.
using Tylo: BoundLayout, Barriers, Buffers, Stage

@inline gemm_layout(stages,bytes) = SharedLayout((;
    full = barriers(stages;arrivals=1), empty = barriers(stages;arrivals=1),
    ready = barriers(2;arrivals=1), drained = barriers(2;arrivals=4),
    slot = scalars(UInt32),
    stage = buffers(bytes,stages;align=1024)))

@testset "Shared layout" begin
    l = gemm_layout(4,49152)
    offsets, plain, aligned = Tylo._offsets(values(l.groups))
    # Barriers first, 8 bytes each, then the scalar; the stages after them,
    # from the first 1024-byte boundary.
    @test offsets == (0,32,64,80,96,0)
    @test plain == 100 && aligned == 4*49152
    @test shared_bytes(l) == 112+1024+4*49152
    # Without aligned groups, no slack.
    @test shared_bytes(SharedLayout((;a=barriers(3;arrivals=1),b=scalars(Float32,5)))) == 48
    # Scalars at their alignment, two-dimensional groups.
    l2 = SharedLayout((;x=scalars(UInt8,3),y=scalars(UInt64,2),z=barriers((2,3);arrivals=128)))
    @test Tylo._offsets(values(l2.groups))[1] == (0,8,24)
    @test shared_bytes(l2) == 80
    @test_throws ArgumentError buffers(1024,2;align=24)
    # The layout of constants is a constant.
    @test Base.return_types(() -> shared_bytes(gemm_layout(4,49152)),())[1] == Int
    @test Core.Compiler.is_foldable(Base.infer_effects(shared_bytes,(typeof(l),)))
end

@testset "Rings" begin
    full = Barriers(reinterpret(Core.LLVMPtr{UInt64,3},UInt64(0x100)),(3,1),(1,3))
    empty = Barriers(reinterpret(Core.LLVMPtr{UInt64,3},UInt64(0x200)),(3,1),(1,3))
    r = Ring(full,empty)
    @test Tylo.stages(r) == 3
    for c in 0:10
        s = r[c]
        @test s.slot == c % 3
        @test Tylo._turn(s) == c ÷ 3
        @test s.full.ptr == full.ptr + 8(c % 3) && s.empty.ptr == empty.ptr + 8(c % 3)
    end
    # A row of a two-dimensional group: [i, j] at i + 2j.
    g = Barriers(reinterpret(Core.LLVMPtr{UInt64,3},UInt64(0x100)),(2,2),(1,2))
    @test g[1,1].ptr == g.ptr + 24
    row = selectdim(g,1,1)
    @test row[0].ptr == g.ptr + 8 && row[1].ptr == g.ptr + 24
    @test Tylo.stages(Ring(row)) == 2
end

@testset "Configurations" begin
    cfg = Config(stages=4,causal=true,emulated=(3,4),name=:x)
    @test cfg.stages === Int32(4) && cfg.causal === true && cfg.emulated == (3,4) && cfg.name === :x
    @test Base.issingletontype(typeof(cfg))
    @test sizeof(cfg) == 0
    @test merge(cfg,(;stages=2)).stages === Int32(2)
    @test Base.return_types(c -> c.stages,(typeof(cfg),))[1] == Int32
    @test Core.Compiler.is_foldable(Base.infer_effects(c -> c.stages,(typeof(cfg),)))
    @test_throws ArgumentError Config{1}()
end

@testset "Named barriers" begin
    nb = named_barriers((;handoff=named(8;threads=64),merge=named(;threads=128)))
    @test nb.handoff[0] == NamedBarrier(1,64) && nb.handoff[7] == NamedBarrier(8,64)
    @test nb.merge == NamedBarrier(9,128)
    @test_throws BoundsError nb.handoff[8]
    # A megakernel that reserves barrier 1 numbers its own from 2.
    @test named_barriers((;turn=named(2;threads=256));first=2).turn[1] == NamedBarrier(3,256)
    @test named_barriers((;a=named(15;threads=32))).a[14] == NamedBarrier(15,32)
    @test_throws ArgumentError named_barriers((;a=named(16;threads=32)))
    @test_throws ArgumentError named(;threads=48)
    @test Base.return_types(() -> named_barriers((;a=named(2;threads=64),b=named(;threads=128))).b,())[1] == NamedBarrier
end
