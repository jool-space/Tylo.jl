# Run via test/gpu/flash_attention.jl. Load the reference kernel into two
# isolated modules; replace ONLY the TMA stripe loads, the two MMA issue
# helpers, correction and epilogue in the second.
const FA_REFERENCE_PATH = joinpath(@__DIR__,"reference.jl")

function attention_module(name; tiled=false)
    source = read(FA_REFERENCE_PATH,String)
    mod = Module(name)
    Core.eval(mod,:(using PTX, CUDACore, Random, Tylo))
    if tiled
        for helper in ("fab_load_tile","fab_corr_tile","fab_epi_stage","fab_mma_operands","fab_qk_mma","fab_pv_mma")
            source = replace(source,"function "*helper*"(" => "function unused_"*helper*"(")
        end
    end
    Base.include_string(mod,source,FA_REFERENCE_PATH)
    tiled && Base.include(mod,joinpath(@__DIR__,"tiles.jl"))
    mod
end

const ReferenceAttention = attention_module(:ReferenceAttention)
const TiledAttention = attention_module(:TiledAttention;tiled=true)
include("runtime.jl")

function attention_types(cfg)
    Tuple{CuDeviceVector{ReferenceAttention.BFloat16,1},
          PTX.TMADescriptorPtr,PTX.TMADescriptorPtr,PTX.TMADescriptorPtr,
          UInt32,UInt32,UInt32,UInt32,UInt32,Float32,
          CuDeviceVector{UInt32,1},typeof(Val(cfg))}
end

@testset "Full FlashAttention code generation" begin
    for (major,minor,feature) in ((10,0,:arch),(10,3,:arch),(10,0,:family)),
            splitp in (true,false)
        arch = CUDACore.SMVersion(major,minor,feature)
        cfg = (;ReferenceAttention.FAB_CFG_DEFAULT...,splitp)
        tag = "$(major)$(minor)-$feature-splitp$splitp"
        codes = map((ReferenceAttention,TiledAttention)) do mod
            code = compile_kernel(mod.fab_kernel!,attention_types(cfg);arch,threads=512)
            save_code("attention-$(nameof(mod))-$tag",code)
            @test !isempty(code.image)
            @test occursin(".reqntid 512, 1, 1",code.ptx)
            # Exception reporting has separate functions and local printf
            # buffers. The entry must not materialize register fragments.
            @test !occursin(".local .",entry_body(code.ptx))
            code
        end
        # The softmax, correction and epilogue code is shared or was shown
        # byte-identical earlier; the MMA issue is where the two differ. The
        # atom's descriptor arithmetic keeps the descriptor's high word
        # constant, so the MMA warp's code is smaller than the reference's
        # hand-written offsets produce, and no exception path remains in
        # either kernel. Preserve load/store widths, waits and publication
        # granularity exactly, and never let the machine code grow.
        @test count("call.uni",codes[1].ptx) == count("call.uni",codes[2].ptx) == 0
        @test length(kernel_text(codes[2].image)) <= length(kernel_text(codes[1].image))
        for spelling in ("tcgen05.ld.sync.aligned.32x32b.x64",
                         "tcgen05.st.sync.aligned.32x32b.x64",
                         "tcgen05.st.sync.aligned.32x32b.x16",
                         "tcgen05.wait::ld", "tcgen05.wait::st",
                         "tcgen05.fence::before_thread_sync",
                         "tcgen05.commit", "mbarrier.arrive",
                         "mbarrier.init", "st.global.v4.b32")
            @test count(spelling,codes[1].ptx) == count(spelling,codes[2].ptx)
        end
        # The publish-group loop of the PV issue: the tiled helper's smaller
        # body is unrolled completely (all 64 PV instructions materialize);
        # in the quarter-granular fallback the reference's loop is only
        # partially unrolled by LLVM, so its static count is lower.
        for spelling in ("tcgen05.mma","tcgen05.fence::after_thread_sync")
            if splitp
                @test count(spelling,codes[1].ptx) == count(spelling,codes[2].ptx)
            else
                @test count(spelling,codes[1].ptx) <= count(spelling,codes[2].ptx)
            end
        end
        @test count("tcgen05.mma",codes[2].ptx) == 128
    end
end

if capability_major(10)
    run_attention_cases()
else
    @testset "FlashAttention execution requires B200/B300" begin
        @test_skip false
    end
    "--bench" in ARGS && error("Attention benchmarks require datacenter Blackwell")
end
