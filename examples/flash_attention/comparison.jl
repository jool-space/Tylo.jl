# Run via test/gpu/runtests.jl --attention. Read the existing PTX kernel into
# two isolated modules; replace ONLY correction and epilogue in the second.
# The digest makes a reference update an explicit review, never a silent change.
using SHA

const FA_REFERENCE_SHA256 = "d4bcc34234bf2a9d85d9fed136f15e035d28dc84123f46d0651958745f132cdc"
const FA_REFERENCE_PATH = joinpath(get(ENV,"TYLO_PTX_ROOT",
    dirname(dirname(pathof(PTX)))),"test","gpu","blackwell","flash_attention_defs.jl")

function attention_module(name; tiled=false)
    source = read(FA_REFERENCE_PATH,String)
    bytes2hex(sha256(source)) == FA_REFERENCE_SHA256 ||
        error("FlashAttention reference changed: review and update FA_REFERENCE_SHA256")
    mod = Module(name)
    Core.eval(mod,:(using PTX, CUDACore, Random, Tylo))
    # Preserve the reference's host quantization exactly.
    Core.eval(mod,:(bf16_bits(x::Float32) =
        UInt16((reinterpret(UInt32,x)+UInt32(0x8000)) >> 16)))
    Core.eval(mod,:(bf16_to_f32(x::UInt16) = reinterpret(Float32,UInt32(x)<<16)))
    if tiled
        for name in ("fab_corr_tile","fab_epi_stage")
            source = replace(source,"function "*name*"(" => "function unused_"*name*"(")
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
    Tuple{CuDeviceVector{UInt16,1},
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
        # Stronger than matching PTX: preserve the complete kernel text,
        # including register allocation, backend unrolling, and spill code.
        @test kernel_text(codes[1].image) == kernel_text(codes[2].image)
        # Preserve load/store widths, waits, and publication granularity.
        for spelling in ("tcgen05.ld.sync.aligned.32x32b.x64",
                         "tcgen05.st.sync.aligned.32x32b.x64",
                         "tcgen05.st.sync.aligned.32x32b.x16",
                         "tcgen05.wait::ld", "tcgen05.wait::st",
                         "tcgen05.fence::before_thread_sync",
                         "tcgen05.fence::after_thread_sync",
                         "tcgen05.mma", "mbarrier.arrive",
                         "mbarrier.init", "st.global.v4.b32")
            @test count(spelling,codes[1].ptx) == count(spelling,codes[2].ptx)
        end
    end
end

println("FlashAttention reference SHA256: ",FA_REFERENCE_SHA256)
if CUDACore.functional() && CUDACore.capability(CUDACore.device()) in (v"10.0",v"10.3")
    run_attention_cases()
else
    @testset "FlashAttention execution requires B200/B300" begin
        @test_skip false
    end
    "--bench" in ARGS && error("Attention benchmarks require datacenter Blackwell")
end
