# TEST_TARGET: cc>=8.0
# Both the conversion path and the bit-preserving path run on ordinary NVIDIA
# GPUs. Typed destinations deliberately retain BF16/FP16 as their element type.
function packing_kernel!(out,back,input,::Type{T},::Val{W}) where {T,W}
    t = Int32(threadIdx().x)-Int32(1)
    f = Fragment(ntuple(i -> @inbounds(input[2W*t+i]),Val(2W)),lane_local(2W,2))
    packed = Fragment{T}(f)
    # Every thread receives a 16-byte aligned destination, including short tails.
    store!(pointer(out)+Int(t)*cld(4W,16)*16,packed)
    Tylo.@rtuple(1:2W) do i
        @inbounds back[2W*t+i] = Float32(packed.data[i])
    end
    nothing
end
function packed_bits_kernel!(out,input,::Type{T}) where T
    t = Int32(threadIdx().x)-Int32(1)
    f = Fragment(ntuple(i -> @inbounds(input[8t+i]),Val(8)),lane_local(8,2))
    g = Tylo._from_words(T,Tylo._pack_words(f),Tylo.Layouts.layout(f))
    Tylo.@rtuple(1:8) do i
        @inbounds out[8t+i] = g.data[i]
    end
    nothing
end
@testset "BF16 and FP16 register packing" begin
    for T in (BFloat16,Float16), w in (1,2,3,4,16)
        begin # assembly checks
            code = compile_kernel(packing_kernel!,Tuple{CuDeviceVector{T,1},CuDeviceVector{Float32,1},CuDeviceVector{Float32,1},Type{T},Val{w}};
                                  arch=CUDACore.SMVersion(12,1,:arch),threads=32)
            save_code("packing-$(T)-w$w",code)
            body = entry_body(code.ptx)
            @test occursin(T === BFloat16 ? "cvt.rn.bf16x2.f32" : "cvt.rn.f16x2.f32",body)
            @test !occursin(".local .",body)
            @test !occursin(r"\bcall",body)
        end
        if runtime_supported(@__FILE__)
            values = randn(MersenneTwister(47),Float32,32*2w)
            values[1:8] .= Float32[0,-0.0,Inf,-Inf,1.00390625,1.01171875,1.00048828125,1f-40]
            input = CuArray(values)
            stride = cld(4w,16)*8
            output = CuArray(fill(T(-7),stride*32))
            back = similar(input)
            @cuda threads=32 packing_kernel!(output,back,input,T,Val(w))
            expected = T.(values)
            actual = reshape(Array(output),stride,32)
            @test reinterpret.(UInt16,vec(actual[1:2w,:])) == reinterpret.(UInt16,expected)
            @test all(==(T(-7)),actual[2w+1:end,:])
            @test isequal(Array(back),Float32.(expected))
        end
    end
    if runtime_supported(@__FILE__)
        for T in (BFloat16,Float16)
            bits = UInt16[0,0x8000,0x7fff,0xffff,0x7f81,0x7c01,1,0x3c00]
            values = repeat(reinterpret.(T,bits),32)
            input = CuArray(values); out = similar(input)
            @cuda threads=32 packed_bits_kernel!(out,input,T)
            @test reinterpret.(UInt16,Array(out)) == reinterpret.(UInt16,values)
        end
    end
end

# FP8 packing converts through Microfloats' funnel. A kernel compiled for a
# target with the instructions gets one native `cvt` per two lanes, for
# packing from FP32, FP16 and BF16 values and for widening back; older targets
# run the same generic code as the host. Either way the bits match the host.
function fp8_packing_kernel!(words,wide,back,input,::Type{T},::Val{W}) where {T,W}
    t = Int32(threadIdx().x)-Int32(1)
    f = Fragment(ntuple(i -> @inbounds(input[4W*t+i]),Val(4W)),lane_local(4W,2))
    packed = Fragment{T}(f)
    a = Fragment{eltype(wide)}(packed)
    b = Fragment{Float32}(packed)
    packed_words = Tylo._pack_words(packed)
    Tylo.@rtuple(1:W) do i
        @inbounds words[W*t+i] = packed_words[i]
    end
    Tylo.@rtuple(1:4W) do i
        @inbounds wide[4W*t+i] = a.data[i]
        @inbounds back[4W*t+i] = b.data[i]
    end
    nothing
end
@testset "FP8 register packing" begin
    names = Dict(Float8_E4M3FN=>"e4m3x2",Float8_E5M2=>"e5m2x2",Float32=>"f32",Float16=>"f16x2",BFloat16=>"bf16x2")
    for T in (Float8_E4M3FN,Float8_E5M2), S in (Float32,Float16,BFloat16), H in (Float16,BFloat16), w in (1,4)
        tt = Tuple{CuDeviceVector{UInt32,1},CuDeviceVector{H,1},CuDeviceVector{Float32,1},CuDeviceVector{S,1},Type{T},Val{w}}
        begin # assembly checks
            native = compile_kernel(fp8_packing_kernel!,tt;arch=CUDACore.SMVersion(12,1,:arch),threads=32)
            save_code("fp8-packing-$(T)-$(S)-$(H)-w$w",native)
            body = entry_body(native.ptx)
            @test count("cvt.rn.satfinite.$(names[T]).$(names[S])",body) == 2w
            @test occursin("cvt.rn.$(names[H]).$(names[T])",body)
            @test !occursin(".local .",body)
            @test !occursin(r"\bcall",body)
            # sm_80 has none of these instructions and takes the generic path.
            generic = compile_kernel(fp8_packing_kernel!,tt;arch=CUDACore.SMVersion(8,0),threads=32)
            @test !occursin(names[T],entry_body(generic.ptx))
        end
        if runtime_supported(@__FILE__)
            values = S.(randn(MersenneTwister(91),Float32,32*4w) .* 40f0)
            values[1:8] .= S.(Float32[0,-0.0,Inf,-Inf,447.9,6f4,-1f9,1f-40])
            input = CuArray(values)
            words = CUDACore.zeros(UInt32,32w); wide = CUDACore.zeros(H,32*4w); back = CUDACore.zeros(Float32,32*4w)
            @cuda threads=32 fp8_packing_kernel!(words,wide,back,input,T,Val(w))
            expected = map(x -> Tylo.convert_element(T,x),values)
            @test reinterpret(UInt8,Array(words)) == reinterpret.(UInt8,expected)
            @test Array(wide) == H.(expected)
            @test Array(back) == Float32.(expected)
        end
    end
end

# A complete allocation/load/store/readback/deallocation path for every exposed
# .32x32b width and logical element type. All addresses are warp-uniform.
function typed_tmem_kernel!(out,input,::Val{W},::Val{A}) where {W,A}
    T = eltype(input)
    tid = UInt32(threadIdx().x)-UInt32(1)
    warp = tid >> UInt32(5)
    slot = @inbounds CuStaticSharedArray(UInt32,1)
    if warp == UInt32(0)
        ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(PTX.smem_addr_u32(pointer(slot)),UInt32(128))
    end
    sync_threads()
    base = @inbounds slot[1]
    n = W*(4÷sizeof(T))
    shape = A == 2 ? (32,n) : (n,32)
    tile = TmemTile(T,base,@Layout((128,n),(1,128)))
    band = @inbounds window(tile,(UInt32(32)*warp,UInt32(0)),(32,n))
    band = A == 2 ? band : permutedims(band)
    plan = TmemTransfer(shape,A)
    access = @inbounds partition(plan,band)
    f = Fragment(ntuple(i -> @inbounds(input[Int(tid)*n+i]),Val(n)),Tylo.Layouts.layout(plan))
    store_async!(access,f)
    wait_stores()
    pending = load_async(access)
    ready = wait_load(pending)
    values = ready
    Tylo.@rtuple(1:n) do i
        @inbounds out[Int(tid)*n+i] = values.data[i]
    end
    sync_threads()
    if warp == UInt32(0)
        ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(base,UInt32(128))
        ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    end
    nothing
end
@testset "Typed TMEM transfers, x1 through x128" begin
    for T in (Float32,BFloat16,Float16), w in (1,2,4,8,16,32,64,128), axis in (1,2)
        begin # assembly checks
            code = compile_kernel(typed_tmem_kernel!,Tuple{CuDeviceVector{T,1},CuDeviceVector{T,1},Val{w},Val{axis}})
            save_code("typed-tmem-$(T)-w$w-a$axis",code)
            body = entry_body(code.ptx)
            @test occursin("tcgen05.ld.sync.aligned.32x32b.x$w.b32",body)
            @test occursin("tcgen05.st.sync.aligned.32x32b.x$w.b32",body)
            @test occursin("tcgen05.wait::ld.sync.aligned",body)
            @test !occursin(r"\bcall",body)
            @test !occursin(".local .",body)
        end
        if capability_major(10)
            values = T.(randn(MersenneTwister(623),Float32,128*w*(4÷sizeof(T))))
            input = CuArray(values); out = similar(input)
            @cuda threads=128 typed_tmem_kernel!(out,input,Val(w),Val(axis))
            @test Array(out) == values
        end
    end
end
