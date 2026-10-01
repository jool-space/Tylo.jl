# TEST_TARGET: cc==10
# The Blackwell FlashAttention forward example, written in Tylo throughout:
# warp-specialized roles with register reallocation, S, P and O in TMEM, P
# published in two parts, in one CTA or CTA pairs. Compiled clean for
# sm_100a, correct on B200/B300.
include("../../examples/blackwell_attention/kernel.jl")

@testset "Blackwell attention assembly" begin
    arch = CUDACore.SMVersion(10,0,:arch)
    # Launches as a B200 (148 SMs) would run them, prepared on any device.
    x(dims...) = CUDACore.zeros(BFloat16,128,dims...)
    attention(q,k=q,v=k;kwargs...) = blackwell_attention_launch(q,k,v;sms=148,kwargs...)
    for ctas in (1,2)
        code = machine_code(attention(x(512,4);ctas,tail=false);arch)
        save_code("blackwell-attention-ctas$ctas",code)
        # No spills anywhere: the softmax warps hold a row at 184 registers,
        # the correction warps a row's merge at 80, the MMA warp its operands
        # at 64, because each product derives its
        # descriptors at the issue.
        @test isempty(audit(code))
        sass = Tylo.kernel_sass(code)
        @test count("USETMAXREG",sass) == 3
        # The MMA warp issues in line, from the lane each instruction's own
        # asm elects: no instruction sits in a loop over lanes that might
        # issue (`BRA.U.ANY`), nor moves operands in one (`R2UR.BROADCAST`).
        @test count("BRA.U.ANY",sass) == 0 && count("R2UR.BROADCAST",sass) == 0
        body = entry_body(code.ptx)
        @test count("tcgen05.ld.sync.aligned.32x32b.x32",body) >= 4
        @test count("tcgen05.st.sync.aligned.32x32b.x16",body) == 4
        # Every tcgen05 instruction in one CTA group.
        @test count("tcgen05.mma.cta_group::$ctas",body) == count("tcgen05.mma",body) > 0
        @test count("tcgen05.commit.cta_group::$ctas",body) == count("tcgen05.commit",body) > 0
        if ctas == 2
            @test count("multicast::cluster",body) == count("tcgen05.commit",body)
            @test occursin("cp.async.bulk.tensor.3d.cta_group::2",body) && occursin("mbarrier.arrive.shared::cluster",body)
        end
    end
    # A split tail's two parts in a cluster (one CTA per item): each merges
    # one tile, the other's part of it copied into its shared memory in bulk.
    # (60 items on 148 SMs split in two, 120 CTAs in pairs.)
    let l = attention(x(512,30))
        @test l.options.clustersize == 2
        code = machine_code(l;arch)
        @test isempty(audit(code))
        body = entry_body(code.ptx)
        @test count("cp.async.bulk.shared::cluster.shared::cta",body) == 1 && occursin("mbarrier.arrive.release.cluster.shared::cluster",body)
        @test !occursin("ld.shared::cluster",body)
        @test !occursin("atom.add",body)                # no counts: the cluster barrier orders the merge
    end
    # Heads and batches along two axes of any strides: four-axis copies,
    # the same clean code.
    let code = machine_code(attention(x(256,2,2);tail=false);arch)
        @test isempty(audit(code))
        body = entry_body(code.ptx)
        @test occursin("cp.async.bulk.tensor.4d",body) && !occursin("cp.async.bulk.tensor.3d",body)
    end
    # Sequences of their own lengths packed along the rows: their geometry
    # read from a table per item, the same clean code.
    for causal in (false,true)
        code = machine_code(attention(x(512,2);lengths=[256,256],causal,tail=false);arch)
        @test isempty(audit(code))
    end
    # Causal: the last two blocks masked, the rest unchanged; items handed
    # out dynamically.
    let code = machine_code(attention(x(512,2);causal=true,emulate=());arch)
        save_code("blackwell-attention-causal",code)
        @test isempty(audit(code))
        @test count("USETMAXREG",Tylo.kernel_sass(code)) == 3
    end
    # Given only its thread count, ptxas drops the register reallocation and
    # says so in a warning: the audit reports it.
    l = attention(x(512,2);emulate=(),tail=false)
    unstated = Tylo.machine_code(l.kernel,argument_types(l);arch,threads=512)
    @test :ptxas in getfield.(audit(unstated),:kind)
end

if runtime_supported(@__FILE__)
@testset "Blackwell attention on Blackwell" begin
    function reference(q,k,v;causal=false)                         # d×seq per head, Float64
        o = similar(q,Float64)
        for h in axes(q,3)
            s = transpose(Float64.(q[:,:,h]))*Float64.(k[:,:,h]) ./ sqrt(size(q,1))
            causal && (s[[c > r+size(k,2)-size(q,2) for r in axes(s,1), c in axes(s,2)]] .= -Inf)   # the diagonal meets the keys' end
            p = exp.(s .- maximum(s;dims=2)); p ./= sum(p;dims=2)
            o[:,:,h] = transpose(p*transpose(Float64.(v[:,:,h])))
        end
        o
    end
    rng = MersenneTwister(3)
    # One block; several (fewer items than SMs: split along the keys); more
    # tiles than SMs, so that CTAs walk several, of
    # several blocks and of one; and scores whose row maxima move by far more
    # than the rescale threshold.
    for ctas in (1,2), (sq,skv,bh,amp) in ((512,128,1,1f0),(512,1024,2,1f0),(2048,1024,80,1f0),(2048,128,40,1f0),(512,512,3,8f0))
        q = BFloat16.(amp .* randn(rng,Float32,128,sq,bh))
        k = BFloat16.(randn(rng,Float32,128,skv,bh)); v = BFloat16.(randn(rng,Float32,128,skv,bh))
        kernel, launch, o = blackwell_attention(CuArray(q),CuArray(k),CuArray(v);ctas)
        fill!(o,BFloat16(NaN)); launch(); synchronize()
        @test maximum(abs.(Float64.(Array(o)) .- reference(q,k,v))) < 0.03
        @test isempty(audit(machine_code(kernel)))
    end
    # Causal: items of different lengths, longest first, walked by fewer
    # CTAs than there are items; launched twice.
    for (sq,bh,amp) in ((256,1,1f0),(1024,3,1f0),(2048,80,1f0),(1024,2,8f0)), dynamic in (true,false)
        q = BFloat16.(amp .* randn(rng,Float32,128,sq,bh))
        k = BFloat16.(randn(rng,Float32,128,sq,bh)); v = BFloat16.(randn(rng,Float32,128,sq,bh))
        _, launch, o = blackwell_attention(CuArray(q),CuArray(k),CuArray(v);causal=true,dynamic)
        want = reference(q,k,v;causal=true)
        for _ in 1:2                                            # the item counter returns to zero
            fill!(o,BFloat16(NaN)); launch(); synchronize()
            @test maximum(abs.(Float64.(Array(o)) .- want)) < 0.03
        end
    end
    @test_throws ArgumentError blackwell_attention(CUDACore.zeros(BFloat16,128,512,1),CUDACore.zeros(BFloat16,128,512,1),CUDACore.zeros(BFloat16,128,512,1);causal=true,ctas=2)
    @test_throws ArgumentError blackwell_attention(CUDACore.zeros(BFloat16,128,512,1),CUDACore.zeros(BFloat16,128,256,1),CUDACore.zeros(BFloat16,128,256,1);causal=true)
    # Other shares of the exponentials on the FMA pipe: none, and more.
    let q = BFloat16.(randn(rng,Float32,128,1024,2)), k = BFloat16.(randn(rng,Float32,128,1024,2)), v = BFloat16.(randn(rng,Float32,128,1024,2))
        for emulate in ((),(1,3,4,6,8,9,11,13,14,15))
            _, launch, o = blackwell_attention(CuArray(q),CuArray(k),CuArray(v);emulate)
            fill!(o,BFloat16(NaN)); launch(); synchronize()
            @test maximum(abs.(Float64.(Array(o)) .- reference(q,k,v))) < 0.03
        end
    end
    # Items of a last, partial round split along the keys (unevenly: 11
    # blocks in 5 parts), merged by the part that finishes last; twice, as
    # the merge returns its counts to zero for the next launch. Then in two
    # parts, which one CTA per item merges in clusters of two: as many
    # items as SMs and 40% more (SMs ÷ 40% = 2 parts), one per head.
    sms = CUDACore.attribute(CUDACore.device(),CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    for (ctas,heads,sq) in ((1,300,512),(2,150,1024),(1,sms+(4sms)÷10,256))
        q,k,v = (BFloat16.(randn(rng,Float32,128,skv,heads)) for skv in (sq,1408,1408))
        _, launch, o = blackwell_attention(CuArray(q),CuArray(k),CuArray(v);ctas)
        _, whole, o2 = blackwell_attention(CuArray(q),CuArray(k),CuArray(v);ctas,tail=false)
        whole(); synchronize()
        want = reference(q,k,v)
        for _ in 1:2
            fill!(o,BFloat16(NaN)); launch(); synchronize()
            @test maximum(abs.(Float64.(Array(o)) .- want)) < 0.03
            @test maximum(abs.(Float64.(Array(o)) .- Float64.(Array(o2)))) < 0.02
        end
    end
    # Heads and batches of any strides: PyTorch's (batch, seq, heads, d)
    # tensors, bound in place and written back in their layout, with query
    # heads sharing K and V heads.
    for (ctas,causal,sq,skv,h,hkv,nb) in ((1,false,512,1024,4,2,2),(2,false,1024,512,4,1,2),(1,true,1024,1024,6,2,1),(1,false,333,517,4,2,2),(1,true,333,333,2,1,2))
        host(s,heads) = BFloat16.(randn(rng,Float32,128,heads,s,nb))   # d×heads×seq×batch
        qh, kh, vh = host(sq,h), host(skv,hkv), host(skv,hkv)
        bound(x) = PermutedDimsArray(CuArray(x),(1,3,2,4))
        _, launch, o = blackwell_attention(bound(qh),bound(kh),bound(vh);ctas,causal)
        @test o isa PermutedDimsArray
        logical(x) = permutedims(x,(1,3,2,4))                         # d×seq×heads×batch
        shared(x) = reshape(logical(x)[:,:,(0:h-1) .÷ (h÷hkv) .+ 1,:],128,:,h*nb)
        want = reference(reshape(logical(qh),128,sq,h*nb),shared(kh),shared(vh);causal)
        for _ in 1:2
            fill!(parent(o),BFloat16(NaN)); launch(); synchronize()
            @test maximum(abs.(Float64.(reshape(logical(Array(parent(o))),128,sq,h*nb)) .- want)) < 0.03
        end
    end
    # Three axes, K and V shared by all query heads, O every other head of a
    # larger array: the heads between keep their contents.
    let q = BFloat16.(randn(rng,Float32,128,512,4)), k = BFloat16.(randn(rng,Float32,128,768,1)), v = BFloat16.(randn(rng,Float32,128,768,1))
        buffer = CuArray(fill(BFloat16(7),128,512,8))
        _, launch, _ = blackwell_attention(CuArray(q),CuArray(k),CuArray(v);o=view(buffer,:,:,2:2:8))
        launch(); synchronize()
        result = Array(buffer)
        @test maximum(abs.(Float64.(result[:,:,2:2:8]) .- reference(q,repeat(k,1,1,4),repeat(v,1,1,4)))) < 0.03
        @test all(==(BFloat16(7)),result[:,:,1:2:8])
    end
    @test_throws ArgumentError blackwell_attention(CUDACore.zeros(BFloat16,128,512,3),CUDACore.zeros(BFloat16,128,512,2),CUDACore.zeros(BFloat16,128,512,2))
    # Lengths of any size: a last, partial item and key block, zero-filled,
    # masked and not stored; split along the keys (items fewer than SMs), in
    # CTA pairs, causal, and single rows; then two parts in clusters.
    for (sq,skv,bh,kw) in ((300,200,3,(;)),(1000,1100,10,(;)),(1000,1100,10,(;ctas=2)),(1,1,2,(;)),(129,1,1,(;)),(77,300,2,(;tail=false)),
                           (300,300,3,(;causal=true)),(1000,1000,2,(;causal=true,dynamic=false)),(1,1,1,(;causal=true)),(129,129,2,(;causal=true)),
                           (300,1000,2,(;causal=true)),(256,512,3,(;causal=true,dynamic=false)),(1,700,2,(;causal=true)),(129,200,2,(;causal=true)),
                           (256,1400,sms+(4sms)÷10,(;)))
        q = BFloat16.(randn(rng,Float32,128,sq,bh))
        k = BFloat16.(randn(rng,Float32,128,skv,bh)); v = BFloat16.(randn(rng,Float32,128,skv,bh))
        _, launch, o = blackwell_attention(CuArray(q),CuArray(k),CuArray(v);kw...)
        want = reference(q,k,v;causal=get(kw,:causal,false))
        for _ in 1:2
            fill!(o,BFloat16(NaN)); launch(); synchronize()
            @test maximum(abs.(Float64.(Array(o)) .- want)) < 0.03
        end
    end
    # Sequences of their own lengths packed along the rows: each attends
    # within itself, with more keys than queries, causal (the diagonal at
    # each's keys' end), with shared K and V heads, from PyTorch's packed
    # (rows, heads, d) layout.
    packed(x,lengths) = [x[:,sum(lengths[1:s-1])+1:sum(lengths[1:s]),:] for s in eachindex(lengths)]
    for (lq,lk,h,hkv,kw) in (([100,300,1,1000],[50,700,3,1000],4,4,(;)),([100,300,1,1000],[100,500,1,1200],2,1,(;causal=true)),
                             ([257,512,129],[257,512,129],4,2,(;causal=true,dynamic=false)),([3000,40],[3000,40],2,2,(;)),([600],[900],2,2,(;ctas=2)))
        q = BFloat16.(randn(rng,Float32,128,h,sum(lq))); k, v = (BFloat16.(randn(rng,Float32,128,hkv,sum(lk))) for _ in 1:2)
        rows(x) = PermutedDimsArray(CuArray(x),(1,3,2))                    # d×rows×heads
        _, launch, o = blackwell_attention(rows(q),rows(k),rows(v);lengths=lq,kv_lengths=lk,kw...)
        shared(x) = x[:,:,(0:h-1) .÷ (h÷hkv) .+ 1]
        want = reduce(hcat,[reference(a,shared(b),shared(c);causal=get(kw,:causal,false)) for (a,b,c) in zip(packed(permutedims(q,(1,3,2)),lq),packed(permutedims(k,(1,3,2)),lk),packed(permutedims(v,(1,3,2)),lk))])
        for _ in 1:2
            fill!(parent(o),BFloat16(NaN)); launch(); synchronize()
            @test maximum(abs.(Float64.(permutedims(Array(parent(o)),(1,3,2))) .- want)) < 0.03
        end
    end
    @test_throws ArgumentError blackwell_attention(CUDACore.zeros(BFloat16,128,512,1),CUDACore.zeros(BFloat16,128,512,1),CUDACore.zeros(BFloat16,128,512,1);lengths=[100,400])
    # P published after one or two of its chunks instead of three.
    for ctas in (1,2), split in (1,2)
        q,k,v = (BFloat16.(randn(rng,Float32,128,1024,2)) for _ in 1:3)
        _, launch, o = blackwell_attention(CuArray(q),CuArray(k),CuArray(v);ctas,split)
        fill!(o,BFloat16(NaN)); launch(); synchronize()
        @test maximum(abs.(Float64.(Array(o)) .- reference(q,k,v))) < 0.03
    end
end
end
