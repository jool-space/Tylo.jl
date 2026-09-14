@testset "Online row state and summary merge" begin
    function summarize(scores,values,width)
        s=SoftmaxState(local_fragment((0f0,)))
        numerator=0f0
        for first in 1:width:length(scores)
            stop=min(first+width-1,length(scores))
            f=local_fragment(Tuple(scores[first:stop]))
            update=softmax_update(s,f)
            numerator=only(update.rescale)*numerator+
                sum(update.weights.data .* Tuple(values[first:stop]))
            s=update.state
        end
        s,numerator
    end
    for shift in (0f0,1000f0,-1000f0), width in (1,2,3,7,17,64)
        scores=Float32[-Inf,-Inf,-23,2,-7,-Inf,15,15.125,-3,4,-Inf,-Inf]
        scores .+= shift
        values=Float32[0,0,10,100,-300,0,2,-1.99,6,20,0,0]
        s,n=summarize(scores,values,width)
        mx=maximum(Float64.(scores)); weights=exp.(Float64.(scores).-mx)
        expected=sum(weights .* values)/sum(weights)
        @test only(s.maximum) == mx
        @test only(s.sum) ≈ sum(weights) rtol=3e-6
        @test only(softmax_logsumexp(s)) ≈ mx+log(sum(weights)) atol=1e-4
        @test only(softmax_normalize(local_fragment((n,)),s).data) ≈ expected atol=2e-6 rtol=3e-5
        for split in 1:length(scores)-1
            a,na=summarize(scores[1:split],values[1:split],width)
            b,nb=summarize(scores[split+1:end],values[split+1:end],width)
            for (a,na,b,nb) in ((a,na,b,nb),(b,nb,a,na))
                merged=softmax_merge(a,b)
                out=only(merged.left_rescale)*na+only(merged.right_rescale)*nb
                @test only(merged.state.sum) ≈ sum(weights) rtol=3e-6
                @test only(softmax_normalize(local_fragment((out,)),merged.state).data) ≈ expected atol=2e-6 rtol=3e-5
            end
        end
    end
    empty=SoftmaxState(local_fragment((0f0,)))
    for f in (local_fragment((-Inf32,)),local_fragment((-Inf32,-Inf32,-Inf32)))
        u=softmax_update(empty,f)
        @test u.state.maximum.data == (-Inf32,)
        @test u.state.sum.data == (0f0,)
        @test all(iszero,u.weights.data)
        @test only(u.rescale) === 0f0
        @test only(softmax_logsumexp(u.state)) === -Inf32
        @test all(iszero,softmax_normalize(f,u.state).data)
    end
    merged=softmax_merge(empty,empty)
    @test only(merged.state.sum) === 0f0
    @test only(merged.state.maximum) === -Inf32
    @test only(merged.left_rescale) === only(merged.right_rescale) === 0f0
    @test_throws MethodError softmax_merge(empty,SoftmaxState(striped_fragment((0f0,))))
    for rm in (1,2),wm in (1,2)
        a=MMAAtom((16,8,16),BFloat16)
        scores=zero_accumulator(TiledMMA(a,Val((wm,1)),Val((rm,2)),Val(16)))
        outputs=map(_ -> 8f0,zero_accumulator(TiledMMA(a,Val((wm,1)),Val((rm,8)),Val(16))))
        o=Tylo._reduced_ownership(Tylo.Layouts.layout(scores),Val(2))
        s=SoftmaxState(Fragment(ntuple(_ -> 2f0,2rm),o),Fragment(ntuple(_ -> 4f0,2rm),o))
        y=softmax_normalize(outputs,s)
        @test all(==(2f0),y.data)
    end
end

@testset "Streaming softmax on either logical axis" begin
    for axis in (1,2)
        f = Fragment((-Inf32,2f0,3f0,-4f0),Tylo.Layouts.LocalOwnership{4,axis}())
        state = @inferred SoftmaxState(f;dims=Val(axis))
        update = @inferred softmax_update(state,f)
        normalized = @inferred softmax_normalize(update.weights,update.state)
        ref = exp.(Float64.(f.data) .- 3)
        @test all(isapprox.(normalized.data, ref ./ sum(ref); atol=2e-7))
        @test size(Tylo.Layouts.layout(update.state.sum)) == (axis == 2 ? (32,1) : (1,32))
        @test softmax_merge(update.state,state).state.sum.data == update.state.sum.data
        other = @inferred SoftmaxState(f;dims=Val(3-axis))
        @test size(Tylo.Layouts.layout(other.sum)) == (axis == 2 ? (1,4) : (4,1))
    end
end
