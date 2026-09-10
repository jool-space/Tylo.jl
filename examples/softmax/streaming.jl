# Include after kernel.jl. Capacity is fixed; width and row count are runtime.
# First pass computes final statistics, second pass rereads and normalizes.
function streaming_softmax_kernel!(output,input,mask,::Val{N}) where N
    tid=Int32(threadIdx().x)-Int32(1); lane=tid%Int32(32)
    row=(Int32(blockIdx().x)-Int32(1))*Int32(4)+tid÷Int32(32)+Int32(1)
    width,rows=size(input,1)%Int32,size(input,2)%Int32
    state=SoftmaxState(WarpRowFragment(ntuple(_ -> 0f0,Val(N))))
    for start in Int32(0):Int32(32N):width-Int32(1)
        f=WarpRowFragment(ntuple(Val(N)) do e
            j=start+lane+Int32(32(e-1)+1)
            row<=rows && j<=width && (@inbounds mask[j,row]) ? Float32(@inbounds input[j,row]) : -Inf32
        end)
        state=softmax_update(state,f).state
    end
    final_max,final_sum=only(state.maximum),only(state.sum)
    for start in Int32(0):Int32(32N):width-Int32(1)
        for e in 1:N
            j=start+lane+Int32(32(e-1)+1)
            if row<=rows && j<=width
                x=@inbounds input[j,row]
                valid=@inbounds mask[j,row]
                @inbounds output[j,row]=valid && final_sum>0f0 ?
                    exp(Float32(x)-final_max)/final_sum : 0f0
            end
        end
    end
    nothing
end
