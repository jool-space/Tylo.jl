# A straightforward three-pass warp kernel: explicit scalar loops and
# shuffles, without Tylo fragments. Same arrays, mask, dtype and row ownership.
@inline function baseline_reduce(op::F,x) where F
    PTX.Utils.@unroll for offset in (16,8,4,2,1)
        x=op(x,shfl_xor_sync(typemax(UInt32),x,offset))
    end
    x
end
function baseline_softmax_kernel!(out,input,mask)
    tid=Int32(threadIdx().x)-Int32(1);lane=tid%Int32(32)
    row=(Int32(blockIdx().x)-Int32(1))*Int32(4)+tid÷Int32(32)+Int32(1)
    width=size(input,1)%Int32;rows=size(input,2)%Int32;m=-Inf32
    for j in lane+Int32(1):Int32(32):width
        if row<=rows && (@inbounds mask[j,row])
            m=max(m,Float32(@inbounds input[j,row]))
        end
    end
    m=baseline_reduce(max,m);s=0f0
    for j in lane+Int32(1):Int32(32):width
        if row<=rows && (@inbounds mask[j,row])
            s+=exp(Float32(@inbounds input[j,row])-m)
        end
    end
    s=baseline_reduce(+,s)
    for j in lane+Int32(1):Int32(32):width
        if row<=rows
            @inbounds out[j,row]=mask[j,row] && s>0f0 ? exp(Float32(input[j,row])-m)/s : 0f0
        end
    end
    nothing
end
