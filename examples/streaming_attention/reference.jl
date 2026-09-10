# Independent Float64 reference and a diagnostic streaming rounding model.
# Input storage matches kernel.jl; no Tylo primitive is used here.
function attention_reference(q,k,v,mask;causal=false,rounded=false)
    m,n=size(q,2),size(k,2)
    scores=(transpose(Float64.(q))*Float64.(k))./8
    out=zeros(Float64,64,m)
    for r in 1:m
        valid=[j for j in 1:n if mask[j,r] && (!causal || j<=r)]
        isempty(valid) && continue
        if !rounded
            x=scores[r,valid];p=exp.(x.-maximum(x));p./=sum(p)
            out[:,r]=transpose(Float64.(v[valid,:]))*p
        else
            # Match the per-32-key BF16 probability boundary, while doing the
            # statistics and weighted sum independently in Float64.
            maximum_old=-Inf;denominator=0.0;numerator=zeros(Float64,64)
            for first in 1:32:n
                js=[j for j in first:min(first+31,n) if mask[j,r] && (!causal || j<=r)]
                isempty(js) && continue
                maximum_new=max(maximum_old,maximum(scores[r,js]))
                alpha=maximum_old == -Inf ? 0.0 : exp(maximum_old-maximum_new)
                p=exp.(scores[r,js].-maximum_new)
                numerator=alpha.*numerator+transpose(Float64.(v[js,:]))*Float64.(BFloat16.(p))
                denominator=alpha*denominator+sum(p);maximum_old=maximum_new
            end
            out[:,r]=numerator./denominator
        end
    end
    out
end
