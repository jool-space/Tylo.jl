# Collective reductions derived from ownership enumeration. Device shuffles
# live in the PTX extension; the recipe itself is chosen while generating.

# Balanced local tree with static tuple access.
@generated function _local_reduce(op::F,x::NTuple{N,T}) where {F,N,T}
    N > 0 || error("empty register reduction")
    function tree(first,last)
        first == last && return :(x[$first])
        mid = (first+last)÷2
        :(op($(tree(first,mid)),$(tree(mid+1,last))))
    end
    quote
        Base.@inline
        $(tree(1,N))
    end
end

"Butterfly reduction over lane offsets W/2 down to 1; implemented by the PTX extension."
_warp_reduce(op,x,width) = throw(ArgumentError("warp shuffles require the PTX extension"))
"One xor-shuffle exchange at a lane offset; implemented by the PTX extension."
_shuffle_xor(op,x,offset) = throw(ArgumentError("warp shuffles require the PTX extension"))

"""
    _reduce_values(op, data, ownership, Val(axis))

Reduce this thread's values along a logical axis using the recipe derived
from the ownership: a local tree over the slots sharing each kept coordinate,
then xor shuffles over the lane bits that replicate it. One result per kept
coordinate held by this thread.
"""
@generated function _reduce_values(op::F,data::NTuple{N,T},::L,::Val{Axis}) where {F,N,T,L,Axis}
    o = _static_instance(L)
    o === nothing && return :(throw(ArgumentError("reductions require a static ownership")))
    plan = reduction_plan(o,Axis)
    plan === nothing &&
        return :(throw(ArgumentError("no reduction implementation for this ownership layout and axis")))
    results = map(plan.groups) do group
        local_value = length(group) == 1 ? :(data[$(only(group))]) :
            :(_local_reduce(op,($([:(data[$e]) for e in group]...),)))
        bits = plan.bits
        if isempty(bits)
            local_value
        elseif bits == collect(0:length(bits)-1)
            :(_warp_reduce(op,$local_value,Val($(1 << length(bits)))))
        else
            foldl((value,b) -> :(_shuffle_xor(op,$value,Val($(1 << b)))),reverse(bits);init=local_value)
        end
    end
    quote
        Base.@inline
        ($(results...),)
    end
end

"The replicated result ownership of reducing `axis`, or an error when no recipe exists."
@generated function _reduced_ownership(::L,::Val{Axis}) where {L,Axis}
    o = _static_instance(L)
    o === nothing && return :(throw(ArgumentError("reductions require a static ownership")))
    plan = reduction_plan(o,Axis)
    plan === nothing &&
        return :(throw(ArgumentError("no reduction implementation for this ownership layout and axis")))
    :($(plan.result))
end

@inline Base.only(x::Fragment{T,1}) where T = only(x.data)
