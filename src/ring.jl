# Rings of pipeline stages over groups of barriers, indexed by a count the
# kernel keeps.

"""
    Ring(full, empty=nothing)

A ring of stages over a group of [`Barriers`](@ref) that complete when a
stage is filled (`full`) and, optionally, a group that completes when it may
be refilled (`empty`), one barrier of each per stage. A ring holds no state:
`ring[c]` is the [`Stage`](@ref) of the kernel's `c`-th use of the ring,
counted from zero since the barriers were initialized, at slot `c % stages`
in round `c ÷ stages`. A producer and its consumers each count their own
uses.

```julia
kv = Ring(smem.full, smem.empty)
s = kv[c]                                   # the producer's c-th fill
acquire(s)                                  # wait until the slot is free
arrive!(s.full, bytes); tma_load!(shared_tile(binding, smem.stage[s.slot]), binding, origin, s.full)
s = kv[c]                                   # a consumer's c-th use
wait(s)                                     # wait until the slot is full
mma(...); commit_mma(atom, s.empty)         # free it once the MMAs complete
```

Waits are explicit and releases use the barriers' own verbs: the ring only
computes which barrier and which phase.
"""
struct Ring{E<:Union{Barriers,Nothing}}
    full::Barriers
    empty::E
end
Ring(full::Barriers) = Ring(full,nothing)
"The number of stages of `ring`."
@inline stages(r::Ring) = length(r.full)

"""
    Stage

The `index`-th use of a [`Ring`](@ref): its slot (`s.slot`, zero-based), its
barriers (`s.full`, `s.empty`), and the phases to wait for ([`wait`](@ref),
[`acquire`](@ref)).
"""
struct Stage{R<:Ring}
    ring::R
    index::Int32
end
@inline Base.getindex(r::Ring,c::Integer) = Stage(r,c % Int32)
# Counts are non-negative: unsigned division by a constant is a shift (or a
# multiply) with no correction for negative values.
# The zero-based slot of stage `s`, the index of its barriers and buffer.
@inline _slot(s::Stage) = ((getfield(s,:index) % UInt32) % (stages(getfield(s,:ring)) % UInt32)) % Int32
@inline _turn(s::Stage) = (getfield(s,:index) % UInt32) ÷ (stages(getfield(s,:ring)) % UInt32)
@inline function Base.getproperty(s::Stage,name::Symbol)
    name === :slot && return _slot(s)
    name === :full && return getfield(s,:ring).full[_slot(s)]
    name === :empty && return getfield(s,:ring).empty[_slot(s)]
    getfield(s,name)
end
Base.propertynames(::Stage) = (:slot,:full,:empty,:ring,:index)

"""
    wait(s::Stage)

Wait until stage `s` is full: its `full` barrier's phase of this round.
"""
@inline Base.wait(s::Stage) = wait(s.full,_turn(s))

"""
    acquire(s::Stage)

Wait until stage `s` may be filled: in the ring's first round at once,
afterwards until the previous round's consumers released its slot (its
`empty` barrier's phase of the previous round).
"""
@inline function Base.acquire(s::Stage{<:Ring{Barriers}})
    getfield(s,:index) >= stages(getfield(s,:ring)) && wait(s.empty,_turn(s)-UInt32(1))
    nothing
end
