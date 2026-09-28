import Adapt

"""
    Timeline(records)

When things happen inside a kernel: [`record!`](@ref) stores the SM's clock
with an event number, in `records`, a zeroed `capacity × recorders ×
blocks` array of `UInt64` (a `CuArray` that the kernel receives as an
argument). A recorder is whatever the kernel chooses, typically one
thread of each warp role, and a slot one of its records: a loop records at
slots it computes from its step, so that a timeline holds the steps it has
room for and drops the rest.

```julia
timeline = Timeline(CUDACore.zeros(UInt64,256,16,1))    # 256 records of 16 recorders, one block
@cuda kernel(..., timeline)                              # r = recorder(timeline, warp); record!(r, 2step, 1) ...
spans(timeline, 1 => 2)                                  # cycles from each event 1 to the next 2
```

Clocks are per SM, so times compare within a block, not across blocks.
[`events`](@ref) lists the records, [`spans`](@ref) measures between them.
A record is a clock read, an OR and a global store, from one thread:
cheap, not free, so a kernel records under a compile-time switch, a
timeline argument that may be `nothing`.
"""
struct Timeline{A<:AbstractArray{UInt64,3}}
    records::A
end

Adapt.adapt_structure(to,t::Timeline) = Timeline(Adapt.adapt(to,t.records))

"""
    recorder(timeline, r) -> Recorder

In a kernel: recorder `r` (counted from 0) of this block, which
[`record!`](@ref) writes. Take it once, before the loops that record: a
record is then a clock read and a store. A recorder or block the timeline
has no room for records nothing; so does the recorder of a `nothing`
timeline, which is `nothing`.
"""
function recorder end

# Slot 0 of a recorder's records in this block, and how many it holds.
struct Recorder{P}
    records::P
    capacity::Int32
end

"""
    record!(recorder, slot, event)
    record!(timeline, r, slot, event)

In a kernel: store the SM's clock with `event` (0-255) as record `slot`
of a [`recorder`](@ref), counted from 0. Records past its capacity are
dropped. The executing thread records; a warp records from one of its
lanes. The clock read keeps its place among memory operations and
barriers, not among pure arithmetic around it. The second form takes the
recorder anew, for a record or two outside loops.
"""
function record! end

# A record: the event in the top byte, the clock below.
const clock_mask = (UInt64(1) << 56)-UInt64(1)

"""
    events(timeline) -> Vector

The records of a [`Timeline`](@ref), copied to the host: named tuples of
`block`, `recorder`, `slot`, `event` and `time`, in cycles since the
block's first record, in time order.
"""
function events(t::Timeline)
    records = Array(t.records)
    list = NamedTuple{(:block,:recorder,:slot,:event,:time),NTuple{5,Int}}[]
    for b in axes(records,3)
        taken = filter(!iszero,view(records,:,:,b))
        isempty(taken) && continue
        start = minimum(x -> x & clock_mask,taken)
        for r in axes(records,2), s in axes(records,1)
            x = records[s,r,b]
            iszero(x) || push!(list,(block=b-1,recorder=r-1,slot=s-1,event=Int(x >> 56),time=Int((x & clock_mask)-start)))
        end
    end
    sort!(list;by=e -> (e.block,e.time))
end

"""
    spans(timeline, from => to; recorders=nothing) -> Vector{Int}

Cycles from each record of event `from` to its recorder's next record of
event `to` (or of `from` again, for `from => from`), over all blocks and
the `recorders` given (all by default).
"""
function spans(t::Timeline,pair::Pair{<:Integer,<:Integer};recorders=nothing)
    from, to = pair
    list = events(t)
    result = Int[]
    for key in unique((e.block,e.recorder) for e in list)
        recorders === nothing || key[2] in recorders || continue
        mine = filter(e -> (e.block,e.recorder) == key,list)
        for (k,e) in enumerate(mine)
            e.event == from || continue
            next = findnext(x -> x.event == to,mine,k+1)
            next === nothing || push!(result,mine[next].time-e.time)
        end
    end
    result
end
