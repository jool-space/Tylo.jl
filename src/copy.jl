function copy_async! end
"Close this thread's uncommitted cp.async operations as one group."
function commit_copies end
"Wait until at most `n` (a constant) committed cp.async groups remain for this thread; no CTA barrier."
function wait_copies end

@inline _valid_coordinate(t::MemoryTile,c::Tuple) =
    0 <= c[1] < size(t)[1] && 0 <= c[2] < size(t)[2]
