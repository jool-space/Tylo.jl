function copy_async! end
"""
    commit_copies()
    commit_copies(barrier::Barrier)

Close this thread's uncommitted cp.async copies as one group, completed by
[`wait_copies`](@ref); or have them arrive on `barrier` as they complete,
as one of its phase's expected arrivals.
"""
function commit_copies end
"Wait until at most `n` (a constant) committed cp.async groups remain for this thread; no CTA barrier."
function wait_copies end

