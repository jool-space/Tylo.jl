function copy_async! end
"Close this thread's uncommitted cp.async operations as one group."
function commit_copies end
"Wait until at most `n` (a constant) committed cp.async groups remain for this thread; no CTA barrier."
function wait_copies end

