module Tylo

import cuTile
import cuTile as ct
using cuTile: Constant, TFloat32, BFloat16

using CUDACore: @cuda, i32, CuArray
using Adapt: Adapt, adapt

macro cutile(args...)
    esc(:(@cuda backend=$cuTile $(args...)))
end

include("utils.jl")
include("buffers.jl")
include("attention/mods.jl")
include("attention/pair.jl")
include("attention/flex.jl")
include("attention/attention.jl")
include("attention/decode.jl")
include("softmax/softmax.jl")
include("norm/rms_norm.jl")
include("norm/layer_norm.jl")

end
