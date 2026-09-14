# FlashAttention integration experiment

`tiles.jl` replaces the existing kernel's correction and epilogue. Everything
else—the softmax, QK/PV issue order, register budgets, barrier plan, and
persistent work loop—comes from `reference.jl`, a raw PTX.jl kernel ported
from pyptx (Apache 2.0; see the file header and `LICENSE`).

`comparison.jl` loads `reference.jl` into two isolated modules. Only the two
helper definitions are replaced in the Tylo module. `runtime.jl` prepares the
TMA descriptors through Tylo's `prepare_tma` and runs both kernels on the same
inputs.

## What the comparison checks

The default and quarter-publication variants assemble for SM100a, SM103a,
and SM100f. Checks compare PTX operation counts for TMEM load/store widths,
waits, fences, MMA, barrier arrivals/initialization, and global vector stores.
The kernel entry must not materialize its register tuples as local arrays.
The complete kernel machine-code section must also match byte for byte.
Debug/source metadata outside the executable section may differ.

On B200/B300 the suite also compares both implementations bit for bit and
against the reference's CPU attention calculation. Cases include strong
inputs that exercise correction, plus a single-CTA grid forcing repeated
work items through the same allocation. A separate TMEM round trip checks
FP32 storage, BF16 aliasing, packing, and raw PTX readback.

The `--bench` option adds paired CUDA-graph measurements: 32 launches per
graph, 21 measured pairs after warmup, alternating execution order.
Compilation, allocation, and CPU reference work are outside the timings.
These modest correctness shapes establish a baseline; they are not a
saturated throughput benchmark.

## Run

`run.jl` runs the comparison in one process; `--bench` adds the paired
timings. The test runner (`test/runtests.jl gpu/flash_attention`) runs the
same code-generation checks and execution cases but cannot pass `--bench`.

```sh
TYLO_EVIDENCE=/tmp/tylo-evidence julia --project=test examples/flash_attention/run.jl

# B200/B300 only:
julia --project=test examples/flash_attention/run.jl --bench
```

Evidence output contains PTX and cubins for inspection. The resource script
below assembles the PTX verbosely and records machine-code statistics; it
needs only `ptxas`, so run it at home rather than on rented time:

```sh
julia --project=test test/tools/resources.jl /tmp/tylo-evidence
```

## On a rented B200/B300

Requirements: Linux, NVIDIA driver 580 or newer (CUDA 13), one visible GPU,
network access, git, and about 10 GB of disk for the Julia depot. Copy this
checkout's `Manifest.toml` next to the clone so the rental resolves the same
package versions. Every command below is wrapped in a process timeout; a
broken asynchronous kernel can otherwise wait indefinitely.

```sh
# 1. Julia 1.13 (about 1 minute)
curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.13
export PATH="$HOME/.juliaup/bin:$PATH"

# 2. Sources and packages (5-10 minutes: CUDA artifacts download, CUDACore precompiles)
git clone https://github.com/jool-space/Tylo.jl Tylo && cd Tylo
git checkout <commit validated on GB10>
# scp the GB10 checkout's Manifest.toml into this directory first, if available
julia --project=. -e 'using Pkg; Pkg.instantiate(; workspace=true)'
julia --project=test -e 'using CUDACore; CUDACore.versioninfo()' 2>&1 | tee versioninfo.log
# Expect: CUDA runtime 13.x, the B200/B300 device, capability 10.0 or 10.3.

# 3. Gate: real TMEM allocation and round trips (about 3 minutes). Stop if red.
timeout 900 julia --project=test test/runtests.jl --jobs=1 gpu/tmem gpu/packing 2>&1 | tee gate.log

# 4. Attention: code generation, paired execution, CPU reference (about 5 minutes)
TYLO_EVIDENCE=$PWD/evidence timeout 1500 julia --project=test examples/flash_attention/run.jl 2>&1 | tee attention.log

# 5. Timings (about 5 minutes). Prints reference_us, tylo_us and ratio per case.
timeout 1500 julia --project=test examples/flash_attention/run.jl --bench 2>&1 | tee bench.log

# 6. Optional, if time remains: the whole GPU suite on real Blackwell (10-20 minutes)
timeout 2400 julia --project=test test/runtests.jl --jobs=4 2>&1 | tee suite.log
```

Bring home `versioninfo.log`, `gate.log`, `attention.log`, `bench.log`, the
`evidence/` directory and `nvidia-smi` output. A matching Compute Sanitizer
run of `run.jl` is optional and slow; the local GB10 sanitizer run covers
register arithmetic and global stores only.
