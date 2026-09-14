# Kernel text-section snapshots

Each manifest records the SHA256 of the executable sections of every kernel
the GPU suite saves through `save_code`, plus the toolchain and source
revisions that produced them. They gate refactors that promise unchanged
machine code.

Regenerate a manifest from a fresh evidence directory:

```sh
TYLO_EVIDENCE=/path/to/evidence TYLO_PTX_ROOT=/path/to/pinned/PTX \
  julia --project=test test/runtests.jl --jobs=1
julia --project=test test/snapshot.jl /path/to/evidence
cp /path/to/evidence/manifest.toml test/snapshots/<date>-<device>-tylo.toml
```

Check a working tree against a manifest. Every saved kernel present in the
manifest must have identical text sections; kernels a milestone intentionally
changes are listed as regular expressions in `TYLO_SNAPSHOT_ALLOW`:

```sh
TYLO_SNAPSHOT=$PWD/test/snapshots/2026-09-14-gb10-tylo.toml \
TYLO_SNAPSHOT_ALLOW='^rows-,^softmax-' \
  julia --project=test test/runtests.jl --jobs=4
```

A manifest is only meaningful for the Julia, CUDACore, CUDA compiler and PTX
revisions in its header. The Megakernels Hopper manifest comes from
`MEGAKERNELS_CODEGEN_OUT` during `test/runtests.jl hopper` in that
repository, fingerprinted with the same script.
