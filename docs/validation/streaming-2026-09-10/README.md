# GB10 streaming attention checkpoint

The implementation is Tylo `60dda118bfc6d65a3b5b0b72fcba8c3d728eb68f`.
See `../../src/validation.md` for results and limitations. `attention.toml` and
`softmax.toml` retain all paired samples; resource tables distinguish offline
assembly from runtime kernel attributes. `validation.json` records commands,
exit statuses and sanitizer summaries. Source hashes identify both Julia repos
and the pinned PTX snapshot. `artifacts.json` is the receipt for the larger local
archive in `reports/streaming-attention-2026-09-10/`; binaries are not checked in.

The main result is a tested streaming dataflow and two reusable primitives.
Performance wins are shape-dependent. No new Hopper/TMEM hardware validation
or comparison against a tuned FlashAttention implementation is claimed.
