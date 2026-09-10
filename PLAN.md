# Streaming attention batch complete

Implementation: `60dda11`. See [the validation checkpoint](docs/src/validation.md)
and [the executed plan](plans/2026-09-10-streaming-attention.md).

Delivered online row state, checked accumulator-to-A conversion, fixed-capacity
wide-row softmax and a complete masked/causal BF16 attention example on GB10.
Both packages pass their full suites and all six sanitizer runs. The bounded
projection candidate was measured and rejected; the decode scheduler and
split-attention merge were left unchanged.

The next useful batch is performance work on this concrete consumer: compare
smaller query CTAs, overlap K/V copies, and examine the remaining register and
addressing costs. Keep D=64, runtime lengths, numerical references and paired
measurements. Add reusable machinery only when the measured schedules need it.

H100/H200 WGMMA and B200/B300 TMEM execution remain hardware-pending. No further
instruction family or generic layout/redistribution engine is required to make
progress on GB10.
