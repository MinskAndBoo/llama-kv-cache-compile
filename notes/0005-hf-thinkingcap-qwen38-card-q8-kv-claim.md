# HF card vmarcelo/ThinkingCap-Qwen3.8-27B-MIX_GGUF — same-family Q8_0-KV speed claim, hybrid-architecture numbers

Recorded at: 2026-09-26T20:00+02:00
Status: observed — re-fetched fresh and verified 2026-09-26. This is the
third source note 0002 listed as an *unverified snippet* under "Online
benchmarks". No local test has run on our stack.

Basis: https://huggingface.co/vmarcelo/ThinkingCap-Qwen3.8-27B-MIX_GGUF
(HF model card, published 2026-09-24). Mixed-tensor IQ4_XS quant (13.40 GB,
3.924 bpw, MTP head embedded) of bottlecapai/ThinkingCap-Qwen3.8-27B — a
fine-tune of Qwen3.8-27B, i.e. *the same model family as our GSQ-RCO file*,
though a different weight build. The page carries a superseded-banner
pointing to `ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF`, the official Swift
1.5 GGUF release — same GSQ-RCO lineage as our file (context only, not a
decision item).

## The practical question

Does a same-family (Qwen3.8-27B) published card back the q8_0-K direction for
our KV experiment, and is its "~10% faster prompt processing for Q8_0 KV"
claim a real measurement? Feeds the project's KV experiment; third and
final source of the verification step.

## Verification of the note 0002 snippet

**Confirmed verbatim.** The card states: "Prefer `Q8_0` KV where the
context allows. It is consistently **about 10% faster on prompt processing**
than `Q5_0`/`Q4_1` in every configuration measured here, with generation
within noise, so compressed KV is a *capacity* lever rather than a speed
one." It does cite the Anbeeld long-context benchmark, as note 0002 said.
The cited 99.9%-precision values (Q5_0/Q5_0 92.70 / Q5_0/Q4_1 92.65 /
Q4_0/Q4_0 89.84; worst-case KLD 0.099 / 0.100 / 0.130) **match exactly the
Anbeeld table already verified in note 0002** — so this card is a
confirmation of that source, not independent evidence.

## Measured (per the card: RX 9070 XT, RADV/Vulkan, 16 GB, this exact file, ~22.5K-token prompt)

- **Architecture**: only **16 of 65 blocks are full attention** (hybrid
  GatedDeltaNet), so the KV cache is small: **64 KiB/token F16, 34 KiB
  Q8_0, 21 KiB Q5_0/Q4_1**.
- **MTP is the dominant variable, not the KV type**: the draft carries its
  own KV + verification buffers, ~1.2 GiB (≈27K tokens of Q8_0 context);
  MTP off buys ~50% more context at ~27% of generation speed and slightly
  *higher* prompt throughput.
- **Key rows** (prompt / generation tok/s, GTT spill):
  - Q8_0/Q8_0 + MTP: 64K max spill-free (965.5 / 44.3); 80K mild spill (914 MiB); 96K heavy (1,624 MiB).
  - Q8_0/Q8_0, MTP off, 96K: clean (192 MiB), **1017.6 / 32.4** — best prompt throughput.
  - Q5_0/Q4_1 + MTP: 96K clean (382 MiB), 864.7 / **47.9** — best generation; 128K mild spill (1,169 MiB).
- **Spill is silent**: it still exits cleanly. ~1 GB GTT spill costs 6–9% of
  prompt throughput and 22% of generation; past ~1.6 GB it collapses
  (35–46%). "Check GTT rather than assuming a configuration fits."
- **Q4_0/Q4_0 vs Q5_0/Q4_1**: 16.7% smaller (18 vs 21 KiB/token) but the worst
  0.1% of positions diverge ~30–32% more — "about 2.8 points of tail
  precision" (92.65 − 89.84 in the cited table, verifiable). "Perplexity
  cannot see this difference. It shows up as broken JSON keys and failed
  tool calls, which is the one failure an agentic session cannot absorb."
- **Rules from the same data**: crossing V below q5 costs ~1 tail-precision
  point (steps within q5 cost ~0.5); spend bits on K first (Q5_0/Q4_0 beats
  symmetric Q4_1 at equal size); dropping V from q8_0 to q5_1 costs ~0.4
  points at −7.8% cache.
- **Sampling**: "greedy decoding can loop, so keep temperature at the
  recommended value" (BottleCap's card recommends the model-card settings —
  our live T=1.0 is in that regime).

## Interpretation for our decision

1. **The q8_0-K direction gains same-family speed evidence**: Q8_0 KV is
   measured ~10% faster on prompt processing than Q5_0/Q4_1 with generation
   within noise, on a Qwen3.8-27B card. *Transfer caveat: the card's build
   (`58367713a`) runs these pairs on native-speed kernels (Vulkan stack); on
   our build 81bc6b8, q8_0/q5_1 is not in `GGML_CUDA_FA_QUANTS` and runs the
   f16-conversion fallback, so this speed claim does not hold there until
   the pair is compiled in (note 0002, Reconciliation).* The quality half of
   the claim still transfers. Combined with notes 0002/0003, q8_0/q5_1 is
   favored on quality evidence, and on speed evidence *for a native-speed
   build* — still pending the fit check and the rebuild.
2. **Our VRAM model is now triply cross-checked**: the card's 34 KiB/tok
   (Q8_0) and 21 KiB/tok (Q5_0/Q4_1) match Anbeeld's 33.2 / 20.7 (note 0004)
   and sanmai's 34 (note 0003). Three independent sources, one number — the
   linear 172K extrapolation (q8_0/q5_1 ≈ +1.9 GiB → ≈23.2 GiB total) holds.
3. **Operational warning for the startup check**: GTT spill is *silent*
   (clean exit). When we dry-run q8_0/q5_1 on a second port, the pass
   criterion must include reading GTT from `nvidia-smi` (or the driver log),
   not just "server started and health OK" — a ~1 GiB spill would read as a
   "fit" while costing ~22% of generation.
4. **Greedy-decode loop risk (observation, feeds open question 3)**: the
   card documents that greedy decoding can loop on this family. Our live
   config (T=1.0) is fine; `start-ninfer.sh` runs **temp 0, top-k 1**
   deterministic decode — that is the documented risk configuration. If the
   user ever runs the Class-B NInfer path, expect loop-prone outputs unless
   sampling is opened up.
5. The card's Q5_0/Q4_1-vs-Q4_0/Q4_0 numbers are Anbeeld in citation form —
   confirm, don't double-count (it is not independent of note 0002).

## Limitations

- Different hardware: RX 9070 XT (16 GB, Vulkan) — the *speed* figures do
  not transfer to the 4090; the architecture/KV-size figures are
  model-derived and do transfer.
- Different weight build (ThinkingCap fine-tune, IQ4_XS-MIX 3.924 bpw) vs
  our GSQ-RCO IQ3_S (~3.5 bpw) — a family-level prior, not a measurement of
  our file.
- A model card, not a benchmark study: one card, one quant, ~22.5K prompt,
  no uncertainty reported; the quality table is cited from Anbeeld (already
  verified separately).
- The superseded-banner target (official Swift 1.5 release) was not fetched
  in this pass; it is noted only as lineage context.

## Next question

Unchanged: A/B the standard-only candidates — q5_0/q4_1 (+0.5 GiB) and
q8_0/q5_1 (+1.9 GiB, tight) — via the launchers in this package +
`scripts/ab-probe.sh` + `scripts/quality-battery.sh`. All three verification sources (note 0002's
"Online benchmarks" entries #23470, Anbeeld #2, and this card) are now
verified. The first move is the q8_0/q5_1 startup VRAM+GTT check on a second
port before touching 8080.
