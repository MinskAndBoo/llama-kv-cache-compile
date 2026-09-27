# llama.cpp discussion #23470 — asymmetric KV (q8/q4): KLD + downstream answer-churn evidence

Recorded at: 2026-09-26T19:47+02:00
Status: observed — fetched in full and verified 2026-09-26; this is the source note 0002
listed as an *unverified snippet* under "Online benchmarks"
Status detail: all figures below were read directly from the fetched page on this date;
no local test has run on our stack.

Basis: https://github.com/ggml-org/llama.cpp/discussions/23470 (sanmai, opened
2026-05-21; comments through 2026-09-17). Fetched 2026-09-26.

## The practical question

Which KV quant split for Qwen3.8-27B at 172K ctx on one 24 GB card keeps the
worst-case tail quality of the f16/bf16 reference, and is 4-bit tolerable on K,
on V, or both? This decided the next step of the project's KV experiment
(we ran q4_0/q4_0).

## What the discussion contains (measured, per the page)

**1. sanmai's KLD/PPL probe (2026-05-21), qwen2.5-7b-q4_0, llama-perplexity,
FA on, base commit 0be8468, ctx 512, vs f16 reference.**
- `-ctk q4_0 -ctv q4_0`: mean KLD 5.509, same-top-p 11.6% — "completely out of
  touch with f16".
- `-ctk q4_0 -ctv f16`: mean KLD 5.487, same-top-p 11.8% — equally broken;
  "is it all in the K?"
- `-ctk f16 -ctv q4_0`: mean KLD 0.0040, same-top-p 96.9% — "V at 4-bit costs
  nearly nothing".
- `-ctk q8_0 -ctv q4_0`: mean KLD 0.0048, same-top-p 96.7%.
- `-ctk q8_0 -ctv q8_0`: mean KLD 0.0018, same-top-p 98.0%.
- Caveat raised in-thread (jamesb6626, 2026-05-22): Qwen2.5 is an older model
  and current Qwen 3.5/3.6 baselines are bf16, not f16.

**2. Downstream answer-churn test (chambejp, first post 2026-08-16; rerun
2026-09-12): 500 ARC-Challenge, grammar-constrained deterministic decode,
fixed seed, F16-cache baseline vs q8_0 / q4_0 / K-only / V-only; rep2
baselines byte-identical; metric = how often the final answer differs from
the F16 run.**
- First run (4 models): q8_0 K+V changed ≤4/500 (0 on two models), and 0 flips
  across 1,600 deep-context comparisons to 41K tokens. q4_0 K+V is strongly
  model-dependent: Qwen3.6-27B 2/500, **Qwen3.8-27B 3/500**, Mistral-7B 34/500,
  Qwen2.5-7B 375/500 (92.0% → 24.2%, below chance). On the collapsing model,
  **q4_0 on K alone reproduced the full collapse, while q4_0 on V alone
  changed 1/500**. Harness: https://github.com/firstpartyworks/churn-harness.
- Rerun (2026-09-12, 3 models, changed answers / 500):
  - q8_0/q8_0: Qwen2.5-7B 0, Mistral-7B 4, Qwen3.6-27B 1
  - q8_0 K / q5_1 V: 0 / 5 / 1
  - q8_0 K / q5_0 V: 1 / 11 / 1
  - q8_0 K / q4_0 V: 1 / 10 / 1
  - q5_1/q5_1: 83 / 11 / 1
  - q5_1 K only: 84 / 10 / 1
  - q5_1 V only: 0 / 3 / 1
  - q4_0/q4_0: 375 / 34 / 2

**3. sanmai's 99% KLD (wikitext-2, 65,536 ctx, Qwen3.8-27B UD-Q6_K — our model
family; posted 2026-09-12):**

| K / V | bpw | KiB/tok | 99% KLD |
|---|---|---|---|
| q8_0/q8_0 | 8.50 | 34 | 0.006425 |
| q8_0/q5_1 | 7.25 | 29 | 0.007628 |
| q8_0/q4_0 | 6.50 | 26 | 0.012340 |

His ranking (2026-09-11/12): q8_0/q8_0 ≈ bf16/f16 > q8_0/q5_1 ≈ q8_0/q5_0 >
q5_1/q5_1 > q8_0/q4_0 > q4_0/q4_0. "q8_0/q4_0 is almost twice as bad while
saving only 24% of cache … let's not do q8_0/q4_0." If exactly one asymmetric
pair may be added to a default build, his pick is **q8_0/q5_1**.

**4. Full 7×7 99% KLD matrix (same model/ctx, posted 2026-09-17).** Row = V,
column = K. Diagonal neighbors that matter for us:

| V \ K | f16 | q8_0 | q5_1 | q4_1 | q4_0 |
|---|---|---|---|---|---|
| f16 | 0.000037 | 0.006311 | 0.008195 | 0.014851 | 0.016561 |
| q8_0 | 0.006390 | 0.006425 | 0.008404 | 0.014597 | 0.016839 |
| q5_1 | 0.007332 | 0.007628 | 0.009367 | 0.016434 | 0.017557 |
| q5_0 | 0.007589 | 0.007670 | 0.009593 | 0.016230 | 0.019095 |
| q4_1 | 0.011663 | 0.011538 | 0.013022 | 0.019584 | 0.021870 |
| q4_0 | 0.012169 | 0.012340 | 0.014621 | 0.020171 | 0.022060 |

Reading: at fixed K, stepping V below q5 costs far more than the q5-internal
step (q8_0 K: f16 V → q5_1 V is +19%, q5_1 V → q4_0 V is +98%); at fixed V=q8_0,
K q8_0 → q5_1 → q5_0 → q4_0 climbs 0.0064 → 0.0084 → 0.0088 → 0.0168. K is the
steep axis — consistent with the Anbeeld "spend bits on K first" finding in
note 0002, and now measured on our model family rather than a proxy.

## Limitations / open confounders

- **Baseline debate unresolved in-thread**: kapojko (2026-09-16) asks why F16,
  not BF16, is the reference for Qwen3.8 (trained with BF16); sanmai (09-17)
  answers f16/f16 is the default KV dtype, has 10 vs 7 mantissa bits, and
  "could just as well be a bug in the bf16 path" — "we can settle it is just
  not the default". All of sanmai's numbers sit on an f16 baseline; treat the
  absolute values as f16-relative, the ordering as the robust part.
- **KLD tail vs downstream answers diverge on Qwen3.8**: the KLD matrix says
  q4_0 K is ~2× q8_0 K on 99% KLD, yet the churn test puts q4_0/q4_0 at only
  3/500 changed answers on Qwen3.8-27B (our family) and q8_0 K / q4_0 V at 1/500.
  KLD tails are necessary but not shown to be sufficient for answer quality on
  this family; the two instruments do not rank identically.
- Qwen2.5-7B is the fragile case in both KLD and churn; it is *not* our family.
- The churn harness is a single third-party implementation
  (firstpartyworks/churn-harness); ARC-Challenge is a static, possibly
  overfitted benchmark.

## Interpretation (ours, not the discussion's)

For the project's KV experiment, this discussion adds:

1. **q8_0/q5_1 is now a first-class candidate, alongside q5_0/q4_1 and
   q5_0/q4_0** — Anbeeld (proxy, Qwen 3.6) favored q5_0/q4_1 as the
   VRAM-constrained default; sanmai's same-family (Qwen3.8-27B) 99% KLD
   matrix puts q8_0/q5_1 clearly above q5_1/q5_1 and far above any q4_0
   combination, at 7.25 bpw. *Speed caveat (confirmed in the live build's
   CMakeCache, 2026-09-26): `GGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;
   bf16-bf16` — none of these three pairs is compiled in, so all three run
   GPU FA via the warned f16-conversion fallback (note 0002,
   Reconciliation). Quality evidence transfers; speed claims do not —
   including the HF card's ~10% pp for Q8_0, which was measured on a
   native-speed stack.*
2. **q4_0 on K is the one axis to keep off** — KLD says so on our family,
   churn says so on the sensitive models; the current q4_0/q4_0 is the
   configuration both instruments flag as the weak point (even though
   Qwen3.8 churned only 3/500 at answer level).
3. **VRAM arithmetic for 172,032 ctx** — *corrected 2026-09-26 after
   note 0004*: the discussion's KiB/tok figures (q8_0/q8_0 34, q8_0/q5_1 29,
   q8_0/q4_0 26) are **pair totals, not per-tensor**, and Anbeeld's standard
   rows confirm them (q8_0-q5_1 = 28.9 KiB/tok at 64k). Scaled to 172,032
   tokens: q4_0/q4_0 ≈ 3.0 GiB (consistent with the live 21,331 MiB total),
   q5_0/q4_1 ≈ 3.4 GiB (+0.5 GiB), q8_0/q5_1 ≈ 4.8 GiB (**+1.9 GiB → ≈23.2
   GiB total on the 24.6 GiB card: tight, ~1.4 GiB headroom**). The linear
   model is validated against one measured point: it predicts 22,675 MiB for
   the fork q6_0 row, measured 22,660 (0.07% off). **Still confirm against an
   actual server startup before committing to a swap** (operational gotcha:
   a swap kills the serving session — see AGENTS.md, gotcha 4).
4. The f16-vs-bf16 baseline question matters only if we later build a local
   KLD harness (open question 1): if we do, the reference choice should be
   documented, and a bf16 sanity row added, because it is contested upstream.

## Next question

Which candidate pair — q5_0/q4_1 (cheap, Anbeeld-favored), q8_0/q5_1
(same-family KLD-favored, VRAM-tight), or q5_0/q4_0 (Anbeeld tail-favored) —
do we A/B with the existing `scripts/ab-probe.sh` + `scripts/quality-battery.sh`?
**Speed confound (CMakeCache confirmed 2026-09-26)**: in build 81bc6b8 the
only compiled-in FA pairs are q4_0-q4_0 and q8_0-q8_0 (plus f16/bf16), so all
three candidates run on the warned f16-conversion fallback. Quality
measurement is clean on that path (the cache stores the quantized values and
the kernel dequantizes them exactly — only speed changes); PP/TG there are
"quantization + conversion" numbers, not the candidate's operating speed. A
decision-grade speed A/B requires a rebuild with the candidate pairs in
`GGML_CUDA_FA_QUANTS` (CMake allowlist `q4_0 q4_1 q5_0 q5_1 q8_0 bf16 f16`
covers all three; their `.cu` template instances exist, checked 2026-09-26).
The answer to "does q8_0/q5_1 fit at 172K" is a startup-time measurement, not
a calculation; the first move is a dry startup (or a short run on a second
port) that records actual VRAM (and GTT) from the server log.
