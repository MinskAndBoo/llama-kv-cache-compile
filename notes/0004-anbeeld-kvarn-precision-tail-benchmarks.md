# Anbeeld "KV Cache Quantization Benchmarks: KVarN, Precision Tail" — verified source, standard-only ladder

Recorded at: 2026-09-26T19:53+02:00
Status: observed — fetched in full and verified 2026-09-26. This is the source
note 0002 listed as an *unverified snippet* under "Online benchmarks".
Fetch caveat: the raw 413-row benchmark tables (§10) were truncated in the
fetched copy (saved locally); sections 1–9 and the recommendation ladders are
fully covered, and the visible table rows are internally consistent with the
ladder numbers quoted below. No local test has run on our stack.

Basis: https://anbeeld.com/articles/kv-cache-quantization-benchmarks-kvarn-precision-tail
(published 2026-08-11). Method (§10): BeeLlama v0.4.0 (fork), 1× RTX 3090,
Windows 11; 413 cache configurations — 238 rows on Qwen 3.6 27B Q5_K_S at
64k ctx, 175 on Gemma 4 31B Q5_K_S at 16k ctx; 48 standard K/V pairs + 15 KVarN
pairs crossed with precision tails (128/1024/2048 exact-suffix tokens), plus a
BF16-cache baseline per model. KLD computed against the matching BF16 cache.
Wikitext-2 raw, batch 2048 / micro 512, seed 1, FlashAttention, unified KV.

## The practical question

Which *standard* (vanilla-runnable) KV tiers are worth staying on or moving
to at our 172K ctx, and how much does a quantized-V or quantized-K step cost
in the worst-case tail? Feeds the project's KV experiment; complements
note 0003 (same-fork discussion evidence on Qwen3.8-27B itself).

## Verification of the note 0002 snippet

The quoted *"the same size step past q5_0 costs 70.9% [KLD], past q4_0 it
costs 130.5%"* is **confirmed — and it is Gemma-specific**: §8.3 states for
Gemma 4 31B, "the same size step past q6_0 costs 21.7%, past q5_0 it costs
70.9%, past q4_0 it costs 130.5%, and past q3_0 it costs 129.3%". The
snippet's "(Qwen cheaper)" is confirmed by §8.2: on Qwen 3.6 the same-size
step from q5_0 to q5_0-q4_1 costs **24%**, and the cliff below q4_0
(q4_0 → q4_0-q3_0, one 64 MiB step) costs **70%**.

## Measured — the standard-only Qwen ladder (§8.2, tail 0, 64k ctx)

The rows marked *fork-only* use KV types our vanilla build 81bc6b8 cannot
run under FA (q6_0 does not exist in vanilla — note 0002).

| Cache | KV MiB @64k | KiB/tok | Median KLD | 99.9% KLD | Note |
|---|---|---|---|---|---|
| bf16 | 4096 | 62.5 | 0 | 0.00005 | reference |
| q8_0 | 2176 | 33.2 | 0.000909 | 0.093029 | |
| q8_0-q6_0 | 1920 | 29.5 | 0.000937 | 0.093575 | fork-only |
| q6_0 | 1664 | 25.6 | 0.00096 | 0.091134 | fork-only; *cleanest 99.9% tail on the ladder* |
| q6_0-q5_0 | 1536 | 23.6 | 0.001054 | 0.09467 | fork-only |
| q5_0 | 1408 | 21.6 | 0.001154 | 0.09707 | "last tier before the cliff" |
| q5_0-q4_1 | 1344 | 20.7 | 0.001433 | 0.122096 | "default when VRAM-constrained" |
| q5_0-q4_0 | 1280 | 19.7 | 0.001516 | 0.121068 | "64 MiB cheaper, worse median" |
| q4_0 | 1152 | 17.7 | 0.001846 | 0.154408 | **"smallest recommended tier"** |
| q4_0-q3_0 | 1024 | 15.7 | 0.003313 | 0.218912 | below the cliff (+70% median) |
| q3_0 | 896 | 13.7 | 0.004696 | 0.304186 | |
| q2_0 | 640 | 9.8 | 0.019374 | 1.198902 | "last resort" |

Prefill stays 882.6–888.4 tok/s across all standard tail-0 rows, so this
ladder is a pure quality-vs-memory trade.

Other measured points that matter:

- **q8_0-q5_1 at tail 1024: 0.000903 median / 0.089102 99.9% KLD at 1952 MiB**
  (§10.1) — i.e. near-q8_0 quality at 224 MiB less. This is the same-fork
  validation of the q8_0/q5_1 candidate from note 0003 (which had
  Qwen3.8-27B-specific KLD data from discussion #23470).
- **q5_0-q4_1 vs q5_0-q4_0 flip by tail**: at tail 0, q5_0-q4_1 is better on
  median (0.001433 vs 0.001516); at tail 1024 they are within noise on both
  metrics (0.000966 / 0.089128 vs 0.000964 / 0.08934).
- **Bit width dominates tails** (§2): at tail 0, q2_0 0.019374 → q4_0 0.001846
  → q8_0 0.000909 median KLD. The tail then does the rescue work: q2_0 falls
  to 0.004648 at tail 1024 and 0.003696 at 2048; q8_0 moves only 0.000909 →
  0.000897. 21 of 28 Qwen Pareto-optimal rows carry a 1024- or 2048-token tail.
- **Gemma is a different animal** (§7): the shared kvarn5-kvarn4 tail-1024 row
  reads 0.036086 median KLD on Gemma vs 0.000936 on Qwen; same-top agreement
  80.5% vs 97.976%. Gemma's 50 sliding-window layers make the tail overlay
  structurally heavier (243–693 MiB vs a flat 96 on Qwen). "If sticking with
  Gemma is non-negotiable, avoid quantized KV cache whatsoever" — our Qwen
  choice is the forgiving family.
- **KVarN rows** (§3): every symmetric KVarN width beats its same-width
  standard counterpart at the matched 128-token suffix by 2.6% (6-bit) to
  12.6% (3-bit); kvarn5 sits within 0.000014 of q6_0 for 232 MiB less; kvarn6
  edges past q8_0 for 488 MiB less. Cost: ~7% slower prefill (816–822 vs
  ~886 tok/s).

## What does NOT transfer to our vanilla build

- **KVarN and the precision tail are BeeLlama fork features** (fork's
  `--kv-tail-tokens`; the third Anbeeld article frames the tail as a BeeLlama
  v0.4.0 addition). The whole §8.1 ladder is fork-only; only §8.2 is runnable
  on vanilla, and within §8.2 the q6_0 rows are also unavailable (no q6_0 KV
  in vanilla). Present in upstream llama.cpp proper: **not confirmed** —
  would need a check before treating the tail as a future vanilla option.
- The benchmark model is Qwen **3.6** with **Q5_K_S weights**, 64k ctx — a
  proxy for our Qwen3.8 GSQ-RCO (IQ4-class weights) at 172k. Note 0002's
  weight-quant interaction finding (IQ4-class weights score 3–5 points higher
  99.9% precision) means the absolute values overstate our expected loss, but
  the *tier structure and cliff positions* are the transferable part.

## Limitations

- KLD on Wikitext, not an agentic-coding evaluation — §9 says so explicitly:
  a row's KLD is evidence about distribution drift, not retained coding
  ability; the 1024-token tail optimum is corpus-shaped (Wikitext scores
  every position, so newest tokens are always in play — the article calls
  short-tail rows "synthetic readings").
- Baseline is BF16 here vs F16 in note 0003's sanmai data — the contested
  baseline from discussion #23470. Absolute values differ by baseline;
  tier orderings are the robust part.
- 64k ctx: KLD tails at our 2.6×-longer context may behave differently
  (Anbeeld's first article, note 0002, showed the IQ4_XS column dropping
  ~0.7–1.3 99.9%-precision points from 64k to 128k).

## Interpretation for our decision

1. **Our current q4_0/q4_0 sits exactly on the article's floor of the
   recommended band** ("smallest recommended tier"). The next step up is
   q5_0/q4_1, the article's VRAM-constrained default.
2. **Memory at 172,032 ctx, from the per-pair KiB/tok column** (scale ×2.625,
   validated below): q4_0/q4_0 ≈ 3.0 GiB (matches the live 21,331 MiB total),
   q5_0/q4_1 ≈ 3.4 GiB (**+0.5 GiB**, consistent with the project's
   measured VRAM table), q8_0/q5_1
   ≈ 4.8 GiB (**+1.9 GiB → ≈23.2 GiB total**: tight on the 24.6 GiB card,
   ~1.4 GiB headroom — a startup measurement, not a certainty).
   **Validation of the linear model against a measured point**: it predicts
   1,664 MiB × 2.625 = 4,368 MiB KV for the fork q6_0 row; adding that to the
   measured 21,331 − 3,024 MiB base gives 22,675 MiB vs the measured 22,660 MiB
   from the A/B — 0.07% off.
3. **This corrects note 0003's VRAM arithmetic**: discussion #23470's
   "34/29/26 KiB/tok" figures are *pair totals*, not per-tensor. The
   q8_0/q5_1 candidate is ~+1.9 GiB at 172K, not the ~+4 GiB previously
   estimated — i.e. plausibly fits, still needs the startup check.
4. **The precision tail is the interesting fork-side idea**: q4_0 body +
   tail 1024 reaches 0.001057 median KLD (≈ q5_0's tail-0), and q5_0-q4_1 +
   tail 1024 lands within noise of q8_0 on median. If the tail feature
   exists upstream in a later build, or the fork is rebuilt, a q4_0-body +
   1024-tail config could match mid-ladder quality at q4_0 memory — but
   that is a fork/rebuild project, not a flag change.
5. **Gemma's 70.9%/130.5% cliffs are not our cliffs**: on our family the
   equivalent steps cost 24% (past q5_0) and 70% (below q4_0). Citing the
   70.9/130.5 figures without the "Gemma" qualifier would mislead.

## Next question

Same as note 0003's: A/B the standard-only candidates — q5_0/q4_1 (cheap,
+0.5 GiB) and q8_0/q5_1 (+1.9 GiB, tight) — via the launchers in this
package + `scripts/ab-probe.sh` + `scripts/quality-battery.sh`. **Neither pair is in
this build's `GGML_CUDA_FA_QUANTS` (only q4_0-q4_0, q8_0-q8_0, f16-f16,
bf16-bf16 — CMakeCache confirmed 2026-09-26), so both run on the warned
f16-conversion fallback: quality clean, speed confounded (note 0002,
Reconciliation). A decision-grade speed A/B needs a rebuild with the
candidate pairs in `GGML_CUDA_FA_QUANTS` first.** The first move remains the
startup VRAM (+GTT) check for q8_0/q5_1 (second port, short run, read the
log) before touching port 8080.
