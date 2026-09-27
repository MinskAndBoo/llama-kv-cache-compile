# KV cache quant types in vanilla llama.cpp `81bc6b8`: what IQ4_NL actually is, and which types the CUDA flash-attention path can run

Recorded at: 2026-09-26T15:36:14+02:00
Status: **observed** (source-verified in the live build; no runtime test of any
KV type change has been run — the runtime-behavior claims below are inferences
from the source, flagged as such)
Basis: source tree of the live vanilla build, `~/repos/custom/llama.cpp-upstream`
(commit `81bc6b8`, 2026-09-26; the same clone our running server was built from).
All line refs are to that tree. Online sources fetched in full on
2026-09-26: Anbeeld's long-context KV-cache benchmark (whole page) and
llama.cpp issue #28455 with comments (GitHub API) — cited inline in the
Online benchmarks section. Supersedes the project's open-question-2
assumption that `iq4_nl` KV was a ready-to-run experiment ("kernel path
untested here" → now tested: the kernel path is **missing** in this build).
The project's main record for the fork-vs-vanilla decision (note 0001) is
a proxy stub in this package — the full document is private.

## Practical question

The project's KV experiment proposed two candidate configs for the
vanilla runner: `--cache-type-k q5_0 --cache-type-v q4_0` (+~0.5 GiB over
the live config) and `--cache-type-k q4_0 --cache-type-v iq4_nl` (same
size: both 4.5 bpw, same VRAM as the live q4_0/q4_0 config).
Before spending a server swap on either, the question: which `--cache-type-*`
types does this build actually *run* on the 4090, and what does IQ4_NL even
change relative to Q4_0?

## What IQ4_NL is (observed, from source)

- Layout per 32 elements (`ggml/src/ggml-common.h:447-452`): one fp16 block
  scale `d` + 16 bytes of 4-bit codes → 18 bytes = **4.5 bits/element**.
- The nibble indexes a fixed 16-entry **non-uniform** LUT
  (`ggml/src/ggml-common.h:1120-1122`):
  `-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113`.
  Level spacings shrink toward zero (23→11 on the negative side, 11→24 on the
  positive), i.e. a non-linear lattice for Gaussian-ish data — exactly what
  KV values look like. Dequant is `d * kvalues_iq4nl[nibble]`
  (`ggml/src/ggml-quants.c:2725-2741`).
- Q4_0 in this build is **byte-identical in layout**: fp16 scale + 16 nibbles
  per 32, also 4.5 bpw (`ggml-common.h:194-199`). The only difference is the
  codebook: Q4_0 is uniform symmetric (`d = max/-8`, value `d·(q−8)`,
  `ggml-quants.c:113-148`). So iq4_nl-vs-q4_0 KV is purely "non-uniform
  codebook vs uniform", at identical VRAM. (The handoff's "both 4.5 bpw"
  is confirmed against this build.)
- IQ4_NL is in the accepted `--cache-type-k/v` allowlist
  (`common/arg.cpp:304-314`: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0,
  q5_1).

## CUDA flash-attention support per KV type (observed, from source)

The live server runs with `--flash-attn on`; with quantized KV, that is the
only attention path that matters:

- FA's supported KV type list is F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
  — **not** IQ4_NL
  (`ggml/src/ggml-cuda/fattn.cu:525-539`, `ggml_cuda_fattn_kv_type_supported`).
- No FA vector-kernel case instantiates any IQ4_NL pair
  (`fattn.cu:437-491`).
- An unsupported type is not a slow fallback: dispatch returns
  `BEST_FATTN_KERNEL_NONE` → `GGML_ABORT("fatal error")` on first attention
  (`fattn.cu:625-627`, `fattn.cu:755-769`). *(Inference from source; the
  crash itself has not been observed — see Limitations.)*
- A quantized **V** cache is additionally forced through FA
  (`src/llama-context.cpp:3793-3801`): FA auto-enabled if `auto`, hard error if
  explicitly disabled. A quantized **K** cache only needs a divisibility check
  (block 32 | head 256, fine for Qwen3.8: `n_embd_head_k/v = 256` per the
  model's startup banner, captured from the server log) —
  `llama-context.cpp:3804-3813`.
- You cannot compile the missing kernel in: CMake's FA-quant allowlist is
  `q4_0 q4_1 q5_0 q5_1 q8_0 bf16 f16`; listing `iq4_nl` in
  `GGML_CUDA_FA_QUANTS` is a `FATAL_ERROR: unknown type`
  (`ggml/cmake/common.cmake:53-77`).

## Reconciliation: what this build actually does per KV type

Dispatch support (above) is commit-level truth; whether a given (K,V) pair
is *compiled in* is decided by CMake, and this build is configured with
`GGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`
(`build/CMakeCache.txt:687`). Only those pairs (plus always-compiled
f16-f16, `ggml/cmake/common.cmake:74`) get a dedicated vec kernel; every
other supported pair hits the warned f16-conversion fallback
(`fattn.cu:501-514`). Net behavior on this box:

- **q4_0/q4_0 (live config), q8_0/q8_0, f16/f16, bf16/bf16**: compiled in,
  native quant FA.
- **q4_1 / q5_0 / q5_1 in either slot** (e.g. q5_0/q4_0, q4_1/q4_0,
  q5_1/q5_1): accepted by `supported()`, stay on the GPU — but the pair
  kernel is not compiled in, so K and V are converted to f16 for **every**
  attention op, with a one-time log warning ("no FlashAttention vector
  kernel compiled for K/V types … Add … to GGML_CUDA_FA_QUANTS",
  `fattn.cu:502-511`). Decode takes the VEC path (`fattn.cu:649-652`),
  prefill the MMA_F16 path (`fattn.cu:664`); both convert. *(Estimate: f16
  K/V at 172K×256 is ~88 MiB per tensor vs ~22 MiB for q4_0 — the
  conversion path moves several times more bytes per attention op than the
  native quant path, so expect a real speed penalty.)*
- **iq4_nl**: still excluded from `supported()` → abort, unchanged.
- Consequence for issue #28455: the **silent CPU fallback does not apply to
  this build** — it is a symptom of builds whose `supported()` excludes
  these types (the issue's b10791/b10819). Our commit is newer and routes
  them to the warned f16 path instead. *(Inference from source + build
  config; the runtime confirmation is the one-time warning line, which
  appears on first attention of any such config.)*
- The startup banner prints the compiled set as a backend feature
  (`ggml-cuda.cu:5737-5738`, `FA_QUANTS`); the current live log has no
  banner section, so confirm on the next startup.
- To run q5_0/q4_0 **natively** (no conversion), rebuild with the pair
  added — CMake allows it (its allowlist includes q4_1 q5_0 q5_1,
  `common.cmake:54`; iq4_nl is not):
  `-DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;q5_0-q4_0"`.

## Online benchmarks (searched 2026-09-26; entries marked verified/unverified)

Context: a local needle probe cannot detect few-pp differences (needs ~100s of
replicates), so published/online evidence is the practical source. Search
turned up, in priority order:

- **Anbeeld, "KV Cache Quantization Benchmarks for Long Context"**
  (`anbeeld.com/articles/kv-cache-quantization-benchmarks-for-long-context`,
  2026-05-22) — the source cited in the project's main record (see the
  note 0001 proxy stub). **Fetched in full and
  verified 2026-09-26.** Qwen 3.6 27B, RTX 3090, BeeLlama v0.1.2 (a
  llama.cpp fork; the standard q4/q5/q8 rows run the normal llama.cpp
  flow, so they transfer to vanilla — the turbo/TCQ rows are fork-only and
  do not). PPL is flat across the board (<0.01 delta vs bf16 through q4_0);
  the discriminating metric is **99.9% precision** = `100·exp(−(KLD_quant
  − KLD_bf16))`, i.e. the worst 0.1% of token positions.

  99.9% precision vs bf16, standard rows only:

  | Cache (K / V) | % of bf16 KV | Q5_K_S 64k | IQ4_XS 64k | IQ4_XS 128k |
  |---|---|---|---|---|
  | q8_0 / q8_0 | 53.1 | 94.61 | 98.69 | 98.52 |
  | q8_0 / q5_1 | 45.3 | 94.21 | 98.28 | 98.13 |
  | q5_1 / q5_1 | 37.5 | 92.77 | 97.46 | 97.70 |
  | q5_0 / q5_0 | 34.4 | 92.70 | 97.32 | 97.30 |
  | q5_1 / q4_1 | 34.4 | 93.08 | 96.58 | 96.56 |
  | q5_0 / q4_1 | 32.8 | 92.65 | 96.21 | 96.50 |
  | q5_1 / q4_0 | 32.8 | 91.82 | 96.20 | 96.03 |
  | q4_1 / q4_1 | 31.3 | 88.82 | 93.73 | 95.04 |
  | q5_0 / q4_0 | 31.3 | 91.39 | 96.06 | 96.18 |
  | q4_0 / q4_0 | 28.1 | 89.84 | 93.01 | 94.34 |

  Key findings (verified from the article, §5/§7/§8/§9/§10):
  - **Symmetric q4_1 is a bad deal**: same 31.3% budget as q5_0/q4_0 but
    strictly worse in all three tables (88.82/93.73/95.04 vs 91.39/96.06/
    96.18). Spending the same bits asymmetrically toward K wins.
  - **The next bit after q5_0 K goes to V, not to q5_1 K**: q5_0/q4_1
    (32.8%) beats q5_1/q4_0 (32.8%) on Q5_K_S 64k (92.65 vs 91.82) and
    IQ4_XS 128k (96.50 vs 96.03); IQ4_XS 64k is a tie (96.21 vs 96.20).
    "q5_1 remains useful as an extra-conservative option, but it is not
    where the marginal value is."
  - **V-tier cliffs**: stepping V within q5 costs ~0.5 pts per tier;
    crossing V below q5 costs a full point (e.g. q8_0/q5_0 93.69 →
    q8_0/q4_1 92.70, Q5_K_S 64k).
  - **Preset ladder verdicts**: q8_0/q5_1 = best precision/size at the high
    end; q5_0/q4_1 = "best default if VRAM-constrained"; q5_0/q4_0 =
    fallback when q5_0/q4_1 doesn't fit by a hair; q4_0/q4_0 = "memory
    saving with precision loss".
  - **Weight-quant interaction** (§9): the same cache preset scores 3–5
    points *higher* 99.9% precision on IQ4_XS weights than on Q5_K_S
    weights (smoother KV distributions have less to lose). Our GSQ-RCO
    3.5 bpw weights are in the IQ4_XS class, so the IQ4_XS column is the
    better proxy for our model.
  - **His §10 corroborates our statistical-power point**: 32k
    passkey/needle tests scored 100% for *every* cache type (even turbo2);
    30-problem AIME was "wildly noisy" with bf16 and q4_0 tied. Small
    retrieval tests cannot separate these types — only long-context KLD
    tails can.
- **llama.cpp issue #28455, "quantized KV cache (q5_x, q4_x) with -fa on
  silently fallback CPU attn"** (opened 2026-09-05, `bug-unconfirmed`, closed
  2026-09-06 by a maintainer as a *known issue*; the report was subsequently
  banned under the repo's AI-usage policy — no fix PR is referenced).
  **Verified via GitHub API, 2026-09-26.** Repro: **Qwen3.8-27B** (our model
  family), `-fa on`, `--cache-type-k/v q5_1`, default build b10791
  (`95ef7fc16`); reporter verified still present on master `6a1a922d2`
  (b10819).
  - Mechanism: in a default build, `supported()` returns false for
    Q4_1/Q5_0/Q5_1 → `BEST_FATTN_KERNEL_NONE` → the CUDA backend rejects the
    FA op and the scheduler **silently moves attention to CPU** (no warning;
    GPU 0%/62 W, CPU ~930%). The tile kernel reads K/V as half2 only, and
    `-fa 0` with quantized V fails outright — so those types have **no
    working GPU path** in such builds.
  - Cost on their RTX 4080 (pp512, `-fa 1 -t 12 -ngl 99`): q8_0/q8_0
    1511.9 t/s → q8_0/q5_1 197.1 (7.7x slower), q5_1/q5_1 196.6 (7.7x);
    llama-server, 3.6K prompt: pp 1348 → 38.4 t/s, 3 s TTFT → ~95 s.
    `llama-bench` still prints `fa=1 q5_1` while actually measuring CPU
    attention + GPU↔host copies — results mislabeled.
  - Explains #27109 (q4_1 K-cache prefill collapse, qwen35 hybrid): Q4_1 sits
    in the same unsupported branch as Q5_0/Q5_1. Related: #24485 (FR: warn
    loudly; closed stale), #20866 (mixed K/V offload), #27905 (RPC + FA fatal
    error, same matrix).
  - Workaround: rebuild with `-DGGML_CUDA_FA_ALL_QUANTS=ON` → pp512
    1509.7 t/s, server pp 1162 / tg 38.5 (vs 38.4/6.5 default).
  - **Build-specific — see reconciliation below**: the issue's symptom is
    for builds whose `supported()` excludes these types. Our pinned commit
    is newer and behaves differently.
- **llama.cpp discussion #23470, "Asymmetric KV q8/q4"** (2026-05-21) —
  downstream 500×ARC-Challenge deterministic test, F16 vs q8_0/q4_0/K-only/
  V-only. **Fetched in full and verified 2026-09-26** — the churn test
  exists and is richer than the snippet: q4_0 K alone reproduces the full
  collapse on the sensitive model; q4_0/q4_0 churns only 3/500 on
  Qwen3.8-27B. Plus a Qwen3.8-27B 99% KLD matrix (q8_0/q5_1 favored;
  q8_0/q4_0 rejected). Full record: note 0003.
- **Anbeeld, "KV Cache Quantization Benchmarks: KVarN, Precision Tail"**
  (second article, 2026-08-11,
  `anbeeld.com/articles/kv-cache-quantization-benchmarks-kvarn-precision-tail`)
  — Gemma vs Qwen KV quant cost comparison. **Fetched in full and verified
  2026-09-26**: the quoted "70.9% past q5_0 / 130.5% past q4_0" are
  **Gemma 4 31B** numbers; on Qwen the same-size step past q5_0 costs 24%
  and the cliff below q4_0 is 70%. KVarN and precision-tail rows are
  BeeLlama-fork-only; the transferable part is the standard-only ladder.
  Full record: note 0004.
- **vmarcelo/ThinkingCap-Qwen3.8-27B-MIX_GGUF** (HF, 2026-09-24) — card for
  the *same model family*. **Re-fetched fresh and verified 2026-09-26**:
  the "Prefer Q8_0 KV where the context allows … about 10% faster on prompt
  processing than Q5_0/Q4_1, with generation within noise" quote is
  confirmed verbatim; its 99.9%-precision table matches the verified
  Anbeeld numbers exactly (confirmation, not independent evidence). Also
  carries: 16-of-65 full-attention architecture (64/34/21 KiB/tok F16/Q8_0/
  Q5_0-Q4_1), silent-GTT-spill warning, and a greedy-decode loop warning.
  Full record: note 0005.
- Cautionary (different stack): vLLM April-2026 128k needle BF16 91% → FP8
  13% collapse from Hopper FP8 accumulation, recovered by two-level
  accumulation (via particula.tech, 2026-08-02) — long-context KV quant
  failures can be silent and hardware-specific.

## Consequence (interpretation)

- `--cache-type-v iq4_nl` (and `--cache-type-k iq4_nl` with FA on) is a
  **dead end in build `81bc6b8`**: startup succeeds, first attention aborts.
  Making it usable is a source-patch job (FA kernel support + CMake allowlist
  entry), not a flag. Deprioritize until upstream has it.
- The q5_0-K/q4_0-V experiment (handoff open question #2) is runnable on
  this box today — with the caveat from the Reconciliation: in the current
  build it runs GPU FA via the **f16 conversion fallback** (one-time log
  warning; K/V dequantized to f16 on every attention op). +~0.5 GiB is
  still inside the ~2.5 GiB headroom at 172K ctx. So a first A/B measures
  quality *and* the conversion speed penalty together: run it as-is if
  quality is the only question (speed numbers are
  guardrails), or rebuild first with the `q5_0-q4_0` pair added to
  `GGML_CUDA_FA_QUANTS` for a native-speed run.
- The quality prior is now much stronger than before (Anbeeld, verified):
  q5_0/q4_0 (31.3% of bf16 KV) beats symmetric q4_1 in all three tables
  (e.g. 91.39 vs 88.82, Q5_K_S 64k; 96.18 vs 95.04, IQ4_XS 128k) and trails
  only ~0.3 pt behind the larger q5_0/q4_1 (96.18 vs 96.50 at IQ4_XS 128k).
  The IQ4_XS column — 3–5 pts better tails than Q5_K_S, the better proxy for
  our GSQ-RCO 3.5-bpw weights — has q5_0/q4_0 leading q4_0 by ~3 pts at both
  contexts (96.06 vs 93.01 at 64k, 96.18 vs 94.34 at 128k).
- The "runnable today" set for this box is now graded by speed, not just
  runnability (see Reconciliation): **native-speed** pairs = q4_0/q4_0,
  q8_0/q8_0, f16/f16, bf16/bf16 (the only compiled-in FA pairs); **supported
  but f16-converted** (one-time warning, expect a speed penalty) = any pair
  with q4_1, q5_0, or q5_1 in either slot; **abort** = iq4_nl. f32 KV is
  VRAM-implausible at 172K (even q8_0 V OOMs there, handoff gotcha #3), so in
  practice the candidates are: baseline q4_0/q4_0, then q5_0/q4_0,
  q4_1/q4_0, q5_0/q4_1 — each either run converted or rebuilt-in natively.

## Limitations

- No runtime test was run: "aborts on first attention" (iq4_nl) is inferred
  from the dispatch code, "quantized V requires FA" from the param check,
  and the f16-conversion path for q4_1/q5_0/q5_1 from source + build config.
  The abort claim is cheap to falsify but not worth it (it would crash this
  session's own runtime — handoff gotcha #4). The conversion path has a free
  runtime check: the one-time warning line on first attention of any such
  config.
- The FA support list and FA_QUANTS config were verified for commit
  `81bc6b8` only. Rebuilds must re-check `fattn.cu:525` (the supported
  list), the CMake allowlist, and the compiled pairs (handoff gotcha #7) —
  and #28455 shows this behavior moves between builds: the silent CPU
  fallback existed on b10791/b10819 but not on our commit.
- The online numbers are from other hardware/models: Anbeeld ran Qwen 3.6
  27B on an RTX 3090 (forked runner); the issue's timings were on an RTX
  4080. They are priors on the *ordering* of cache types, not measurements
  on our box.
- Nothing here measures *quality on this box* — and the local battery
  cannot settle it either: needle-style probes score 100% for *every*
  quantized KV type (Anbeeld §10), so the A/B in "Next question" can only
  exclude gross failures and measure speed. Quality ordering is prior-only
  (the published tables) until a powered instrument — high-replicate
  non-ceiling probes, or a KLD/perplexity harness against a reference —
  is run.

## Next question

Does `--cache-type-k q5_0 --cache-type-v q4_0` beat the live q4_0/q4_0?
Be clear about what the local run can answer: the battery can only
exclude gross failures (needle probes score 100% for every quantized KV
type, so it cannot rank the configs), and the PP/TG probe measures speed
(under the f16-conversion penalty, as-is). The quality ranking itself
rests on the verified priors below — settling it locally would need a
powered instrument (high-replicate non-ceiling probes, or a KLD/perplexity
harness), not this battery. Per those priors q5_0/q4_0 is the top
candidate (it dominates same-size symmetric q4_1 and trails q5_0/q4_1 by
~0.3 pt at the IQ4_XS 128k tail); q4_1/q4_0 and q5_0/q4_1 are the
alternates if a powered run disagrees.

Mechanics, in order:
1. Pass separate K and V to the launcher — this package's
   `start-llama-candidates.sh` already takes `K` and `V` independently.
2. Run mode: **as-is first** is defensible for the gross-failure and
   speed questions — the f16 conversion is an exact dequant of the cached
   values, so attention sees the same numbers an f16 KV cache would (only
   PP/TG *speed* is
   contaminated by the conversion traffic). *(Inference from source; the
   one-time warning line in the log confirms the path is taken.)* Rebuild
   with the `q5_0-q4_0` pair in `GGML_CUDA_FA_QUANTS` only if clean speed
   numbers matter.
3. Launch the vanilla build for comparison, then `scripts/ab-probe.sh` +
   `scripts/quality-battery.sh` at 172K on both configs.

Observations to record: battery pass rate and needle/tool-call scores at
25%/90% depth on both configs (gross-failure detection only — cannot rank
the configs), the PP/TG probe numbers (speed, with the conversion penalty
as-is), and the presence of the FA conversion warning in each run's log.
