# Recompiling KV cache

Once upon a time I thought the only K/V cache options I could get from llama.cpp 
on a 4090 were 4_0 / 4_0 or 8_0 / 8_0 and that's a pretty punishing choice 
when it comes to selecting your context size vs quality tradeoff.

There's all sorts of (apparent) misinformation around about symmetric vs 
assymetric or the 'oddball' types like 4_1 or 5_1 dropping to CPU at 30x speed
reduction. Well, dear reader, point thine agent at this repo and have an infinite
number of choices... actually not infinite, only 7^2, one for each pair of
[q4_0 q4_1 q5_0 q5_1 q8_0 bf16 f16] quants. No, I don't have IQ(X)_NL yet, 
maybe another day.

Rule of thumb is K>V, and using the glorious Qwen3.8 27B GSQ-RCO IQ3_S MTP (with
vision and a couple of Youtube videos) I can run a solid ~140K context at 
k = q8_0 / v = q5_1 and broken tool calls have magically disappeared.

My model thinks you want to read what they wrote below in excruciating detail.
Hey, don't blame me, Qwen3.8 gonna Qwen3.8. O_o

fwiw this is C0: Public Domain, but a shoutout is appreciated if you find it 
handy.

## Make any quantized KV pair run at native flash-attention speed in llama.cpp

**TL;DR — the whole trick is one CMake flag.** The CUDA flash-attention (FA)
kernel is compiled in per K/V quant pair. The stock build ships only
`q4_0-q4_0; q8_0-q8_0; f16-f16; bf16-bf16`. Every other legal pair *works* at
runtime but dequantizes to f16 and pays a conversion penalty (one warning per
launch, quality untouched). If you add your pair to `GGML_CUDA_FA_QUANTS` and
rebuild, it runs on the native FA kernel — and in our measurements it was
**indistinguishable from q4_0 prefill speed**. That is the entire procedure.
No source changes, no new types, no heroics.

**The surprise.** There is a lot of conflicting noise out there: *will an
asymmetric pair (q8_0/q4_0, q5_0/q4_1) silently fall back to CPU? will q4_1
"go up in a ball of flames"?* The real answer, verified live on a 24 GB RTX
4090 against a real 27B model: **nothing bad happens.** Prefill on
q8_0/q5_1 measured flat within ~5% of q4_0/q4_0, with no fallback warning, no
quality failure, and the whole model + KV cache still fitting in VRAM.
llama.cpp buries this lede — the flag isn't in the main README, its canonical
statement is a row in `docs/build.md`, and the default list sits exactly two
brackets (`4-4` and `8-8`) around the whole sweet spot of asymmetric pairs
people actually want.

## What actually happens at runtime

Three states for a K/V pair on a CUDA FA build:

| Pair status | Behavior |
|---|---|
| **Compiled in** via `GGML_CUDA_FA_QUANTS` | Native FA kernel. Fast. |
| **In the type allowlist** (7 types: `q4_0 q4_1 q5_0 q5_1 q8_0 bf16 f16` → 49 ordered pairs) but **not compiled in** | GPU FA via the **warned f16-conversion fallback**. The cache stores the quantized values; the kernel dequantizes them *exactly* — **quality is identical**, speed pays the conversion cost. One `W` log line per launch. |
| **Not in the allowlist at all** | `FATAL_ERROR: unknown type` — the server refuses to start. |

Two consequences worth internalizing:

- **The rebuild is a pure speed/bandwidth matter.** Quality is unaffected on
  both fast paths, so the KV choice is free to be driven by quality priors
  and VRAM fit — the two things people actually care about.
- **`f16-f16` is always compiled**, even if you omit it from the list. That is
  the mechanical reason the warned fallback always exists and always works.

## The procedure (in one breath)

Shallow-clone llama.cpp at a pinned commit, add your pair to the
`GGML_CUDA_FA_QUANTS` CMake list, rebuild, start the server with
`--flash-attn on --cache-type-k <K> --cache-type-v <V>`, confirm there is no
FA-fallback warning at startup, probe the speed. The exact commands, the
validation checklist with pass/fail criteria, and the gotchas are in
**[AGENTS.md](AGENTS.md)** — it is written so an agent (or a tired human) can
run it verbatim.

## What we measured (RTX 4090, 24.6 GiB, Qwen3.8-27B, 143,360 ctx)

Four pairs, one probe per pair, plus an n=11 replicate on the tightest pair
(q8_0/q5_1):

| K/V | Prefill tok/s | VRAM |
|---|---|---|
| q4_0/q4_0 | 2529 | 21.0 GiB |
| q5_0/q4_0 | 2437 | 21.2 GiB |
| q5_0/q4_1 | 2409 | 21.4 GiB |
| **q8_0/q5_1** | 2475 | **22.4 GiB — still fully on GPU** |

- **Prefill is flat within ~5% across all four pairs** — the "crippled by
  fallback" scenario did not materialize for any compiled-in pair.
- **Decode tok/s is not a KV-speed signal on this stack**: it tracked
  speculative-decoding (MTP) acceptance noise at r = 0.996 across replicas.
  Judge KV pairs on prefill + VRAM, not decode.
- **Quality: no gross failures at any rung** — needle-in-haystack at 25% and
  90% of a ~99K-token context, a tool-calling round, and a vision check all
  passed identically across pairs and across 11 replicas. Honest caveat: this
  battery *detects* catastrophic failures; it cannot *rank* KV types
  (needle-style probes score 100% for every quantized KV). The local quality
  ordering is prior-only — the published priors (below) favor keeping K off
  q4_0, with q8_0/q5_1 on top among the measured pairs.

## Where the online noise comes from (and why it contradicts itself)

The failure stories are real — for **older builds** and **uncompiled pairs**:

- [#20866](https://github.com/ggml-org/llama.cpp/issues/20866) — asymmetric
  q8_0/q4_0 on an RTX 3090 Ti: 30.6 t/s (CPU fallback) vs 1340 t/s native.
  Pre-dates the current fallback behavior; the exact failure mode a
  compiled-in list eliminates.
- [#24485](https://github.com/ggml-org/llama.cpp/issues/24485) — measured 96 →
  2361 tok/s (~25×) with all quants compiled in; closed *not planned* —
  upstream will not make `=all` the default, so the explicit list is the
  sanctioned path.
- [#28455](https://github.com/ggml-org/llama.cpp/issues/28455) — silent CPU
  fallback with q5_x on a default build; also older builds.
- Discussion [#23470](https://github.com/ggml-org/llama.cpp/discussions/23470)
  and [Anbeeld's KVarN precision-tail benchmarks](https://anbeeld.de) — the
  evidence base for *which* pair to pick on quality grounds.

The contradiction is generational: those measurements describe what a
**default build** does (fallback), while this project describes what
**your build** does (native). Both can be true simultaneously. The full
record, with verified quotes, is in [`notes/`](notes/).

## Package layout

| Path | What |
|---|---|
| [README.md](README.md) | This file — the story and the TL;DR. |
| [AGENTS.md](AGENTS.md) | The operational guide: build, run, validate, gotchas. Verbatim commands. |
| [`notes/`](notes/) | The evidence trail: `0001` (proxy stub for the private fork-vs-vanilla record), `0002` fallback mechanics + allowlist, `0003` the asymmetric-KV discussion, `0004` precision-tail benchmarks, `0005` the HF model-card q8_0 prior. |
| [`scripts/`](scripts/) | The instruments: fixed PP/TG probe, quality battery, n-replica run, safe stopper, server launchers. |
