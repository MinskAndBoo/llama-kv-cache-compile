# 0001 — ik-llama-cpp vs vanilla, for GSQ-RCO (proxy note)

*These are not the droids you are looking for.*

The original note with this slug — the full record of the
**fork-vs-vanilla** decision — is a private project document and is not
part of this package. It is deeply unsatisfying for a curious human to be
told that, so here is the residue that matters for this package:

**What the original question was.** The machine in question runs
Qwen3.8-27B-GSQ-RCO-IQ3_S (a GGUF quant) + BF16 mmproj on a single 24 GB
RTX 4090, and was originally served by `ik_llama.cpp` — a llama.cpp fork.
The question was whether that fork is required over vanilla upstream.

**The answer (measured A/B, identical fixed prompts, 172,032 ctx, MTP on):**

| config | Prefill tok/s | Decode tok/s | VRAM (MiB) |
|---|---|---|---|
| fork, q6_0 KV | 2078.7 | 65.32 | 22,660 |
| vanilla, q4_0 KV | **2421.2** | **91.69** | **21,331** |

Vanilla won on every axis — +16% prefill, +40% decode, ~1.3 GiB less VRAM.
The daily runner was switched to vanilla; the fork remains fully
restorable. Everything the fork had that was special (the `q6_0` KV type,
`--spec-ckpt-mode`, `--no-mmap`) has a vanilla equivalent or was not
load-bearing.

**Why this note is out of scope here.** This package is about a different
question: **KV-cache compilation and selection** — which quantized K/V
pairs to compile into the flash-attention kernels (`GGML_CUDA_FA_QUANTS`),
and which one to run (see [../README.md](../README.md),
[../AGENTS.md](../AGENTS.md), and notes 0002–0005). The fork-vs-vanilla
decision is provenance for *which* server the KV work ran on, not part of
the KV evidence chain.
