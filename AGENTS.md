# AGENTS.md — operational guide

Goal: build llama.cpp (CUDA) with a chosen quantized K/V pair compiled into
the flash-attention kernels, serve it, and validate — using only this folder.
Written for an agent to run verbatim; every command is literal.

**Pinned reference build:** llama.cpp @ commit `81bc6b8` (Sep 2026). FA
fallback behavior is build-specific — measurements here do not transfer to
other commits. Pin, build, then probe.

## 0. Prerequisites

- Single NVIDIA GPU with the CUDA toolkit, `cmake`, `ninja` (or `make`),
  `python3`, `curl`, `ss`, `nvidia-smi`.
- A GGUF model (+ optional mmproj GGUF) that you want to serve.
- The scripts in `scripts/` use `bash`, no other dependencies.

Clone once, outside this folder:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$HOME/llama.cpp"
cd "$HOME/llama.cpp" && git fetch origin 81bc6b8 && git checkout 81bc6b8
```

## 1. Build

Two build trees, both from the same source:

- `build/` — **vanilla**: default FA pairs only (`q4_0-q4_0;q8_0-q8_0;
  f16-f16;bf16-bf16`). Useful as the baseline.
- `build-candidates/` — **candidates**: your extra pairs compiled in. This is
  the one you serve.

```bash
cd "$HOME/llama.cpp"
cmake -S . -B build-candidates \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_COMPRESSION_MODE=size \
  -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;q5_0-q4_1;q5_0-q4_0;q8_0-q5_1"
cmake --build build-candidates -j"$(nproc)"
```

**To use a different pair:** append `K-V` (semicolon-separated; K and V each
from `q4_0 q4_1 q5_0 q5_1 q8_0 bf16 f16`) to `GGML_CUDA_FA_QUANTS` and
rebuild. That is the entire cost. `f16-f16` is always compiled whether you
list it or not. A type *outside* the 7 is not a flag — it is a source change
(ggml type registration → KV paths → FA integration; see the reference in
`notes/0002`).

### Verify the build

1. `grep GGML_CUDA_FA_QUANTS build-candidates/CMakeCache.txt` — your list is there.
2. Build exits 0.
3. **Runtime (the real check):** after startup, the server log's startup
   section (grep `n_ctx_slot`, scan nearby `W` lines) shows **no**
   cache-type/FA fallback warning. A warning line means you are on the
   f16-conversion fallback — quality-fine, speed-penalized.

## 2. Run the server

Launchers in `scripts/` take everything by environment; set at minimum
`MODEL`, and `BIN` if your build tree isn't the default location:

```bash
export MODEL=/path/to/model.gguf
export MMPROJ=/path/to/mmproj.gguf          # optional
export BIN="$HOME/llama.cpp/build-candidates/bin/llama-server"
export K=q8_0 V=q5_1 CTX=143360 PORT=8080
bash scripts/start-llama-candidates.sh
```

- `start-llama.sh` = the vanilla build (default pairs only — anything else
  falls back there). `start-llama-candidates.sh` = the build above.
- Both safely take down whatever already listens on `$PORT` (via
  `stop-server.sh`, which refuses to kill a non-llama-server process or a
  llama-server running a model you haven't whitelisted in its `MODEL_MATCH`
  env var).
- Logs append to `logs/llama-server-<flavor>.log`; the newest `n_ctx_slot`
  line marks each instance start.
- `stop-server.sh` alone: graceful stop, SIGKILL only after 20 s.

## 3. Validate, in order

| # | Check | Command | Pass |
|---|---|---|---|
| 1 | No FA fallback | `grep -nE 'n_ctx_slot' logs/llama-server-candidates.log` then scan the following ~20 lines for `W` | No cache-type/FA warning in the newest startup section |
| 2 | VRAM + GTT | `free -h` **before** the swap, `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` after | VRAM ≈ expected footprint **and** no matching system-RAM rise (GTT spill is *silent* — see gotcha 2) |
| 3 | Speed | `bash scripts/ab-probe.sh <tag> logs/llama-server-candidates.log $PORT` | Fixed 10,944-token fresh-slot prefill + 256-token decode + MTP acceptance, printed from the server log |
| 4 | Quality | `bash scripts/quality-battery.sh <tag> $PORT` | N1/N2 needle-in-haystack (25% / 90% of ~99K tokens), T1 tool call, V1 4×4 checkerboard — all byte-identical requests |
| 5 | Decision-grade (optional) | `bash scripts/ab-replicate.sh <N=11> <K> <V>` | N replicas of probe + battery + 800-token V1 re-ask (~2.2 min each); aborts first if the live server is not the claimed K/V pair |

Outputs land in `artifacts/` (git-ignored when this folder is its own repo):
`ab-results-<tag>.txt`, `quality-<tag>.txt`, `replicate-summary.txt`, raw
responses.

### How to read the numbers

- **Prefill (PP) is the clean speed axis.** Flat within ~5% across all four
  compiled-in pairs in our run; ±1.6% around the median within one config.
- **Decode (TG) is not a KV-speed signal** on stacks with speculative
  decoding: corr(TG, MTP acceptance) = 0.996 across 11 replicas. The whole
  TG spread was MTP-acceptance variance at temp 1.0.
- **The quality battery detects catastrophic failures; it does not rank KV
  types.** Needle-style probes score 100% for every quantized KV type, so a
  pass tie proves nothing about quality ordering. Local quality ordering is
  prior-only (`notes/0003`–`0005`).

## 4. Gotchas (each cost a real incident)

1. **Never `pkill -f` with the model path inline in a shell command** — the
   pattern matches the caller's own cmdline and kills your own shell. Use
   `stop-server.sh`.
2. **GTT spill is silent.** The KV cache that doesn't fit in VRAM lands in
   system RAM (GPU texture memory) with no warning: clean startup, then a
   ~1 GiB spill ≈ −22% generation speed. The `free -h` before/after check is
   the only detector.
3. **Chat `max_tokens` counts thinking tokens.** A small budget returns empty
   `content` (truncated reasoning), not a model failure. Allow headroom or
   check `reasoning_content`.
4. **Swapping the server swaps the agent's own runtime if the agent is
   served by it.** A dead server = a dead session. Swaps must be atomic
   (stop → wait for VRAM release → start → poll health) and the session
   should not be mid-critical-work during one.
5. **Version pinning matters.** A real example: an Ollama release shipped a
   Qwen3.8 MTP regression (~193 → ~17 tok/s) six days before this work.
   Rebuild deliberately at a pinned commit and re-run `ab-probe.sh` after
   every rebuild — do not trust that "the build is the same".
6. **`q4_1`, asymmetric pairs, and the rest of the allowlist do not blow
   up.** They either run native (compiled in) or run the exact-dequant f16
   fallback with one warning. The scary stories in the issue tracker are
   pre-fix builds or the uncompiled fallback path — see `notes/0002`.

## 5. Evidence

- `notes/0001-ik-llama-cpp-vs-vanilla-for-gsq-rco.md` — **proxy stub only**:
  the full fork-vs-vanilla record is private; the stub states the outcome
  (vanilla won every axis) and why it is out of scope for this package.
- `notes/0002-kv-cache-quant-type-support-in-vanilla-llama-cpp.md` — the
  three runtime states, the 7-type allowlist, verified upstream doc text.
- `notes/0003-llama-cpp-discussion-23470-asymmetric-kv-evidence.md` — the
  500×ARC asymmetric-KV discussion, KLD/KVarN evidence, VRAM arithmetic.
- `notes/0004-anbeeld-kvarn-precision-tail-benchmarks.md` — precision-tail
  numbers per pair (the quality prior).
- `notes/0005-hf-thinkingcap-qwen38-card-q8-kv-claim.md` — the HF model-card
  q8_0-KV claim, the GTT-silent-spill measurement, MTP note.
