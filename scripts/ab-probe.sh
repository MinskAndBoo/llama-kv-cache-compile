#!/usr/bin/env bash
# ab-probe.sh <tag> <server-log-file> [port]
#
# Reproducible PP/TG probe for a running llama-server.
#
# What it does:
#   1. Regenerates two FIXED, deterministic prompts (byte-identical across
#      runs — they are embedded in this file):
#        ab-pp-prompt-<tag>.txt  ~27k chars of repeated code (~11k tokens) -> long prefill
#        ab-tg-prompt-<tag>.txt  fixed 3-paragraph question                 -> 256-token decode
#   2. Sends them to the llama server (127.0.0.1:PORT) as /v1/completions
#      (non-streaming; temperature 1.0, top_p 0.95, top_k 20, min_p 0.0).
#   3. Extracts the server's own timing lines from <server-log-file>:
#        "prompt eval time = ..."  (PP tok/s)
#        "eval time = ..."         (TG tok/s)
#        "draft acceptance = ..."  (MTP acceptance)
#   4. Writes ab-results-<tag>.txt under artifacts/ and prints it.
#
# Usage:  scripts/ab-probe.sh <tag> logs/llama-server-candidates.log 8080
#
# Reproduction: same script + same prompts + same model file + same
# server flags = same measurement conditions.
set -uo pipefail
TAG="$1"; LOG="$2"; PORT="${3:-8080}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH="$ROOT/artifacts"
mkdir -p "$SCRATCH"
PP="$SCRATCH/ab-pp-prompt-$TAG.txt"
TG="$SCRATCH/ab-tg-prompt-$TAG.txt"
OUT="$SCRATCH/ab-results-$TAG.txt"
LAST_JSON="$SCRATCH/ab-last-response-$TAG.json"

[ -r "$LOG" ] || { echo "cannot read log: $LOG" >&2; exit 2; }

# --- fixed prompts (deterministic; regenerated every run) -----------------
python3 - "$PP" "$TG" <<'PYEOF'
import sys
block = (
    "def is_prime(n: int) -> bool:\n"
    "    if n < 2:\n"
    "        return False\n"
    "    for i in range(2, int(n ** 0.5) + 1):\n"
    "        if n % i == 0:\n"
    "            return False\n"
    "    return True\n"
    "\n"
    "def sieve(limit: int) -> list:\n"
    "    keep = [True] * (limit + 1)\n"
    "    keep[0] = keep[1] = False\n"
    "    for i in range(2, int(limit ** 0.5) + 1):\n"
    "        if keep[i]:\n"
    "            for j in range(i * i, limit + 1, i):\n"
    "                keep[j] = False\n"
    "    return [i for i, k in enumerate(keep) if k]\n"
    "\n"
    "assert is_prime(97) and not is_prime(100)\n"
    "assert sieve(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]\n"
)
open(sys.argv[1], "w").write(block * 48)  # ~27.3k chars, ~7k tokens
open(sys.argv[2], "w").write(
    "Explain, in exactly three paragraphs, how speculative decoding works in "
    "large language model inference. Cover the role of the draft model, the "
    "verification step, and the acceptance criterion. Plain prose, no headings.\n"
)
PYEOF

measure() { # $1=label $2=prompt_file $3=max_tokens
  local bytes0 body
  bytes0=$(wc -c < "$LOG")
  body=$(python3 - "$2" "$3" <<'PYEOF'
import json, sys
print(json.dumps({
    "model": "local",
    "prompt": open(sys.argv[1]).read(),
    "max_tokens": int(sys.argv[2]),
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "stream": False,
}))
PYEOF
)
  curl -sS "http://127.0.0.1:$PORT/v1/completions" \
    -H 'Content-Type: application/json' -d "$body" -o "$LAST_JSON" \
    || { echo "ERROR: curl failed for probe $1 (server may be down)" >&2; return 1; }
  sleep 1
  {
    echo "=== $TAG / $1 (prompt_bytes=$(wc -c < "$2"), max_tokens=$3) ==="
    tail -c +"$((bytes0 + 1))" "$LOG" \
      | grep -aE "prompt eval time|eval time|draft acceptance|statistics mtp"
  } >> "$OUT"
}

echo "=== A/B probe tag=$TAG $(date -u +%FT%TZ) ===" > "$OUT"
echo "GPU memory: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | head -1)" >> "$OUT"
measure PP "$PP" 1
measure TG "$TG" 256
cat "$OUT"
echo "probe done: $TAG (raw responses in $LAST_JSON)"
