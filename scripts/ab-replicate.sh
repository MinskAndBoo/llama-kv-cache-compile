#!/usr/bin/env bash
# ab-replicate.sh — N replicas of the A/B test set on the CURRENT live
# server, for percentile speed stats (n>1) + output-stability checks
# (does V1 flip-flop? do N1/N2/T1 answers change?).
#
# Usage: ab-replicate.sh [N=11] [K=q8_0] [V=q5_1]
#   N   number of replicas (~2.2 min each; 11 ≈ 25 min)
#   K V the --cache-type-k / --cache-type-v of the LIVE server. The script
#       aborts immediately if the live server is not this pair (safety
#       before burning the run).
#
# Per replica:
#   1. ab-probe.sh rep-r<i>        (fixed PP + TG prompts; timings from server log)
#   2. quality-battery.sh rep-b<i> (N1, N2, T1, V1 @ 200-token budget)
#   3. one 800-token V1 re-ask (the comparable V1 condition; prints
#      content + reasoning length + usage)
#
# Outputs (package-root artifacts/):
#   artifacts/replicate-summary.txt              (V1-800 + VRAM per replica)
#   artifacts/ab-results-rep-r<i>.txt            (probe timings per replica)
#   artifacts/quality-rep-b<i>.txt               (battery per replica)
#
# Aggregate afterward with a python percentile pass over the artifacts
# (method: linear interpolation, p-th percentile at rank p/100*(n-1)).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG=${LLAMA_LOG:-"$ROOT/logs/llama-server-candidates.log"}
PORT=8080
N=${1:-11}
K=${2:-q8_0}
V=${3:-q5_1}
PROJ="$ROOT/scripts"
SCRATCH="$ROOT/artifacts"
SUM="$SCRATCH/replicate-summary.txt"

# Safety: the live server must be the expected K/V pair before we burn ~25 min.
ps -eo args | grep "[l]lama-server" | grep -q -- "--cache-type-k $K --cache-type-v $V" \
  || { echo "ABORT: live server is not $K/$V (check: ps -eo args | grep llama-server)" >&2; exit 1; }

: > "$SUM"
echo "=== replicate start $(date -u +%FT%TZ) pair=$K/$V n=$N VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) ===" >> "$SUM"

for i in $(seq 1 "$N"); do
  echo "--- replica $i/$N $(date -u +%FT%TZ) ---" >> "$SUM"
  bash "$PROJ/ab-probe.sh" "rep-r$i" "$LOG" "$PORT" >/dev/null 2>&1 \
    || echo "PROBE FAIL $i" >> "$SUM"
  bash "$PROJ/quality-battery.sh" "rep-b$i" "$PORT" >/dev/null 2>&1 \
    || echo "BATTERY FAIL $i" >> "$SUM"
  python3 - "$SCRATCH/qb-req.json" "$PORT" >> "$SUM" 2>&1 <<'PYEOF'
import json, sys, urllib.request
path, port = sys.argv[1], sys.argv[2]
body = json.load(open(path))
body['max_tokens'] = 800
req = urllib.request.Request(
    f'http://127.0.0.1:{port}/v1/chat/completions',
    data=json.dumps(body).encode(),
    headers={'Content-Type': 'application/json'})
r = json.load(urllib.request.urlopen(req))
m = r['choices'][0]['message']
print('V1-800 content:', (m.get('content') or '(empty)')[:200])
print('V1-800 reasoning_chars:', len(m.get('reasoning_content') or ''))
print('V1-800 usage:', json.dumps(r.get('usage', {})))
PYEOF
  echo "VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> "$SUM"
done

echo "=== replicate done $(date -u +%FT%TZ) ===" >> "$SUM"
cat "$SUM"
