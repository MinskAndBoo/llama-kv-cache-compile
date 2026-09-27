#!/usr/bin/env bash
# start-llama-candidates.sh — start the CANDIDATES build (FA_QUANTS extended,
# recipe in AGENTS.md §1) serving the model in $MODEL.
#
# Env:
#   MODEL        path to the model GGUF            (required)
#   MMPROJ       path to the mmproj GGUF           (optional)
#   LLAMACPP_DIR source/build tree                 (default: $HOME/llama.cpp)
#   BIN          llama-server binary               (default: $LLAMACPP_DIR/build-candidates/bin/llama-server)
#   K, V         --cache-type-k / --cache-type-v   (defaults q8_0 / q5_1)
#   CTX          --ctx-size                        (default 143360)
#   PORT         (default 8080)  NMAX (default 3)  CKPT (default 4)
#
# SAFELY TAKES DOWN ANY EXISTING INSTANCE ON $PORT FIRST (via stop-server.sh,
# which refuses to kill a non-llama-server or a non-whitelisted model).
# Logs append to logs/llama-server-candidates.log (package root); the newest
# n_ctx_slot line marks each instance start.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:?set MODEL to the path of the model GGUF}"
MMPROJ="${MMPROJ:-}"
LLAMACPP_DIR="${LLAMACPP_DIR:-$HOME/llama.cpp}"
BIN="${BIN:-$LLAMACPP_DIR/build-candidates/bin/llama-server}"
LOG="$ROOT/logs/llama-server-candidates.log"

K="${K:-q8_0}"
V="${V:-q5_1}"
CTX="${CTX:-143360}"
PORT="${PORT:-8080}"
NMAX="${NMAX:-3}"
CKPT="${CKPT:-4}"

[ -x "$BIN" ] || { echo "candidates binary missing: $BIN (build per AGENTS.md §1)" >&2; exit 3; }
[ -f "$MODEL" ] || { echo "model file missing: $MODEL" >&2; exit 3; }
mkdir -p "$ROOT/logs"

echo "==> taking down any existing instance on port $PORT (safely)"
bash "$ROOT/scripts/stop-server.sh" "$PORT"

echo "==> starting candidates llama-server (K=$K V=$V CTX=$CTX port=$PORT) -> $LOG"
nohup "$BIN" \
  --model "$MODEL" \
  ${MMPROJ:+--mmproj "$MMPROJ"} \
  --jinja \
  --gpu-layers 99 \
  --ctx-size "$CTX" \
  --flash-attn on \
  --cache-type-k "$K" \
  --cache-type-v "$V" \
  --load-mode none \
  --spec-type draft-mtp \
  --spec-draft-n-max "$NMAX" \
  --ctx-checkpoints "$CKPT" \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --host 127.0.0.1 --port "$PORT" \
  >> "$LOG" 2>&1 &
echo "    pid $! (health check: curl -s localhost:$PORT/v1/models)"
