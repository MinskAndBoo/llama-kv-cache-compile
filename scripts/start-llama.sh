#!/usr/bin/env bash
# start-llama.sh — start the VANILLA build (default FA pairs only:
# q4_0-q4_0; q8_0-q8_0; f16-f16; bf16-bf16) serving the model in $MODEL.
# Anything else falls back to the warned f16-conversion path (AGENTS.md §1).
# Useful as the speed baseline next to the candidates build.
#
# Env:
#   MODEL        path to the model GGUF            (required)
#   MMPROJ       path to the mmproj GGUF           (optional)
#   LLAMACPP_DIR source/build tree                 (default: $HOME/llama.cpp)
#   BIN          llama-server binary               (default: $LLAMACPP_DIR/build/bin/llama-server)
#   KV           --cache-type-k/v (symmetric)      (default q4_0)
#   CTX          --ctx-size                        (default 143360)
#   PORT         (default 8080)  NMAX (default 3)  CKPT (default 4)
#
# SAFELY TAKES DOWN ANY EXISTING INSTANCE ON $PORT FIRST (via stop-server.sh).
# Logs append to logs/llama-server-vanilla.log (package root).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:?set MODEL to the path of the model GGUF}"
MMPROJ="${MMPROJ:-}"
LLAMACPP_DIR="${LLAMACPP_DIR:-$HOME/llama.cpp}"
BIN="${BIN:-$LLAMACPP_DIR/build/bin/llama-server}"
LOG="$ROOT/logs/llama-server-vanilla.log"

KV="${KV:-q4_0}"
CTX="${CTX:-143360}"
PORT="${PORT:-8080}"
NMAX="${NMAX:-3}"
CKPT="${CKPT:-4}"

[ -x "$BIN" ] || { echo "vanilla binary missing: $BIN (build per AGENTS.md §1)" >&2; exit 3; }
[ -f "$MODEL" ] || { echo "model file missing: $MODEL" >&2; exit 3; }
mkdir -p "$ROOT/logs"

echo "==> taking down any existing instance on port $PORT (safely)"
bash "$ROOT/scripts/stop-server.sh" "$PORT"

echo "==> starting vanilla llama-server (KV=$KV CTX=$CTX port=$PORT) -> $LOG"
nohup "$BIN" \
  --model "$MODEL" \
  ${MMPROJ:+--mmproj "$MMPROJ"} \
  --jinja \
  --gpu-layers 99 \
  --ctx-size "$CTX" \
  --flash-attn on \
  --cache-type-k "$KV" \
  --cache-type-v "$KV" \
  --load-mode none \
  --spec-type draft-mtp \
  --spec-draft-n-max "$NMAX" \
  --ctx-checkpoints "$CKPT" \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --host 127.0.0.1 --port "$PORT" \
  >> "$LOG" 2>&1 &
echo "    pid $! (health check: curl -s localhost:$PORT/v1/models)"
