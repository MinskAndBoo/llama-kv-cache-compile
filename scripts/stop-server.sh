#!/usr/bin/env bash
# stop-server.sh [PORT] — stop the llama-server listening on PORT (default 8080).
#
# Safety rules (refuses instead of killing when uncertain):
#   1. The port holder must be a process whose executable name is
#      exactly "llama-server" — anything else (nginx, ollama, ...) is left alone.
#   2. Its command line must match MODEL_MATCH (default: the Qwen3.8-27B
#      GSQ-RCO model of this package's measurements) — guards against
#      killing an unrelated llama-server instance. Override:
#      MODEL_MATCH='*path/to/your-model.gguf*' stop-server.sh
#   3. Graceful SIGTERM first, SIGKILL only after ~20 s if it will not die.
#
# Exit codes: 0 = stopped or nothing to stop; 3 = refused (port held by a
# non-llama-server process or a non-matching model); 4 = could not determine
# port holder.
set -euo pipefail

PORT="${1:-8080}"
MODEL_MATCH="${MODEL_MATCH:-*Qwen3.8-27B-GSQ-RCO*}"

pid=$(ss -tlnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p {print $NF}' \
      | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)
if [ -z "$pid" ]; then
  echo "==> nothing listening on port $PORT"
  exit 0
fi

comm_name=$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ')
if [ "$comm_name" != "llama-server" ]; then
  echo "REFUSED: port $PORT is held by pid $pid ($comm_name), not llama-server" >&2
  exit 3
fi

args=$(ps -o args= -p "$pid" 2>/dev/null || true)
case "$args" in
  $MODEL_MATCH)
    : ;;  # one of ours — proceed
  *)
    echo "REFUSED: pid $pid on port $PORT is a llama-server but not one of our model instances" >&2
    exit 3
    ;;
esac

echo "==> stopping llama-server (pid $pid, port $PORT)"
kill "$pid" 2>/dev/null || true
for i in $(seq 1 10); do
  kill -0 "$pid" 2>/dev/null || { echo "    stopped."; exit 0; }
  sleep 2
done
echo "    still alive after 20s, sending SIGKILL"
kill -9 "$pid" 2>/dev/null || true
for i in $(seq 1 15); do
  kill -0 "$pid" 2>/dev/null || { echo "    stopped (SIGKILL)."; exit 0; }
  sleep 2
done
echo "ERROR: pid $pid did not die" >&2
exit 5
