#!/usr/bin/env bash
# 在一个独立 GPU 任务中启动指定 vLLM 服务并运行一个 GUI-CC model/setting。
# 适用于单机（单卡或多卡）GPU 环境。不在日志中打印 key。
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$ROOT/scripts/env.sh"
source "$ROOT/scripts/vllm_ipc_tmpdir.sh"
cd "$ROOT"

MODEL="${1:-}"
PHASE="${2:-offline}"
SETTING="${3:-WM-Markov}"
if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 <code2world> <offline|online> <WM-Markov|WM-FullHist> [runner args...]" >&2
  exit 2
fi
shift $(( $# >= 3 ? 3 : $# ))

case "$MODEL" in
  code2world) PORT="${PORT:-4244}"; EXPECTED=code2world ;;
  *) echo "ERROR: unsupported vLLM model: $MODEL" >&2; exit 2 ;;
esac
case "$PHASE" in offline|online) ;; *) echo "ERROR: phase must be offline or online" >&2; exit 2 ;; esac
case "$SETTING" in WM-Markov|WM-FullHist) ;; *) echo "ERROR: invalid setting: $SETTING" >&2; exit 2 ;; esac

HISTORY_DIR=markov
[[ "$SETTING" == WM-FullHist ]] && HISTORY_DIR=fullhist
LOG_DIR="$ROOT/outputs/logs/vllm_jobs/${MODEL}/${HISTORY_DIR}"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server.log"
RUN_LOG="$LOG_DIR/${PHASE}.log"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
if [[ -z "${WORKERS:-}" ]]; then
  if [[ "$SETTING" == WM-FullHist ]]; then
    WORKERS=2
  else
    WORKERS=8
  fi
fi

server_pid=""
descendants_of() {
  local parent="$1" child
  while IFS= read -r child; do
    [[ -n "$child" ]] || continue
    descendants_of "$child"
    printf '%s\n' "$child"
  done < <(pgrep -P "$parent" 2>/dev/null || true)
}
stop_service() {
  local pid="$1" label="$2"
  [[ -n "$pid" ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  local -a children=()
  mapfile -t children < <(descendants_of "$pid")
  ((${#children[@]})) && kill -TERM "${children[@]}" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
  for _ in {1..30}; do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  echo "WARN: forcing shutdown of $label pid=$pid" >&2
  ((${#children[@]})) && kill -KILL "${children[@]}" 2>/dev/null || true
  kill -KILL "$pid" 2>/dev/null || true
}
cleanup() {
  stop_service "$server_pid" world_model
  gui_cc_cleanup_vllm_ipc_tmpdir
}
trap cleanup EXIT INT TERM

ensure_port_free() {
  local port="$1" label="$2"
  PORT_TO_CHECK="$port" PORT_LABEL="$label" "$PYTHON_BIN" - <<'PY'
import os
import socket

port = int(os.environ["PORT_TO_CHECK"])
label = os.environ["PORT_LABEL"]
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as error:
        raise SystemExit(f"{label} port {port} is already in use: {error}")
PY
}

if [[ "$PHASE" == online ]]; then
  planner_api_key="${OPENAI_API_KEY:-}"
  planner_base_url="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
  [[ -n "$planner_api_key" ]] || {
    echo "ERROR: Online planner credential is missing; set OPENAI_API_KEY" >&2
    exit 2
  }
  [[ -n "$planner_base_url" ]] || {
    echo "ERROR: Online planner endpoint is missing; set OPENAI_BASE_URL" >&2
    exit 2
  }
  export OPENAI_API_KEY="$planner_api_key"
  export OPENAI_BASE_URL="$planner_base_url"
fi
ensure_port_free "$PORT" world_model

gui_cc_prepare_vllm_ipc_tmpdir
printf '%s\n' "$GUI_CC_VLLM_IPC_TMPDIR" > "$LOG_DIR/vllm_ipc_tmpdir.txt"

printf '%s\n' "$launch_output"
server_pid="$(sed -nE 's/.*pid=([0-9]+).*/\1/p' <<<"$launch_output" | tail -1)"
[[ "$server_pid" =~ ^[0-9]+$ ]] || { echo "ERROR: unable to parse vLLM PID" >&2; exit 1; }

case "$MODEL" in
  code2world) export CODE2WORLD_URL="http://127.0.0.1:$PORT/v1" ;;
esac

deadline=$((SECONDS + STARTUP_TIMEOUT))
until payload="$(curl -fsS --max-time 10 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)"; do
  if ((SECONDS >= deadline)); then
    echo "ERROR: vLLM readiness timeout; see $SERVER_LOG" >&2
    exit 1
  fi
  kill -0 "$server_pid" 2>/dev/null || { echo "ERROR: vLLM exited; see $SERVER_LOG" >&2; exit 1; }
  sleep 5
done
MODELS_PAYLOAD="$payload" EXPECTED_MODEL="$EXPECTED" "$PYTHON_BIN" - <<'PY'
import json, os
payload = json.loads(os.environ["MODELS_PAYLOAD"])
ids = {row.get("id") for row in payload.get("data", []) if isinstance(row, dict)}
expected = os.environ["EXPECTED_MODEL"]
if expected not in ids:
    raise SystemExit(f"endpoint model mismatch: expected={expected!r}, actual={sorted(ids)}")
PY

echo "GUI-CC job: model=$MODEL phase=$PHASE setting=$SETTING" | tee -a "$RUN_LOG"
if [[ "$PHASE" == offline ]]; then
  "$PYTHON_BIN" -m offline.rollout --model "$MODEL" --setting "$SETTING" --workers "$WORKERS" "$@" 2>&1 | tee -a "$RUN_LOG"
else
  "$PYTHON_BIN" -m online.rollout --model "$MODEL" --setting "$SETTING" "$@" 2>&1 | tee -a "$RUN_LOG"
fi
echo "GUI-CC job complete: model=$MODEL phase=$PHASE setting=$SETTING" | tee -a "$RUN_LOG"
