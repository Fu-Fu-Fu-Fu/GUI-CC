#!/usr/bin/env bash
# Start the vLLM service for one world model: serve.sh <code2world>
set -euo pipefail

MODEL="${1:?Usage: serve.sh <code2world>}"
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$ROOT/scripts/env.sh"

EXTRA_ARGS=()
case "$MODEL" in
  code2world)
    GPU="${GPU:-2}"; PORT="${PORT:-4244}"; TP="${TP:-1}"
    CKPT="${CKPT:-${CODE2WORLD_CKPT:?Set CODE2WORLD_CKPT in utils/configs/paths.env}}"
    MAX_LEN=32768
    MM_KWARGS='{"size": {"longest_edge": 3072000, "shortest_edge": 65536}}'
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
    ;;
  *) echo "ERROR: unsupported vLLM model: $MODEL" >&2; exit 2 ;;
esac

LOG="${LOG:-$ROOT/outputs/logs/${MODEL}_vllm.log}"
mkdir -p "$(dirname "$LOG")"
echo "$MODEL vllm serve: gpu=$GPU port=$PORT" > "$LOG"

CUDA_VISIBLE_DEVICES="$GPU" \
VLLM_USE_DEEP_GEMM=0 \
nohup "${VLLM:-$VLLM_BIN}" serve "$CKPT" \
    --port "$PORT" \
    --served-model-name "$MODEL" \
    --max-model-len "$MAX_LEN" \
    --mm-processor-kwargs "$MM_KWARGS" \
    --limit-mm-per-prompt '{"image": 5}' \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "${GPU_MEM:-0.9}" \
    --allowed-local-media-path '/' \
    --trust-remote-code \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    >> "$LOG" 2>&1 &

PID=$!
echo "PID=$PID" >> "$LOG"
echo "$MODEL vllm started, pid=$PID. Tail $LOG to follow."
