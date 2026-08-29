#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
source "$ROOT/scripts/env.sh"

"$PYTHON_BIN" scripts/validate_data.py
"$PYTHON_BIN" -m compileall -q offline online utils scripts tests
"$PYTHON_BIN" -m pytest -q
for script in scripts/*.sh; do
  bash -n "$script"
done

check_path_var() {
  local name="$1"
  require_var "$name"
  [[ -e "${!name}" ]] || {
    echo "ERROR: $name does not exist: ${!name}" >&2
    return 2
  }
}

check_model_endpoint() {
  local base_url="$1"
  local expected_model="$2"
  local payload
  payload="$(curl -fsS --max-time 10 "${base_url%/}/models")" || {
    echo "ERROR: 模型 endpoint 不可达：${base_url%/}/models" >&2
    return 2
  }
  MODELS_PAYLOAD="$payload" EXPECTED_MODEL="$expected_model" "$PYTHON_BIN" - <<'PY'
import json
import os

payload = json.loads(os.environ["MODELS_PAYLOAD"])
model_ids = {
    row.get("id") for row in payload.get("data", []) if isinstance(row, dict)
}
expected = os.environ["EXPECTED_MODEL"]
if expected not in model_ids:
    raise SystemExit(f"endpoint 未提供预期模型 {expected!r}；实际为 {sorted(model_ids)}")
PY
}

if [[ -n "${RUNTIME_MODEL:-}" ]]; then
  "$PYTHON_BIN" - <<'PY'
import importlib
for name in ("numpy", "openai", "PIL", "torch"):
    importlib.import_module(name)
PY
  case "$RUNTIME_MODEL" in
    code2world) check_path_var CODE2WORLD_CKPT; require_var CODE2WORLD_REVISION ;;
    qwen_image_edit)
      check_path_var DIFFSYNTH_DIR
      check_path_var QWEN_IMAGE_EDIT_2511_DIR
      require_var QWEN_IMAGE_EDIT_REVISION
      ;;
    gpt55)
      [[ -n "${CLOSED_MODEL_API_KEY:-${OPENAI_API_KEY:-}}" ]] || {
        echo "ERROR: set CLOSED_MODEL_API_KEY or OPENAI_API_KEY." >&2; exit 2;
      }
      ;;
    *) echo "ERROR: unknown RUNTIME_MODEL=$RUNTIME_MODEL" >&2; exit 2 ;;
  esac

  case "$RUNTIME_MODEL" in
    code2world|gpt55)
      "$PYTHON_BIN" - <<'PY'
from pathlib import Path
from playwright.sync_api import sync_playwright
with sync_playwright() as playwright:
    executable = Path(playwright.chromium.executable_path)
if not executable.is_file():
    raise SystemExit("Chromium is missing; run: python -m playwright install chromium")
PY
      ;;
  esac
  if [[ "${CHECK_MODEL_ENDPOINT:-0}" == "1" ]]; then
    case "$RUNTIME_MODEL" in
      code2world) check_model_endpoint "${CODE2WORLD_URL:-http://localhost:4244/v1}" code2world ;;
    esac
  fi
  echo "Static runtime dependency check passed for $RUNTIME_MODEL."
fi

if [[ "${ONLINE_RUNTIME:-0}" == "1" ]]; then
  [[ -n "${OPENAI_API_KEY:-}" ]] || {
    echo "ERROR: Online planner 缺少可用的 API key；请设置 OPENAI_API_KEY。" >&2
    exit 2
  }
  echo "Online planner 凭据检查通过。"
fi

echo "GUI-CC preflight passed."
