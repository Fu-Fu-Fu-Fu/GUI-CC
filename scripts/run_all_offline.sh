#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
cd "$ROOT"

mapfile -t MATRIX < <("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("utils/configs/offline.json").read_text())
for model in config["models"]:
    for setting in model["settings"]:
        print(f'{model["id"]}\t{setting}')
PY
)

for row in "${MATRIX[@]}"; do
  IFS=$'\t' read -r model setting <<<"$row"
  if [[ -n "${RUN_MODELS:-}" && ",${RUN_MODELS}," != *",${model},"* ]]; then
    continue
  fi
  echo "==> Offline rollout: $model / $setting"
  "$PYTHON_BIN" -m offline.rollout --model "$model" --setting "$setting" \
    ${SUBSET:+--subset "$SUBSET"} "$@"
  if [[ "${EVALUATE_AFTER_RUN:-0}" == "1" ]]; then
    "$PYTHON_BIN" -m offline.cli --model "$model" --setting "$setting" \
      ${SUBSET:+--subset "$SUBSET"}
  fi
done
