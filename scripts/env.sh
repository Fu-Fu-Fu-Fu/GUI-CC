#!/usr/bin/env bash
# GUI-CC 脚本共用的本地环境加载器。

if [[ -z "${GUI_CC_ROOT:-}" ]]; then
  GUI_CC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export GUI_CC_ROOT

GUI_CC_ENV_FILE="${GUI_CC_ENV_FILE:-$GUI_CC_ROOT/utils/configs/paths.env}"
if [[ -f "$GUI_CC_ENV_FILE" ]]; then
  # paths.env 只补充调用者未设置的变量；命令行前缀和已导出的环境变量优先。
  # macOS 自带 Bash 3.2：使用两个同下标的普通数组，避免 Bash 4 才支持的
  # `declare -A` 和 `[[ -v name ]]`。
  _GUI_CC_CALLER_ENV_NAMES=()
  _GUI_CC_CALLER_ENV_VALUES=()
  while IFS= read -r _GUI_CC_ENV_LINE || [[ -n "$_GUI_CC_ENV_LINE" ]]; do
    if [[ "$_GUI_CC_ENV_LINE" =~ ^[[:space:]]*(export[[:space:]]+)?([a-zA-Z_][a-zA-Z0-9_]*)= ]]; then
      _GUI_CC_ENV_NAME="${BASH_REMATCH[2]}"
      if [[ ${!_GUI_CC_ENV_NAME+x} ]]; then
        _GUI_CC_ENV_INDEX=${#_GUI_CC_CALLER_ENV_NAMES[@]}
        _GUI_CC_CALLER_ENV_NAMES[$_GUI_CC_ENV_INDEX]="$_GUI_CC_ENV_NAME"
        _GUI_CC_CALLER_ENV_VALUES[$_GUI_CC_ENV_INDEX]="${!_GUI_CC_ENV_NAME}"
      fi
    fi
  done < "$GUI_CC_ENV_FILE"

  case "$-" in
    *a*) _GUI_CC_HAD_ALLEXPORT=1 ;;
    *) _GUI_CC_HAD_ALLEXPORT=0 ;;
  esac
  set -a
  # shellcheck source=/dev/null
  source "$GUI_CC_ENV_FILE"
  if [[ "$_GUI_CC_HAD_ALLEXPORT" == "0" ]]; then
    set +a
  fi
  for ((_GUI_CC_ENV_INDEX=0; _GUI_CC_ENV_INDEX<${#_GUI_CC_CALLER_ENV_NAMES[@]}; _GUI_CC_ENV_INDEX++)); do
    _GUI_CC_ENV_NAME="${_GUI_CC_CALLER_ENV_NAMES[$_GUI_CC_ENV_INDEX]}"
    printf -v "$_GUI_CC_ENV_NAME" '%s' "${_GUI_CC_CALLER_ENV_VALUES[$_GUI_CC_ENV_INDEX]}"
    export "${_GUI_CC_ENV_NAME?}"
  done
  unset _GUI_CC_ENV_NAME _GUI_CC_ENV_LINE _GUI_CC_ENV_INDEX
  unset _GUI_CC_CALLER_ENV_NAMES _GUI_CC_CALLER_ENV_VALUES _GUI_CC_HAD_ALLEXPORT
fi

VLLM_BIN="${VLLM_BIN:-vllm}"

if [[ -z "${ENV_PREFIX:-}" && -n "${GUI_CC_CONDA_ENV:-}" ]]; then
  ENV_PREFIX="$GUI_CC_CONDA_ENV"
fi
if [[ -n "${ENV_PREFIX:-}" && -x "$ENV_PREFIX/bin/python" ]]; then
  export PATH="$ENV_PREFIX/bin:$PATH"
  export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
VIMO_TEXT_INFILL_FONT_PATH="${VIMO_TEXT_INFILL_FONT_PATH:-/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf}"
VIMO_TEXT_INFILL_FONT_FALLBACKS="${VIMO_TEXT_INFILL_FONT_FALLBACKS:-/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:/usr/share/fonts/truetype/freefont/FreeSans.ttf:/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf}"

export PYTHON_BIN VLLM_BIN ENV_PREFIX HF_HOME HF_HUB_CACHE
export VIMO_TEXT_INFILL_FONT_PATH VIMO_TEXT_INFILL_FONT_FALLBACKS

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: set $name in utils/configs/paths.env or the environment." >&2
    return 2
  fi
}
