#!/usr/bin/env bash
# vLLM/ZeroMQ 的 Unix socket 路径有长度上限，长 TMPDIR 会让服务静默失败；
# 因此在 /dev/shm 下建一个短的私有目录当 TMPDIR。

GUI_CC_VLLM_IPC_TMPDIR=""

gui_cc_prepare_vllm_ipc_tmpdir() {
  local base="${GUI_CC_VLLM_IPC_BASE:-/dev/shm}"
  GUI_CC_VLLM_IPC_TMPDIR="$(umask 077; mktemp -d "$base/gc.XXXXXXXX")"
  export GUI_CC_VLLM_IPC_TMPDIR
}

gui_cc_cleanup_vllm_ipc_tmpdir() {
  if [[ -n "${GUI_CC_VLLM_IPC_TMPDIR:-}" ]]; then
    rm -rf "$GUI_CC_VLLM_IPC_TMPDIR"
    GUI_CC_VLLM_IPC_TMPDIR=""
  fi
}
