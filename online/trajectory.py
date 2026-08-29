"""Online task loading and rollout reconstruction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from online.actions import is_terminal_action

REPO_ROOT = Path(__file__).resolve().parents[1]
ONLINE_SAMPLES_FILE = REPO_ROOT / "data" / "online_samples.jsonl"
ONLINE_DATA_ROOT = REPO_ROOT / "data" / "online_data"
ONLINE_OUTPUT_ROOT = REPO_ROOT / "outputs" / "online"

# Online judge 一次最多接收的帧数；budget+1（含 seed 帧）不得超过该值。
MAX_TRAJECTORY_FRAMES = 25


def load_task_definitions(path: Path = ONLINE_SAMPLES_FILE) -> dict[str, dict]:
    """按行序加载 online_samples.jsonl，返回 task_id -> 任务定义。"""
    tasks: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        tasks[item["task_id"]] = item
    for task_id, task in tasks.items():
        if task["step_budget"] + 1 > MAX_TRAJECTORY_FRAMES:
            raise ValueError(
                f"任务 {task_id!r} 的 step_budget={task['step_budget']} 超出 judge 帧上限 "
                f"{MAX_TRAJECTORY_FRAMES - 1}（budget+1 帧必须 ≤ {MAX_TRAJECTORY_FRAMES}）"
            )
    return tasks


def select_task_ids(task_defs: dict[str, dict], requested: Optional[list[str]]) -> list[str]:
    return list(requested) if requested else list(task_defs)


def resolve_recorded_path(value: Any, rollout_dir: Path, task_id: str) -> Path:
    """解析 rollout 记录里的相对产物路径（保证产物可随目录整体迁移）。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"任务 {task_id}：记录的产物路径为空")
    recorded = Path(value)
    if recorded.is_absolute():
        raise ValueError(f"任务 {task_id}：产物路径必须相对于 rollout 目录：{value}")
    candidate = rollout_dir / recorded
    if not candidate.is_file():
        raise FileNotFoundError(f"任务 {task_id}：找不到记录的产物：{candidate}")
    return candidate


def build_trajectory(rollout_dir: Path, task_id: str, rollout: dict) -> dict:
    """把 rollout 记录重建为对齐的 (frames, actions) 序列。"""
    if rollout.get("complete") is not True:
        raise ValueError(f"任务 {task_id}：rollout 未标记为完成")
    if rollout.get("error"):
        raise ValueError(f"任务 {task_id}：rollout 包含错误：{rollout['error']}")
    if rollout.get("task_id") != task_id:
        raise ValueError(f"任务 {task_id}：rollout 中的 task_id 为 {rollout.get('task_id')!r}")
    entries = rollout.get("trajectory")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"任务 {task_id}：rollout 轨迹缺失或为空")

    frames: list[Path] = []
    actions: list[Optional[dict]] = []
    terminated = False
    for expected_step, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"任务 {task_id}：轨迹条目 {expected_step} 无效")
        if entry.get("step") != expected_step:
            raise ValueError(
                f"任务 {task_id}：预期轨迹步骤 {expected_step}，实际为 {entry.get('step')!r}"
            )
        frames.append(resolve_recorded_path(entry.get("frame"), rollout_dir, task_id))
        action = entry.get("semantic_action")
        if action is not None and not isinstance(action, dict):
            raise ValueError(f"任务 {task_id}：步骤 {expected_step} 的 semantic_action 无效")
        if action is None and not (entry.get("error") or entry.get("wm_error")):
            raise ValueError(f"任务 {task_id}：步骤 {expected_step} 既无语义动作，也无错误记录")
        if entry.get("terminated"):
            if expected_step != len(entries) - 1:
                raise ValueError(f"任务 {task_id}：终止动作不是最后一个轨迹条目")
            if not isinstance(action, dict) or not is_terminal_action(action):
                raise ValueError(f"任务 {task_id}：终止轨迹条目缺少终止动作")
            terminated = True
            continue
        actions.append(action)

    if not terminated:
        final_frame = rollout_dir / task_id / "final_frame.png"
        if not final_frame.is_file():
            raise FileNotFoundError(f"任务 {task_id}：找不到 rollout 最终帧：{final_frame}")
        frames.append(final_frame)
    if len(frames) != len(actions) + 1:
        raise ValueError(
            f"任务 {task_id}：轨迹未对齐，{len(actions)} 个动作对应 {len(frames)} 帧"
        )
    return {
        "frames": frames,
        "actions": actions,
        "terminated": terminated,
    }


def validate_rollouts(
    rollouts: Any,
    task_defs: dict[str, dict],
    task_ids: list[str],
    rollout_dir: Path,
    run_sha256: str | None = None,
) -> None:
    """校验待评测的 rollout 集合齐全、与任务对应且可重建。"""
    if not isinstance(rollouts, dict):
        raise ValueError("rollout_results.json 必须包含一个 JSON 对象")
    missing = [task_id for task_id in task_ids if task_id not in rollouts]
    if missing:
        raise ValueError(f"缺少 {len(missing)} 个必需的 rollout：{', '.join(missing[:20])}")
    for task_id in task_ids:
        rollout = rollouts[task_id]
        if not isinstance(rollout, dict):
            raise ValueError(f"任务 {task_id}：rollout 记录不是 JSON 对象")
        if rollout.get("step_budget") != task_defs[task_id]["step_budget"]:
            raise ValueError(
                f"任务 {task_id}：rollout 的 step_budget={rollout.get('step_budget')!r} 与"
                f"任务定义（{task_defs[task_id]['step_budget']}）不一致"
            )
        if run_sha256 and rollout.get("run_sha256") != run_sha256:
            raise ValueError(f"任务 {task_id}：rollout 运行标识与 _RUN 不一致")
        build_trajectory(rollout_dir, task_id, rollout)
