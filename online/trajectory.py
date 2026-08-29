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

# Online judge 一次最多接收的帧数；budget+1（contains seed frames)不得超过该值。
MAX_TRAJECTORY_FRAMES = 25


def load_task_definitions(path: Path = ONLINE_SAMPLES_FILE) -> dict[str, dict]:
    """Load online_samples.jsonl in line order and return task_id -> task definition."""
    tasks: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        tasks[item["task_id"]] = item
    for task_id, task in tasks.items():
        if task["step_budget"] + 1 > MAX_TRAJECTORY_FRAMES:
            raise ValueError(
                f"task {task_id!r} step_budget={task['step_budget']}  exceeds the judge frame limit of "
                f"{MAX_TRAJECTORY_FRAMES - 1} (budget+1 frames must be <= {MAX_TRAJECTORY_FRAMES})"
            )
    return tasks


def select_task_ids(task_defs: dict[str, dict], requested: Optional[list[str]]) -> list[str]:
    return list(requested) if requested else list(task_defs)


def resolve_recorded_path(value: Any, rollout_dir: Path, task_id: str) -> Path:
    """Resolve a recorded relative artifact path so the output tree stays movable."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"task {task_id}: the recorded artifact path is empty")
    recorded = Path(value)
    if recorded.is_absolute():
        raise ValueError(f"task {task_id}: the artifact path must be relative to the rollout directory: {value}")
    candidate = rollout_dir / recorded
    if not candidate.is_file():
        raise FileNotFoundError(f"task {task_id}: recorded artifact not found: {candidate}")
    return candidate


def build_trajectory(rollout_dir: Path, task_id: str, rollout: dict) -> dict:
    """Rebuild a rollout record into aligned (frames, actions) sequences."""
    if rollout.get("complete") is not True:
        raise ValueError(f"task {task_id}: the rollout is not marked complete")
    if rollout.get("error"):
        raise ValueError(f"task {task_id}: the rollout contains an error: {rollout['error']}")
    if rollout.get("task_id") != task_id:
        raise ValueError(f"task {task_id}: the rollout carries task_id {rollout.get('task_id')!r}")
    entries = rollout.get("trajectory")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"task {task_id}: the rollout trajectory is missing or empty")

    frames: list[Path] = []
    actions: list[Optional[dict]] = []
    terminated = False
    for expected_step, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"task {task_id}: trajectory entry {expected_step} is invalid")
        if entry.get("step") != expected_step:
            raise ValueError(
                f"task {task_id}: expected trajectory step {expected_step}, got {entry.get('step')!r}"
            )
        frames.append(resolve_recorded_path(entry.get("frame"), rollout_dir, task_id))
        action = entry.get("semantic_action")
        if action is not None and not isinstance(action, dict):
            raise ValueError(f"task {task_id}: step {expected_step} has an invalid semantic_action")
        if action is None and not (entry.get("error") or entry.get("wm_error")):
            raise ValueError(f"task {task_id}: step {expected_step} has neither a semantic action nor an error record")
        if entry.get("terminated"):
            if expected_step != len(entries) - 1:
                raise ValueError(f"task {task_id}: the terminal action is not the last trajectory entry")
            if not isinstance(action, dict) or not is_terminal_action(action):
                raise ValueError(f"task {task_id}: the terminal trajectory entry has no terminal action")
            terminated = True
            continue
        actions.append(action)

    if not terminated:
        final_frame = rollout_dir / task_id / "final_frame.png"
        if not final_frame.is_file():
            raise FileNotFoundError(f"task {task_id}: final rollout frame not found: {final_frame}")
        frames.append(final_frame)
    if len(frames) != len(actions) + 1:
        raise ValueError(
            f"task {task_id}: trajectory is misaligned, {len(actions)} actions for {len(frames)} frames"
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
    """Validate that the rollouts to evaluate are complete, matched to tasks, and reconstructible."""
    if not isinstance(rollouts, dict):
        raise ValueError("rollout_results.json must contain a JSON object")
    missing = [task_id for task_id in task_ids if task_id not in rollouts]
    if missing:
        raise ValueError(f"missing {len(missing)} required rollout(s): {', '.join(missing[:20])}")
    for task_id in task_ids:
        rollout = rollouts[task_id]
        if not isinstance(rollout, dict):
            raise ValueError(f"task {task_id}: the rollout record is not a JSON object")
        if rollout.get("step_budget") != task_defs[task_id]["step_budget"]:
            raise ValueError(
                f"task {task_id}: the rollout step_budget={rollout.get('step_budget')!r} 与"
                f"任务定义（{task_defs[task_id]['step_budget']})"
            )
        if run_sha256 and rollout.get("run_sha256") != run_sha256:
            raise ValueError(f"task {task_id}: the rollout run identity does not match _RUN")
        build_trajectory(rollout_dir, task_id, rollout)
