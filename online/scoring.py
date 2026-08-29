"""Input validation, caching, scoring, and aggregation for online evaluation."""
from __future__ import annotations

import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from online import judges
from online.actions import is_terminal_action
from online.trajectory import build_trajectory, validate_rollouts
from utils.io import atomic_write_json


@dataclass(frozen=True)
class PreparedTask:
    task_id: str
    task: dict
    rollout: dict
    frames: tuple[Path, ...]
    actions: tuple[Optional[dict], ...]
    signature: str


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_frame(path: Path) -> None:
    """在发起任何付费 judge 请求前确认帧可解码。"""
    try:
        payload = path.read_bytes()
        with Image.open(BytesIO(payload)) as image:
            image.verify()
        with Image.open(BytesIO(payload)) as image:
            size = (int(image.width), int(image.height))
    except Exception as error:  # Pillow 会按图像格式抛出不同类型的异常。
        raise ValueError(f"轨迹帧 {path} 无效：{error}") from error
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"轨迹帧 {path} 的尺寸无效：{size}")


def _task_signature(
    task: dict,
    rollout: dict,
    *,
    rollout_run_sha256: Any,
    judge_model: str,
    base_url: str,
) -> str:
    """评测配置的稳定标识，用作 per-task judge 缓存的 key。"""
    payload = {
        "task": task,
        "rollout_record": rollout,
        "rollout_run_sha256": rollout_run_sha256,
        "prompts": judges.prompt_payload(),
        "configuration": {
            "judge_model": judge_model,
            "base_url": base_url,
            "image_encoding": judges.image_encoding_payload(),
        },
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def prepare_tasks(
    task_defs: dict[str, dict],
    rollouts: dict[str, dict],
    task_ids: list[str],
    rollout_dir: Path,
    *,
    judge_model: str,
    base_url: str,
) -> dict[str, PreparedTask]:
    """在调用 judge 前校验所有请求的 rollout 和帧。"""
    rollout_run = rollouts.get("_RUN")
    if not isinstance(rollout_run, dict):
        raise ValueError("rollout_results.json 缺少必需的 _RUN 元数据")
    validate_rollouts(
        rollouts,
        task_defs,
        task_ids,
        rollout_dir,
        run_sha256=rollout_run.get("run_sha256"),
    )
    prepared: dict[str, PreparedTask] = {}
    for task_id in task_ids:
        trajectory = build_trajectory(rollout_dir, task_id, rollouts[task_id])
        frames = tuple(trajectory["frames"])
        actions = tuple(trajectory["actions"])
        if len(actions) > task_defs[task_id]["step_budget"]:
            raise ValueError(
                f"任务 {task_id}：rollout 含 {len(actions)} 个动作，"
                f"但预算为 {task_defs[task_id]['step_budget']}"
            )
        if len(frames) > judges.MAX_TRAJECTORY_FRAMES:
            raise ValueError(
                f"任务 {task_id}：{len(frames)} 帧超过正式评测上限 "
                f"{judges.MAX_TRAJECTORY_FRAMES}"
            )
        if any(action is None for action in actions):
            raise ValueError(f"任务 {task_id}：rollout 包含不可用的智能体动作")
        for frame in frames:
            _validate_frame(frame)
        prepared[task_id] = PreparedTask(
            task_id=task_id,
            task=task_defs[task_id],
            rollout=rollouts[task_id],
            frames=frames,
            actions=actions,
            signature=_task_signature(
                task_defs[task_id],
                rollouts[task_id],
                rollout_run_sha256=rollout_run.get("run_sha256"),
                judge_model=judge_model,
                base_url=base_url,
            ),
        )
    return prepared


class JudgeCache:
    """按签名匹配复用的 per-metric judge 结果缓存。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, task_id: str, key: str) -> Path:
        safe_key = re.sub(r"[^a-zA-Z0-9_.-]", "_", key)
        return self.root / task_id / f"{safe_key}.json"

    def get(self, task_id: str, key: str, signature: str) -> Optional[dict]:
        path = self._path(task_id, key)
        if not path.is_file():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(envelope, dict):
            return None
        if envelope.get("signature") != signature or envelope.get("metric_key") != key:
            return None
        result = envelope.get("result")
        if not _valid_result(result):
            return None
        return result

    def put(self, task_id: str, key: str, signature: str, result: dict) -> None:
        if not _valid_result(result):
            return
        atomic_write_json(self._path(task_id, key), {
            "signature": signature,
            "metric_key": key,
            "result": result,
        })


def _is_score(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _valid_result(result: Any) -> bool:
    return (
        isinstance(result, dict)
        and not result.get("error")
        and _is_score(result.get("score"))
    )


def _cached_judge(
    cache: JudgeCache,
    prepared: PreparedTask,
    key: str,
    force: bool,
    call: Callable[[], dict],
) -> dict:
    cached = None if force else cache.get(prepared.task_id, key, prepared.signature)
    if cached is not None:
        return cached
    result = call()
    cache.put(prepared.task_id, key, prepared.signature, result)
    return result


def _step_metric(
    cache: JudgeCache,
    prepared: PreparedTask,
    client,
    model: str,
    step: int,
    force: bool,
) -> dict:
    action = prepared.actions[step]
    if action is None:
        return {"step": step, "skipped": True, "error": {"kind": "input_error", "message": "缺少动作"}}
    record: dict[str, Any] = {
        "step": step,
        "action": action,
    }
    if is_terminal_action(action):
        record["skipped"] = True
        return record
    before, after = prepared.frames[step], prepared.frames[step + 1]
    calls = {
        "S_ad": lambda: judges.judge_s_ad(client, model, before, after, prepared.task["instruction"], action),
        "S_id": lambda: judges.judge_s_id(client, model, before, after, action),
        "S_use": lambda: judges.judge_s_use(client, model, after),
    }
    for metric, call in calls.items():
        record[metric] = _cached_judge(
            cache, prepared, f"{metric}_step_{step:03d}", force, call
        )
    return record


def _mean_step_metric(steps: list[dict], metric: str) -> tuple[Optional[float], list[dict]]:
    considered = [step for step in steps if not step.get("skipped")]
    if not considered:
        return 0.0, []
    errors = []
    for step in considered:
        result = step.get(metric)
        if not isinstance(result, dict):
            errors.append({
                "path": f"steps[{step['step']}].{metric}",
                "kind": "missing_result",
                "message": "缺少 judge 结果",
            })
        elif result.get("error"):
            errors.append({"path": f"steps[{step['step']}].{metric}", **result["error"]})
    if errors:
        return None, errors
    scores = [step[metric].get("score") for step in considered]
    if not scores or not all(_is_score(score) for score in scores):
        return None, [{"path": metric, "kind": "missing_score", "message": "没有完整的步骤分数"}]
    return sum(float(score) for score in scores) / len(scores), []


def _paper_scores(metrics: dict[str, Optional[float]]) -> dict[str, Optional[float]]:
    return {
        metric: round(float(value) * 100.0, 1) if _is_score(value) else None
        for metric, value in metrics.items()
    }


def overall(metrics: dict[str, Any], errors: list[dict]) -> Optional[float]:
    values = [metrics.get(metric) for metric in judges.PAPER_METRICS]
    if errors or not all(_is_score(value) for value in values):
        return None
    return sum(float(value) for value in values) / len(values)


def evaluate_task(
    prepared: PreparedTask,
    client,
    model: str,
    cache: JudgeCache,
    *,
    force: bool = False,
) -> dict:
    steps = []
    for index in range(len(prepared.actions)):
        try:
            steps.append(_step_metric(cache, prepared, client, model, index, force))
        except Exception as error:  # 保留可审计记录，并将本次运行标记为失败。
            steps.append({"step": index,
                          "error": {"kind": "judge_exception", "message": str(error)[:500]}})
    errors = [
        {"path": f"steps[{step['step']}]", **step["error"]}
        for step in steps if step.get("error")
    ]
    metrics: dict[str, Optional[float]] = {}
    for metric in ("S_ad", "S_id", "S_use"):
        metrics[metric], metric_errors = _mean_step_metric(steps, metric)
        errors.extend(metric_errors)

    trajectory_calls = {
        "S_cp": lambda: judges.judge_s_cp(
            client, model, prepared.task["instruction"], list(prepared.actions), list(prepared.frames)
        ),
        "S_rd": lambda: judges.judge_s_rd(
            client, model, prepared.task["instruction"], list(prepared.actions), list(prepared.frames)
        ),
        "S_mp": lambda: judges.judge_s_mp(client, model, prepared.task, list(prepared.frames)),
    }
    trajectory_results: dict[str, dict] = {}
    for metric, call in trajectory_calls.items():
        try:
            result = _cached_judge(cache, prepared, metric, force, call)
        except Exception as error:
            result = {"score": None, "error": {"kind": "judge_exception", "message": str(error)[:500]}}
        trajectory_results[metric] = result
        if result.get("error"):
            errors.append({"path": metric, **result["error"]})
            metrics[metric] = None
        else:
            metrics[metric] = float(result["score"]) if _is_score(result.get("score")) else None
            if metrics[metric] is None:
                errors.append({"path": metric, "kind": "missing_score", "message": "judge 未返回有效分数"})

    metrics["Overall"] = overall(metrics, errors)
    return {
        "task_id": prepared.task_id,
        "signature": {"sha256": prepared.signature},
        "status": "complete" if _is_score(metrics["Overall"]) else "error",
        "metrics": metrics,
        "paper_scores": _paper_scores(metrics),
        "errors": errors,
        "steps": steps,
        "trajectory": trajectory_results,
        "input": {
            "frame_count": len(prepared.frames),
            "action_count": len(prepared.actions),
        },
    }


def reusable_result(record: Any, signature: str) -> bool:
    if not isinstance(record, dict):
        return False
    recorded_signature = record.get("signature")
    if not isinstance(recorded_signature, dict) or recorded_signature.get("sha256") != signature:
        return False
    metrics = record.get("metrics")
    return (
        record.get("status") == "complete"
        and not record.get("errors")
        and isinstance(metrics, dict)
        and all(_is_score(metrics.get(metric)) for metric in (*judges.PAPER_METRICS, "Overall"))
    )


def failed_task(prepared: PreparedTask, error: Exception, *, failure_class: str = "infrastructure") -> dict:
    metrics = {metric: None for metric in (*judges.PAPER_METRICS, "Overall")}
    return {
        "task_id": prepared.task_id,
        "signature": {"sha256": prepared.signature},
        "status": "error",
        "failure_class": failure_class,
        "metrics": metrics,
        "paper_scores": _paper_scores(metrics),
        "errors": [{"path": "task", "kind": "evaluation_exception", "message": str(error)[:500]}],
    }


def evaluate_tasks(
    prepared: dict[str, PreparedTask],
    task_ids: list[str],
    client,
    model: str,
    cache: JudgeCache,
    *,
    previous: Optional[dict[str, dict]] = None,
    force: bool = False,
    task_parallelism: int = 8,
    on_result: Optional[Callable[[str, dict, dict[str, dict]], None]] = None,
) -> dict[str, dict]:
    previous = previous or {}
    results = {
        task_id: previous[task_id]
        for task_id in task_ids
        if not force and reusable_result(previous.get(task_id), prepared[task_id].signature)
    }
    pending = [task_id for task_id in task_ids if task_id not in results]
    with ThreadPoolExecutor(max_workers=max(1, task_parallelism)) as pool:
        futures = {
            pool.submit(
                evaluate_task,
                prepared[task_id],
                client,
                model,
                cache,
                force=force,
            ): task_id
            for task_id in pending
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                results[task_id] = future.result()
            except Exception as error:
                results[task_id] = failed_task(prepared[task_id], error)
            if on_result:
                on_result(task_id, results[task_id], results)
    return {task_id: results[task_id] for task_id in task_ids}


def aggregate_results(
    task_ids: list[str], results: dict[str, dict], *, full: bool = True
) -> dict:
    """仅聚合 ``task_ids``，忽略恢复运行时出现的无关任务。"""
    requested = len(task_ids)
    metric_names = (*judges.PAPER_METRICS, "Overall")
    counts: dict[str, int] = {}
    metrics: dict[str, Optional[float]] = {}
    task_errors: dict[str, list[dict]] = {}
    model_failed_count = 0
    infra_blocked_count = 0
    n_complete = 0
    for task_id in task_ids:
        record = results.get(task_id)
        if not isinstance(record, dict):
            task_errors[task_id] = [{"kind": "missing_task", "message": "缺少评测结果"}]
            continue
        failure_class = record.get("failure_class")
        if failure_class == "model":
            model_failed_count += 1
            continue
        if failure_class == "infrastructure":
            task_errors[task_id] = [{"kind": "infra_blocked", "message": "基础设施失败（INFRA_BLOCKED）"}]
            infra_blocked_count += 1
            continue
        errors = list(record.get("errors") or [])
        record_metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        missing = [metric for metric in metric_names if not _is_score(record_metrics.get(metric))]
        if missing:
            errors.append({
                "kind": "missing_metrics",
                "message": f"缺少有效指标：{', '.join(missing)}",
            })
        if errors:
            task_errors[task_id] = errors
        else:
            n_complete += 1
    for metric in metric_names:
        values: list[float] = []
        for task_id in task_ids:
            record = results.get(task_id, {})
            fc = record.get("failure_class")
            if fc == "model":
                values.append(0.0)
            elif fc == "infrastructure":
                continue
            else:
                score = record.get("metrics", {}).get(metric)
                if _is_score(score):
                    values.append(float(score))
        counts[metric] = len(values)
        metrics[metric] = (
            sum(values) / requested
            if requested and len(values) == requested
            else None
        )
    complete = (
        requested > 0
        and not task_errors
        and all(counts[metric] == requested for metric in metric_names)
    )
    if not complete or not full:
        metrics["Overall"] = None
    return {
        "status": "complete" if complete else "error",
        "scope": "full" if full else "partial",
        "n_requested": requested,
        "n_complete_tasks": n_complete,
        "failure_counts": {"model": model_failed_count, "infra_blocked": infra_blocked_count},
        "n": counts,
        "metrics": metrics,
        "paper_scores": _paper_scores(metrics),
        "errors": {"count": len(task_errors), "tasks": task_errors},
    }
