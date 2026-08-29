"""Command-line entry point for evaluating the 200 GUI-CC online tasks."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from utils.config import ONLINE_CONFIG, env, load_project_json

from online.scoring import (
    JudgeCache,
    aggregate_results,
    evaluate_tasks,
    prepare_tasks,
)
from online.trajectory import ONLINE_SAMPLES_FILE, load_task_definitions, select_task_ids
from online.judges import REQUEST_TIMEOUT
from utils.io import atomic_write_json
from utils.subset import subset_ids

# 默认为 None，直接建 OpenAI 客户端。API 与 GPU 分处两台机器时，外部驱动可以
# 在此换成自己的客户端；本仓库不依赖任何外部模块。
API_CLIENT = None

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_project_json(ONLINE_CONFIG)
JUDGE_MODEL = CONFIG["judge_model"]


def _requested_tasks(value: Optional[str]) -> Optional[list[str]]:
    return [item.strip() for item in value.split(",") if item.strip()] if value else None


def _output_path(rollout_dir: Path, partial: bool) -> Path:
    return rollout_dir / ("evaluation_partial.json" if partial else "evaluation_results.json")


def _load_previous(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    tasks = data.get("tasks") if isinstance(data, dict) else None
    return tasks if isinstance(tasks, dict) else {}


def _failed_rollout_result(task_id: str, rollout: dict) -> dict:
    """固定分母计零所需的失败记录。"""
    return {
        "task_id": task_id,
        "status": "error",
        "failure_class": rollout.get("failure_class", "infrastructure"),
        "error": str(rollout.get("error") or "rollout failed"),
        "metrics": {},
        "errors": [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 GPT-5.5 评测 GUI-CC Online rollout。")
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--task-ids", help="只评测这些 task（逗号分隔）；缺省评测全部 200 个")
    parser.add_argument("--subset", type=int, metavar="N",
                        help="只评测固定的 N 条小样本（等间距取样，所有模型相同；见 utils/subset.py）")
    parser.add_argument("--tasks-json", help="任务定义（默认：data/online_samples.jsonl）")
    parser.add_argument("--api-key", default=env("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--parallel", type=int, default=8,
                        help="并发评测的 task 数，即打向 judge API 的并发请求数")
    parser.add_argument("--judge-model", default=JUDGE_MODEL,
                        help="打分用的 VLM；默认是论文使用的模型")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if API_CLIENT is None and not args.api_key:
        raise SystemExit("未设置 OPENAI_API_KEY")

    rollout_dir = Path(args.rollout_dir).expanduser().resolve()
    rollout_json = rollout_dir / "rollout_results.json"
    if not rollout_json.is_file():
        raise SystemExit(f"找不到 {rollout_json}")
    try:
        rollouts = json.loads(rollout_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"rollout JSON 无效：{error}") from error
    if not isinstance(rollouts, dict):
        raise SystemExit("rollout_results.json 必须包含一个 JSON 对象")

    tasks_path = Path(args.tasks_json).expanduser() if args.tasks_json else ONLINE_SAMPLES_FILE
    task_defs = load_task_definitions(tasks_path)
    if args.subset:
        task_ids = subset_ids(list(task_defs), args.subset)
    else:
        task_ids = select_task_ids(task_defs, _requested_tasks(args.task_ids))
    partial = set(task_ids) != set(task_defs)

    # 分离可评测任务和失败 rollout 任务（失败任务固定分母计零）
    preparable_ids: list[str] = []
    failed_rollout_results: dict[str, dict] = {}
    for task_id in task_ids:
        rollout = rollouts.get(task_id, {})
        if rollout.get("complete") is True and not rollout.get("error"):
            preparable_ids.append(task_id)
        else:
            failed_rollout_results[task_id] = _failed_rollout_result(task_id, rollout)

    prepared = prepare_tasks(
        task_defs,
        rollouts,
        preparable_ids,
        rollout_dir,
        judge_model=args.judge_model,
        base_url=args.base_url,
    )

    output_path = _output_path(rollout_dir, partial)
    cache = JudgeCache(rollout_dir / "judge_cache")
    previous = _load_previous(output_path)
    run = {
        "schema": "gui_cc_online_evaluation",
        "status": "running",
        "mode": "partial" if partial else "full",
        "is_partial": partial,
        "n_requested": len(task_ids),
        "task_ids": task_ids,
        "judge_model": args.judge_model,
        "base_url": args.base_url,
        "rollout_dir": str(rollout_dir),
        "rollout": rollouts["_RUN"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    client = API_CLIENT or OpenAI(
        api_key=args.api_key, base_url=args.base_url, timeout=REQUEST_TIMEOUT)

    def save_progress(task_id: str, record: dict, current: dict[str, dict]) -> None:
        score = record.get("paper_scores", {}).get("Overall")
        print(f"[{task_id}] status={record.get('status')} Overall={score}")
        merged = {**failed_rollout_results, **current}
        atomic_write_json(output_path, {
            "run": run,
            "tasks": {selected: merged[selected] for selected in task_ids if selected in merged},
            "aggregate": aggregate_results(task_ids, merged, full=not partial),
        })

    evaluated = evaluate_tasks(
        prepared,
        preparable_ids,
        client,
        args.judge_model,
        cache,
        previous=previous,
        force=args.force,
        task_parallelism=args.parallel,
        on_result=save_progress,
    )
    results = {**failed_rollout_results, **evaluated}
    aggregate = aggregate_results(task_ids, results, full=not partial)
    run["status"] = aggregate["status"]
    run["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(output_path, {"run": run, "tasks": results, "aggregate": aggregate})
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print(f"输出：{output_path}")
    return 0 if aggregate["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
