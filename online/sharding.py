"""Sharding and merge utilities for online rollout."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from online.trajectory import ONLINE_SAMPLES_FILE, load_task_definitions
from utils.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"missing or corrupt JSON: {path}: {error}") from error


def history_dir(setting: str) -> str:
    if setting == "WM-NoHist":
        return "nohist"
    if setting == "WM-FullHist":
        return "fullhist"
    raise ValueError(f"unsupported history setting: {setting}")


def partition_task_ids(
    all_ids: list[str], shard_count: int, shard_index: int
) -> list[str]:
    """按任务序号做稳定轮询分片。"""
    if shard_count < 1:
        raise ValueError("shard_count must be a positive integer")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must be in [0, {shard_count}), got {shard_index}"
        )
    return [
        task_id for position, task_id in enumerate(all_ids)
        if position % shard_count == shard_index
    ]


def shard_worker_root(
    output_root: Path, model: str, setting: str, shard_count: int, shard_index: int
) -> Path:
    """返回 shard 独占的输出根目录。"""
    partition_task_ids([], shard_count, shard_index)
    return (
        output_root / ".shards" / f"{model}-{history_dir(setting)}"
        / f"shard-{shard_index:05d}-of-{shard_count:05d}"
    )


def task_output_present(output_dir: Path, record: dict[str, Any]) -> bool:
    """判定一个 rollout 记录是否已终结且盘上产物齐全（不校验哈希)。"""
    task_id = record.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        return False
    task_dir = output_dir / task_id
    if record.get("complete") is True:
        return (task_dir / "trajectory.json").is_file() and (task_dir / "final_frame.png").is_file()
    if record.get("failure_class") == "model" and record.get("error"):
        return (task_dir / "trajectory.json").is_file()
    return False


def merge_shards(
    *,
    model: str,
    setting: str,
    tasks_file: Path,
    output_root: Path,
    shard_count: int,
) -> Path:
    """把全部 shard 的产物按任务顺序合并到单一输出目录。"""
    if shard_count < 1:
        raise ValueError("shard_count must be a positive integer")
    task_defs = load_task_definitions(tasks_file)
    all_ids = list(task_defs)
    history = history_dir(setting)
    final_dir = output_root / model / history
    if final_dir.exists():
        raise ValueError(f"refusing to overwrite an existing output directory: {final_dir}; use a new --output-root")

    common_run: dict[str, Any] | None = None
    records: dict[str, dict[str, Any]] = {}
    shard_dirs: dict[str, Path] = {}
    for index in range(shard_count):
        output_dir = (
            shard_worker_root(output_root, model, setting, shard_count, index)
            / model / history
        )
        results = _load_json(output_dir / "rollout_results.json")
        run = results.get("_RUN", {})
        if run.get("model", {}).get("model_id") != model:
            raise ValueError(f"shard {index} has a mismatched model")
        if run.get("model", {}).get("history_setting") != setting:
            raise ValueError(f"shard {index} has a mismatched setting")
        if common_run is None:
            common_run = run
        elif run.get("run_sha256") != common_run.get("run_sha256"):
            raise ValueError(f"shard {index} has a run configuration that differs from the other shards")
        for task_id in partition_task_ids(all_ids, shard_count, index):
            record = results.get(task_id)
            if not isinstance(record, dict) or not task_output_present(output_dir, record):
                raise ValueError(f"shard {index} task {task_id} is missing or unfinished")
            records[task_id] = record
            shard_dirs[task_id] = output_dir

    assert common_run is not None
    missing = [task_id for task_id in all_ids if task_id not in records]
    if missing:
        raise ValueError(f"tasks missing after merge: {missing[:3]}")

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{history}.merge.", dir=final_dir.parent))
    try:
        merged = {"_RUN": common_run}
        merged.update({task_id: records[task_id] for task_id in all_ids})
        atomic_write_json(staging / "rollout_results.json", merged)
        for task_id in all_ids:
            source = shard_dirs[task_id] / task_id
            if not source.is_dir():
                raise ValueError(f"missing task artifact directory: {source}")
            shutil.copytree(source, staging / task_id, copy_function=shutil.copy2)
        os.replace(staging, final_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return final_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independent GUI-CC online rollout shards")
    sub = parser.add_subparsers(dest="command", required=True)
    merge = sub.add_parser("merge", help="merge every shard")
    merge.add_argument("--model", required=True)
    merge.add_argument("--setting", choices=["WM-NoHist", "WM-FullHist"], required=True)
    merge.add_argument("--tasks-file", default=str(ONLINE_SAMPLES_FILE))
    merge.add_argument("--output-root", required=True)
    merge.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    output = merge_shards(
        model=args.model,
        setting=args.setting,
        tasks_file=Path(args.tasks_file).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        shard_count=args.shard_count,
    )
    print(f"Online shard merge complete: {output}", flush=True)


if __name__ == "__main__":
    main()
