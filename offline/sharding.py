"""Sharding and merge utilities for offline rollout."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

from utils.config import OFFLINE_CONFIG, load_project_json

from offline.data import (
    OFFLINE_PREDICTIONS_ROOT,
    OFFLINE_SAMPLES_FILE,
    load_sample_ids,
)
from utils.io import atomic_write_json, load_json

ROOT = Path(__file__).resolve().parents[1]
_CONFIG = load_project_json(OFFLINE_CONFIG)


def history_dir(setting: str) -> str:
    if setting == "WM-Markov":
        return "markov"
    if setting == "WM-FullHist":
        return "fullhist"
    raise ValueError(f"unsupported history setting: {setting}")


def partition_sample_ids(
    all_ids: list[str], shard_count: int, shard_index: int
) -> list[str]:
    """按样本序号做稳定轮询分片。"""
    if shard_count < 1:
        raise ValueError("shard_count must be a positive integer")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must be in [0, {shard_count}), got {shard_index}"
        )
    return [
        sample_id for position, sample_id in enumerate(all_ids)
        if position % shard_count == shard_index
    ]


def shard_worker_root(
    output_root: Path, model: str, setting: str, shard_count: int, shard_index: int
) -> Path:
    """返回 shard 独占的 adapter output root。"""
    partition_sample_ids([], shard_count, shard_index)
    return (
        output_root / ".shards" / f"{model}-{history_dir(setting)}"
        / f"shard-{shard_index:05d}-of-{shard_count:05d}"
    )


def sample_output_present(config_dir: Path, record: dict[str, Any]) -> bool:
    """判定一个 summary 记录是否已终结且盘上产物齐全（不校验哈希)。

    完成样本要求每个已执行步骤存在 ``pred.png``；模型失败样本要求
    失败步骤之前的 ``pred.png`` 存在。
    """
    steps = record.get("steps")
    sample_id = record.get("sample_id")
    if not isinstance(steps, list) or not isinstance(sample_id, str) or not sample_id:
        return False
    if record.get("complete") is True:
        needs_png = steps
    elif record.get("failure_class") == "model" and record.get("error"):
        needs_png = steps[:-1]
    else:
        return False
    for step in needs_png:
        if not isinstance(step, dict) or not isinstance(step.get("step"), int):
            return False
        if not (config_dir / sample_id / f"step_{step['step']:03d}" / "pred.png").is_file():
            return False
    return True


def merge_shards(
    *, model: str, setting: str,
    output_root: Path, shard_count: int,
    shard_output_roots: Sequence[Path] | None = None,
    samples_file: str | Path = OFFLINE_SAMPLES_FILE,
) -> Path:
    """把全部 shard 的产物按样本顺序合并到单一输出目录。"""
    if shard_count < 1:
        raise ValueError("shard_count must be a positive integer")
    all_ids = load_sample_ids(samples_file=samples_file)
    history = history_dir(setting)
    final_dir = output_root / model / history
    if final_dir.exists():
        raise ValueError(f"refusing to overwrite an existing output directory: {final_dir}; use a new --output-root")
    source_roots = (
        [Path(path).expanduser().resolve() for path in shard_output_roots]
        if shard_output_roots is not None
        else [output_root] * shard_count
    )
    if len(source_roots) != shard_count:
        raise ValueError(f"shard_output_roots must provide exactly {shard_count} paths")

    common_run: dict[str, Any] | None = None
    records: dict[str, dict[str, Any]] = {}
    shard_dirs: dict[str, Path] = {}
    for index, source_root in enumerate(source_roots):
        config_dir = (
            shard_worker_root(source_root, model, setting, shard_count, index)
            / model / history
        )
        run = load_json(config_dir / "run.json")
        summary = load_json(config_dir / "summary.json")
        if run.get("model", {}).get("model_id") != model:
            raise ValueError(f"shard {index} has a mismatched model")
        if run.get("model", {}).get("history_setting") != setting:
            raise ValueError(f"shard {index} has a mismatched setting")
        if common_run is None:
            common_run = run
        elif run.get("run_sha256") != common_run.get("run_sha256"):
            raise ValueError(f"shard {index} has a run configuration that differs from the other shards")
        expected_ids = partition_sample_ids(all_ids, shard_count, index)
        samples = summary.get("samples", {})
        for sample_id in expected_ids:
            record = samples.get(sample_id)
            if not isinstance(record, dict) or not sample_output_present(config_dir, record):
                raise ValueError(f"shard {index} sample {sample_id} is missing or unfinished")
            records[sample_id] = record
            shard_dirs[sample_id] = config_dir

    assert common_run is not None
    missing = [sample_id for sample_id in all_ids if sample_id not in records]
    if missing:
        raise ValueError(f"samples missing after merge: {missing[:3]}")

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{history}.merge.", dir=final_dir.parent))
    try:
        atomic_write_json(staging / "run.json", common_run)
        ordered = {sample_id: records[sample_id] for sample_id in all_ids}
        atomic_write_json(staging / "summary.json", {"run": common_run, "samples": ordered})
        for sample_id in all_ids:
            source = shard_dirs[sample_id] / sample_id
            if not source.is_dir():
                raise ValueError(f"missing sample artifact directory: {source}")
            shutil.copytree(source, staging / sample_id, copy_function=shutil.copy2)
        os.replace(staging, final_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return final_dir


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge independent GUI-CC offline rollout shards")
    sub = parser.add_subparsers(dest="command", required=True)
    merge = sub.add_parser("merge", help="merge every shard")
    merge.add_argument(
        "--model",
        choices=[str(spec["id"]) for spec in _CONFIG.get("models", [])],
        required=True,
    )
    merge.add_argument("--setting", choices=["WM-Markov", "WM-FullHist"], required=True)
    merge.add_argument("--samples", default=str(OFFLINE_SAMPLES_FILE))
    merge.add_argument("--output-root", default=str(OFFLINE_PREDICTIONS_ROOT))
    merge.add_argument("--shard-count", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    output = merge_shards(
        model=args.model,
        setting=args.setting,
        output_root=Path(args.output_root).expanduser().resolve(),
        shard_count=args.shard_count,
        samples_file=args.samples,
    )
    print(f"Offline shard merge complete: {output}", flush=True)


if __name__ == "__main__":
    main()
