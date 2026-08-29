#!/usr/bin/env python3
"""Compare results and API cost across models on the same small subset.

First run rollout and evaluation with `--subset N` (every model gets the same samples, see
utils/subset.py), then use this script to lay out each model/setting row side by side with its
scores, failure counts, and token usage, extrapolating token cost linearly from N to the full
run so you can estimate the API bill before committing to it.

    python scripts/compare_subset.py --split offline --subset 10
    python scripts/compare_subset.py --split online  --subset 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline.data import load_sample_ids  # noqa: E402
from online.trajectory import load_task_definitions  # noqa: E402
from utils.subset import subset_ids  # noqa: E402

OFFLINE_METRICS = ("S_ele", "S_lay", "S_sig", "S_dino", "S_ad",
                   "S_id", "S_use", "S_cp", "S_rd", "S_rap")
ONLINE_METRICS = ("S_ad", "S_id", "S_use", "S_cp", "S_rd", "S_mp")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _matrix(split: str) -> list[tuple[str, str]]:
    config = _load(ROOT / "utils" / "configs" / f"{split}.json")
    return [(model["id"], setting) for model in config["models"] for setting in model["settings"]]


def _hist(setting: str) -> str:
    return "fullhist" if setting == "WM-FullHist" else "markov"


def add_usage(value: Any, total: dict[str, int]) -> None:
    """递归累加结果文件里所有 usage 记录（chat: prompt/completion，images: input/output）。"""
    if isinstance(value, dict):
        if "prompt_tokens" in value or "input_tokens" in value:
            total["in"] += int(value.get("prompt_tokens") or value.get("input_tokens") or 0)
            total["out"] += int(value.get("completion_tokens") or value.get("output_tokens") or 0)
            total["calls"] += 1
            return
        for item in value.values():
            add_usage(item, total)
    elif isinstance(value, list):
        for item in value:
            add_usage(item, total)


def _usage_of_files(paths: list[Path]) -> dict[str, int]:
    total = {"in": 0, "out": 0, "calls": 0}
    for path in paths:
        add_usage(_load(path), total)
    return total


def _offline_row(output_root: Path, model: str, setting: str, ids: list[str]) -> dict | None:
    pred_dir = output_root / "offline" / "predictions" / model / _hist(setting)
    eval_dir = output_root / "offline" / "evaluation" / model / _hist(setting)
    summary = _load(pred_dir / "summary.json")
    if summary is None:
        return None
    samples = summary.get("samples", {})
    rollout = {"complete": 0, "model_failed": 0, "missing": 0}
    scored, zeroed, missing = [], 0, 0
    for sample_id in ids:
        record = samples.get(sample_id, {})
        if record.get("complete") is True:
            rollout["complete"] += 1
        elif record.get("failure_class") == "model":
            rollout["model_failed"] += 1
        else:
            rollout["missing"] += 1
        evaluation = _load(eval_dir / sample_id / "evaluation.json")
        if evaluation and evaluation.get("complete") is True:
            scored.append(evaluation["metrics"])
        elif record.get("failure_class") == "model":
            zeroed += 1
        else:
            missing += 1
    n_scored = len(scored) + zeroed
    metrics = {
        metric: (sum(float(item[metric]) for item in scored) / n_scored if n_scored else None)
        for metric in OFFLINE_METRICS
    }
    return {
        "rollout": rollout,
        "evaluation": {"scored": len(scored), "zeroed": zeroed, "missing": missing},
        "metrics": metrics,
        "wallclock_s": sum(float(samples.get(s, {}).get("wallclock_s") or 0) for s in ids),
        "usage_rollout": _usage_of_files(
            [p for s in ids for p in sorted((pred_dir / s).glob("step_*/meta.json"))]),
        "usage_judge": _usage_of_files([eval_dir / s / "evaluation.json" for s in ids]),
    }


def _online_row(output_root: Path, model: str, setting: str, ids: list[str]) -> dict | None:
    rollout_dir = output_root / "online" / model / _hist(setting)
    rollouts = _load(rollout_dir / "rollout_results.json")
    if rollouts is None:
        return None
    tasks: dict[str, dict] = {}
    for name in ("evaluation_partial.json", "evaluation_results.json"):
        payload = _load(rollout_dir / name) or {}
        tasks.update(payload.get("tasks") or {})
    rollout = {"complete": 0, "model_failed": 0, "missing": 0}
    scored, zeroed, missing = [], 0, 0
    for task_id in ids:
        record = rollouts.get(task_id, {})
        if record.get("complete") is True:
            rollout["complete"] += 1
        elif record.get("failure_class") == "model":
            rollout["model_failed"] += 1
        else:
            rollout["missing"] += 1
        result = tasks.get(task_id) or {}
        if result.get("status") == "complete":
            scored.append(result["metrics"])
        elif record.get("failure_class") == "model" or result.get("failure_class") == "model":
            zeroed += 1
        else:
            missing += 1
    n_scored = len(scored) + zeroed
    metrics = {
        metric: (sum(float(item[metric]) for item in scored) / n_scored if n_scored else None)
        for metric in ONLINE_METRICS
    }
    wm_cache = rollout_dir / "wm_cache" / model / _hist(setting)
    return {
        "rollout": rollout,
        "evaluation": {"scored": len(scored), "zeroed": zeroed, "missing": missing},
        "metrics": metrics,
        "wallclock_s": sum(float(rollouts.get(t, {}).get("total_time_s") or 0) for t in ids),
        "usage_planner": _usage_of_files(
            [p for t in ids for p in sorted((rollout_dir / t).glob("step_*/agent_evidence.json"))]),
        "usage_rollout": _usage_of_files(
            [p for t in ids for p in sorted((wm_cache / t).glob("step_*/meta.json"))]),
        "usage_judge": _usage_dict([tasks[t] for t in ids if t in tasks]),
    }


def _usage_dict(values: list[Any]) -> dict[str, int]:
    total = {"in": 0, "out": 0, "calls": 0}
    add_usage(values, total)
    return total


def _fmt_tokens(usage: dict[str, int], scale: float) -> str:
    if not usage["calls"]:
        return "-"
    return (f"{usage['calls']} calls, in {usage['in'] / 1000:.0f}K / out {usage['out'] / 1000:.0f}K"
            f"  (x{scale:.0f} -> in {usage['in'] * scale / 1e6:.2f}M / out {usage['out'] * scale / 1e6:.2f}M)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("offline", "online"), required=True)
    parser.add_argument("--subset", type=int, required=True, metavar="N")
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    args = parser.parse_args()
    output_root = Path(args.output_root).expanduser().resolve()

    all_ids = load_sample_ids() if args.split == "offline" else list(load_task_definitions())
    ids = subset_ids(all_ids, args.subset)
    metrics = OFFLINE_METRICS if args.split == "offline" else ONLINE_METRICS
    scale = len(all_ids) / len(ids)
    print(f"{args.split} subset N={len(ids)} of {len(all_ids)}: {', '.join(ids[:5])}, ...\n")

    header = ["model/setting", "rollout ok/mfail/miss", "eval scored/zero/miss", *metrics, "mean"]
    rows, notes = [], []
    for model, setting in _matrix(args.split):
        row = (_offline_row if args.split == "offline" else _online_row)(output_root, model, setting, ids)
        if row is None:
            continue
        # 子集里还有样本没评完时不报分数，只报计数：部分平均值会误导。
        values = [row["metrics"][metric] for metric in metrics]
        complete = row["evaluation"]["missing"] == 0 and all(value is not None for value in values)
        rows.append([
            f"{model}/{_hist(setting)}",
            "{complete}/{model_failed}/{missing}".format(**row["rollout"]),
            "{scored}/{zeroed}/{missing}".format(**row["evaluation"]),
            *[f"{value * 100:.1f}" if complete else "-" for value in values],
            f"{sum(values) / len(values) * 100:.1f}" if complete else "-",
        ])
        notes.append((f"{model}/{_hist(setting)}", row))

    if not rows:
        print("没有任何输出目录；先用 --subset N 跑 rollout 与评测。")
        return
    widths = [max(len(str(line[i])) for line in [header, *rows]) for i in range(len(header))]
    for line in [header, *rows]:
        print("  ".join(str(cell).ljust(width) for cell, width in zip(line, widths)))

    print("\nAPI 开销（实测 -> 按已跑完的样本数线性外推到全量）")
    for name, row in notes:
        # 子集没跑完时按实际跑完的样本数外推，而不是按 N。
        n_rollout = row["rollout"]["complete"] + row["rollout"]["model_failed"]
        n_eval = row["evaluation"]["scored"] + row["evaluation"]["zeroed"]
        print(f"- {name}: rollout 墙钟 {row['wallclock_s'] / 60:.1f} min"
              f"（rollout {n_rollout} 条、评测 {n_eval} 条）")
        for label in ("usage_planner", "usage_rollout", "usage_judge"):
            if label in row:
                n_done = n_eval if label == "usage_judge" else n_rollout
                print(f"    {label[6:]:8s} "
                      f"{_fmt_tokens(row[label], len(all_ids) / n_done if n_done else scale)}")


if __name__ == "__main__":
    main()
