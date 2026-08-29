#!/usr/bin/env python3
"""Aggregate the complete offline and online result matrices into JSON and CSV."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OFFLINE_METRICS = (
    "S_ele", "S_lay", "S_sig", "S_dino", "S_ad",
    "S_id", "S_use", "S_cp", "S_rd", "S_rap", "Overall",
)
ONLINE_METRICS = ("S_ad", "S_id", "S_use", "S_cp", "S_rd", "S_mp", "Overall")


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require(condition: bool, message: str, path: Path) -> None:
    if not condition:
        raise ValueError(f"{message}: {path}")


def _matrix(name: str) -> list[tuple[str, str]]:
    config = _load(ROOT / "utils" / "configs" / f"{name}.json")
    return [
        (model["id"], setting)
        for model in config["models"]
        for setting in model["settings"]
    ]


def _history_dir(setting: str) -> str:
    return "fullhist" if setting == "WM-FullHist" else "markov"


def _offline_rows(output_root: Path) -> list[dict]:
    rows = []
    for model, setting in _matrix("offline"):
        path = (
            output_root / "offline" / "evaluation" / model / _history_dir(setting)
            / "_aggregate.json"
        )
        aggregate = _load(path)
        _require(aggregate.get("complete") is True, "offline aggregate 不完整", path)
        _require(aggregate.get("scope") == "full", "offline 结果不是完整评测", path)
        _require(aggregate.get("wm") == model, "offline 模型与目录不一致", path)
        _require(aggregate.get("setting") == setting, "offline setting 与目录不一致", path)
        scores = aggregate.get("paper_scores")
        _require(
            isinstance(scores, dict)
            and all(isinstance(scores.get(metric), (int, float)) for metric in OFFLINE_METRICS),
            "offline paper_scores 不完整",
            path,
        )
        rows.append({"model": model, "setting": setting, **{
            metric: scores[metric] for metric in OFFLINE_METRICS
        }})
    return rows


def _online_rows(output_root: Path) -> list[dict]:
    rows = []
    for model, setting in _matrix("online"):
        path = (
            output_root / "online" / model / _history_dir(setting)
            / "evaluation_results.json"
        )
        result = _load(path)
        run = result.get("run", {})
        aggregate = result.get("aggregate", {})
        _require(run.get("mode") == "full", "online 结果不是完整评测", path)
        _require(aggregate.get("status") == "complete", "online aggregate 未完成", path)
        _require(run.get("rollout", {}).get("model", {}).get("model_id") == model,
                 "online rollout 模型与目录不一致", path)
        scores = aggregate.get("paper_scores")
        _require(
            isinstance(scores, dict)
            and all(isinstance(scores.get(metric), (int, float)) for metric in ONLINE_METRICS),
            "online paper_scores 不完整",
            path,
        )
        rows.append({"model": model, "setting": setting, **{
            metric: scores[metric] for metric in ONLINE_METRICS
        }})
    return rows


def _write(name: str, rows: list[dict], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": f"gui_cc_{name}_results",
        "n_configurations": len(rows),
        "scores": "percentage",
        "rows": rows,
    }
    (destination / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (destination / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="收集完整的 GUI-CC 结果矩阵。")
    parser.add_argument(
        "--split", choices=("offline", "online", "all"), default="all",
        help="选择要收集的实验分支。",
    )
    parser.add_argument("--output-root", help="实验输出根目录。")
    parser.add_argument("--destination", help="结果 JSON 和 CSV 的保存目录。")
    args = parser.parse_args()
    output_value = Path(args.output_root).expanduser() if args.output_root else ROOT / "outputs"
    output_root = (
        output_value.resolve()
        if output_value.is_absolute()
        else (ROOT / output_value).resolve()
    )
    destination_value = Path(args.destination).expanduser() if args.destination else output_root / "results"
    destination = (
        destination_value.resolve()
        if destination_value.is_absolute()
        else (ROOT / destination_value).resolve()
    )
    if args.split in {"offline", "all"}:
        _write("offline", _offline_rows(output_root), destination)
    if args.split in {"online", "all"}:
        _write("online", _online_rows(output_root), destination)
    print(f"Collected results: {destination}")


if __name__ == "__main__":
    main()
