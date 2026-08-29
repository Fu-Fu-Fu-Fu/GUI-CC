#!/usr/bin/env python3
"""Compute the local visual metrics S_sig / S_dino from an offline prediction tree.

No judge API is involved. The encoders are configured through the VISUAL_SIM_* environment
variables (see utils/configs/paths.env.example).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.io import atomic_write_json, load_json  # noqa: E402
from offline.data import (  # noqa: E402
    OFFLINE_DATA_ROOT,
    OFFLINE_EVALUATION_ROOT,
    OFFLINE_PREDICTIONS_ROOT,
    load_reference,
    load_sample_ids,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--setting", choices=("WM-Markov", "WM-FullHist"), required=True)
    parser.add_argument("--destination", help="结果目录（默认 outputs/offline/evaluation/<model>/<markov|fullhist>）")

    args = parser.parse_args(argv)
    history = "markov" if args.setting == "WM-Markov" else "fullhist"
    config_dir = OFFLINE_PREDICTIONS_ROOT / args.model / history
    destination = (
        Path(args.destination).expanduser().resolve()
        if args.destination
        else OFFLINE_EVALUATION_ROOT / args.model / history
    )

    sample_ids = load_sample_ids()
    run = load_json(config_dir / "run.json")
    if run.get("model", {}).get("model_id") != args.model:
        raise SystemExit("rollout 模型与 --model 不一致")
    if run.get("model", {}).get("history_setting") != args.setting:
        raise SystemExit("rollout history setting 与 --setting 不一致")
    samples = load_json(config_dir / "summary.json").get("samples", {})

    from offline.visual_similarity import backend_status, dino_cosine, siglip_cosine

    episode_metrics: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    total_transitions = 0
    for sample_id in sample_ids:
        record = samples.get(sample_id, {})
        if record.get("complete") is not True:
            if record.get("failure_class") != "model":
                raise SystemExit(f"{sample_id} 的 rollout 记录缺失或非 model failure")
            failures.append(sample_id)
            continue
        sig_values, dino_values = [], []
        for step, _row in enumerate(load_reference(OFFLINE_DATA_ROOT / sample_id)):
            gt = OFFLINE_DATA_ROOT / sample_id / f"step_{step:03d}_after.png"
            pred = config_dir / sample_id / f"step_{step:03d}" / "pred.png"
            sig = siglip_cosine(gt, pred)
            dino = dino_cosine(gt, pred)
            if sig is None or dino is None:
                raise SystemExit(f"视觉后端失败于 {sample_id} step {step}：{backend_status()}")
            sig_values.append(float(sig))
            dino_values.append(float(dino))
        if not sig_values:
            raise SystemExit(f"{sample_id} 没有可评分的 transition")
        episode_metrics[sample_id] = {
            "S_sig": sum(sig_values) / len(sig_values),
            "S_dino": sum(dino_values) / len(dino_values),
        }
        total_transitions += len(sig_values)
        print(f"[{sample_id}] S_sig={episode_metrics[sample_id]['S_sig']:.4f} "
              f"S_dino={episode_metrics[sample_id]['S_dino']:.4f}", flush=True)

    # model failure 计零分并保留在固定分母里，与 judge 评测的聚合口径一致。
    means = {}
    for metric in ("S_sig", "S_dino"):
        values = [
            0.0 if sample_id in failures else episode_metrics[sample_id][metric]
            for sample_id in sample_ids
        ]
        means[metric] = sum(values) / len(values)

    atomic_write_json(destination / "local_metric_results.json", {
        "schema": "gui_cc_offline_local_metrics",
        "model": args.model,
        "setting": args.setting,
        "n_episodes": len(sample_ids),
        "n_model_failures_zeroed": len(failures),
        "n_transitions_scored": total_transitions,
        "metrics": means,
        "paper_scores": {name: round(value * 100.0, 1) for name, value in means.items()},
        "per_episode": episode_metrics,
        "backend_status": backend_status(),
    })
    print(json.dumps({"metrics": means}, ensure_ascii=False))
    print(f"输出：{destination / 'local_metric_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
