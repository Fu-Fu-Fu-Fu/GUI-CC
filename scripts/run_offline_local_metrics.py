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
    parser.add_argument("--setting", choices=("WM-NoHist", "WM-FullHist"), required=True)
    parser.add_argument("--destination",
                        help="output directory (default outputs/offline/evaluation/<model>/<nohist|fullhist>)")
    parser.add_argument("--subset", type=int, metavar="N",
                        help="score only the fixed evenly spaced subset of N samples "
                             "(identical across models; see utils/subset.py)")
    parser.add_argument("--sample-ids", help="score only these samples (comma separated)")

    args = parser.parse_args(argv)
    history = "nohist" if args.setting == "WM-NoHist" else "fullhist"
    config_dir = OFFLINE_PREDICTIONS_ROOT / args.model / history
    destination = (
        Path(args.destination).expanduser().resolve()
        if args.destination
        else OFFLINE_EVALUATION_ROOT / args.model / history
    )

    all_ids = load_sample_ids()
    sample_ids = list(all_ids)
    if args.subset:
        from utils.subset import subset_ids
        sample_ids = subset_ids(all_ids, args.subset)
    elif args.sample_ids:
        wanted = {value.strip() for value in args.sample_ids.split(",") if value.strip()}
        sample_ids = [sid for sid in all_ids if sid in wanted]
        if not sample_ids:
            raise SystemExit(f"--sample-ids matched no sample: {args.sample_ids}")
    run = load_json(config_dir / "run.json")
    if run.get("model", {}).get("model_id") != args.model:
        raise SystemExit("rollout model does not match --model")
    if run.get("model", {}).get("history_setting") != args.setting:
        raise SystemExit("rollout history setting does not match --setting")
    samples = load_json(config_dir / "summary.json").get("samples", {})

    from offline.visual_similarity import backend_status, dino_cosine, siglip_cosine

    episode_metrics: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    missing: list[str] = []
    total_transitions = 0
    for sample_id in sample_ids:
        record = samples.get(sample_id, {})
        if record.get("complete") is not True:
            if record.get("failure_class") == "model":
                failures.append(sample_id)
            else:
                missing.append(sample_id)
            continue
        sig_values, dino_values = [], []
        for step, _row in enumerate(load_reference(OFFLINE_DATA_ROOT / sample_id)):
            gt = OFFLINE_DATA_ROOT / sample_id / f"step_{step:03d}_after.png"
            pred = config_dir / sample_id / f"step_{step:03d}" / "pred.png"
            sig = siglip_cosine(gt, pred)
            dino = dino_cosine(gt, pred)
            if sig is None or dino is None:
                raise SystemExit(f"visual backend failed on {sample_id} step {step}: {backend_status()}")
            sig_values.append(float(sig))
            dino_values.append(float(dino))
        if not sig_values:
            raise SystemExit(f"{sample_id} has no scorable transition")
        episode_metrics[sample_id] = {
            "S_sig": sum(sig_values) / len(sig_values),
            "S_dino": sum(dino_values) / len(dino_values),
        }
        total_transitions += len(sig_values)
        print(f"[{sample_id}] S_sig={episode_metrics[sample_id]['S_sig']:.4f} "
              f"S_dino={episode_metrics[sample_id]['S_dino']:.4f}", flush=True)

    # model failure 计零分并保留在固定分母里，与 judge 评测的聚合口径一致。
    scored_ids = [sid for sid in sample_ids if sid not in missing]
    if not scored_ids:
        raise SystemExit(
            "no sample has a finished rollout; run offline.rollout first "
            "(add --subset N to match a trial run)"
        )
    means = {}
    for metric in ("S_sig", "S_dino"):
        values = [
            0.0 if sample_id in failures else episode_metrics[sample_id][metric]
            for sample_id in scored_ids
        ]
        means[metric] = sum(values) / len(values)

    atomic_write_json(destination / "local_metric_results.json", {
        "schema": "gui_cc_offline_local_metrics",
        "model": args.model,
        "setting": args.setting,
        "n_episodes_requested": len(sample_ids),
        "n_episodes_scored": len(scored_ids),
        "n_episodes_skipped_no_rollout": len(missing),
        "n_model_failures_zeroed": len(failures),
        "n_transitions_scored": total_transitions,
        "metrics": means,
        "paper_scores": {name: round(value * 100.0, 1) for name, value in means.items()},
        "per_episode": episode_metrics,
        "backend_status": backend_status(),
    })
    print(json.dumps({"metrics": means}, ensure_ascii=False))
    if missing:
        print(f"skipped {len(missing)} sample(s) without a finished rollout "
              f"(first: {missing[0]})")
    print(f"written to {destination / 'local_metric_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
