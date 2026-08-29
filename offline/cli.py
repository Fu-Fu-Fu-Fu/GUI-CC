"""Command-line entry point for offline evaluation."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from PIL import Image

from offline.data import (
    OFFLINE_DATA_ROOT,
    OFFLINE_EVALUATION_ROOT,
    OFFLINE_PREDICTIONS_ROOT,
    gt_after_path,
    gt_before_path,
    load_reference,
    load_sample_ids,
    load_samples,
    pred_after_path,
)
from offline.judges import DEFAULT_MAX_TOKENS, REQUEST_TIMEOUT
from offline.scoring import (
    JUDGE_MODEL,
    aggregate_results,
    build_signature,
    evaluate_episode,
    evaluation_config,
)
from utils.config import env
from utils.io import atomic_write_json, load_json
from utils.subset import subset_ids

# 默认为 None，直接建 OpenAI 客户端。API 与 GPU 分处两台机器时，外部驱动可以
# 在此换成自己的客户端；本仓库不依赖任何外部模块。
API_CLIENT = None


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an offline GUI world-model rollout.")
    parser.add_argument("--model", required=True, help="world-model identifier")
    parser.add_argument("--setting", choices=["WM-Markov", "WM-FullHist"], default="WM-Markov")
    parser.add_argument("--sample-ids", help="evaluate only these samples (comma separated); default is all 500")
    parser.add_argument("--subset", type=int, metavar="N",
                        help="evaluate only the fixed evenly spaced subset of N samples (identical across models; see utils/subset.py)")
    parser.add_argument("--api-key", default=env("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=env("OPENAI_BASE_URL"))
    parser.add_argument("--parallel", type=int, default=8,
                        help="episodes evaluated in parallel, i.e. concurrent requests to the judge API")
    parser.add_argument("--judge-model", default=JUDGE_MODEL,
                        help="VLM used for judging; defaults to the model reported in the paper")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="safety ceiling for one judge reply; reasoning models spend this budget on reasoning tokens too")
    parser.add_argument("--force", action="store_true",
                        help="ignore any cached evaluation.json and re-run the judge")
    parser.add_argument("--dry-run", action="store_true", help="validate the inputs without calling the judge")
    return parser.parse_args()


def _select_samples(args: argparse.Namespace) -> tuple[list[str], bool]:
    """不传 --sample-ids/--subset 评测全部样本并产出正式结果；传了则是子集试跑，不出 Overall。"""
    all_ids = load_sample_ids()
    if args.subset:
        return subset_ids(all_ids, args.subset), True
    if not args.sample_ids:
        return all_ids, False
    wanted = {value.strip() for value in args.sample_ids.split(",") if value.strip()}
    selected = [sample_id for sample_id in all_ids if sample_id in wanted]
    if not selected:
        raise ValueError(f"--sample-ids matched no sample: {args.sample_ids}")
    return selected, True


def _output_names(is_partial: bool) -> tuple[str, str, str]:
    """下划线前缀让这三个文件在 ls 里排在 500 个样本目录之前。"""
    suffix = "_partial" if is_partial else ""
    return f"_aggregate{suffix}.json", f"_results{suffix}.json", f"_preflight{suffix}.json"


def _model_failure_step(record: dict) -> int:
    """model failure 记录的最后一步就是失败步 (rollout 在此中断）。"""
    return record["steps"][-1]["step"]


def _load_episodes(sample_ids: list[str], wm: str, setting: str,
                   samples: dict[str, dict]) -> list[dict]:
    """装配每个 episode 的评分输入：动作 + 四张图（gt/pred × before/after）的路径。

    data/ 下的数据集是仓库内不可变内容（由 scripts/validate_data.py 保证），这里直接信任。
    """
    episodes = []
    history_dir = "fullhist" if setting == "WM-FullHist" else "markov"
    for sample_id in sample_ids:
        sample_dir = OFFLINE_DATA_ROOT / sample_id
        prediction_dir = OFFLINE_PREDICTIONS_ROOT / wm / history_dir / sample_id
        reference = load_reference(sample_dir)
        initial = sample_dir / "initial.png"
        transitions, previous_prediction = [], initial
        for step, row in enumerate(reference):
            prediction = pred_after_path(prediction_dir, step)
            transitions.append({
                "step": step + 1, "action": row["semantic_action"],
                "gt_before": gt_before_path(sample_dir, step),
                "gt_after": gt_after_path(sample_dir, step),
                "pred_before": previous_prediction, "pred_after": prediction,
            })
            previous_prediction = prediction
        episodes.append({
            "sample_id": sample_id, "task": samples[sample_id]["task_instruction"],
            "initial": initial, "reference": reference, "transitions": transitions,
        })
    return episodes


def _preflight(episodes: list[dict], model_failure_steps: dict[str, int]) -> dict:
    """在发起任何付费 judge 请求前解码全部预测图片。

    只检查模型产物：预测树可能缺步或写坏，而 data/ 下的 GT 图片是已验证的仓库内容。
    """
    errors, n_transitions, skipped = [], 0, 0
    for episode in episodes:
        failure_step = model_failure_steps.get(episode["sample_id"])
        predictions: set[Path] = set()
        for transition in episode["transitions"]:
            n_transitions += 1
            # transition["step"] 从 1 开始，failure_step 从 0 开始：
            # 第 N 步失败意味着第 N+1 个 transition 起都没有产物。
            if failure_step is not None and transition["step"] > failure_step:
                skipped += 1
                continue
            predictions.update((transition["pred_before"], transition["pred_after"]))
        for path in sorted(predictions):
            try:
                with Image.open(path) as image:
                    image.load()
            except Exception as error:  # noqa: BLE001
                errors.append({"sample_id": episode["sample_id"],
                               "path": str(path), "error": str(error)[:300]})
    return {"n_episodes": len(episodes), "n_transitions": n_transitions,
            "n_model_failure_transitions_skipped": skipped,
            "errors": errors, "passed": not errors}


def run(args: argparse.Namespace) -> int:
    sample_ids, is_partial = _select_samples(args)
    history_dir = "fullhist" if args.setting == "WM-FullHist" else "markov"
    prediction_config_dir = OFFLINE_PREDICTIONS_ROOT / args.model / history_dir
    output_dir = OFFLINE_EVALUATION_ROOT / args.model / history_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_name, results_name, preflight_name = _output_names(is_partial)

    episodes = _load_episodes(sample_ids, args.model, args.setting, load_samples())

    rollout_run = load_json(prediction_config_dir / "run.json")
    if rollout_run["model"]["model_id"] != args.model:
        raise ValueError("rollout model does not match --model")
    if rollout_run["model"]["history_setting"] != args.setting:
        raise ValueError("rollout history setting does not match --setting")
    rollout_summary = load_json(prediction_config_dir / "summary.json")

    # 模型失败的样本按协议记 0 分进固定分母；基础设施失败必须修复后重跑。
    rollout_errors, model_failures = [], {}
    for sample_id in sample_ids:
        record = rollout_summary["samples"].get(sample_id, {})
        if record.get("complete") is True:
            continue
        if record.get("failure_class") == "model":
            model_failures[sample_id] = {
                "record": record,
                "failure_step": _model_failure_step(record),
            }
        else:
            reason = record.get("error") or "the rollout summary has no record for this sample"
            rollout_errors.append({
                "sample_id": sample_id,
                "error": f"INFRA_BLOCKED: {reason}",
            })

    scorable_episodes = [
        episode for episode in episodes if episode["sample_id"] not in model_failures
    ]
    # 在昂贵的图片 preflight 之前确认 judge 凭据（全是模型失败时不调 judge，无需 key）。
    if scorable_episodes and not args.dry_run and API_CLIENT is None and not args.api_key:
        raise ValueError("OPENAI_API_KEY is not set; pass --api-key or set the environment variable")

    report = _preflight(
        episodes,
        {sample_id: item["failure_step"] for sample_id, item in model_failures.items()},
    )
    report["errors"] = rollout_errors + report["errors"]
    report["passed"] = not report["errors"]
    report["scope"] = "partial" if is_partial else "full"
    atomic_write_json(output_dir / preflight_name, report)
    print(
        f"judge={args.judge_model} 配置={args.model}/{args.setting} "
        f"episodes={len(sample_ids)} scope={report['scope']}"
    )
    if report["errors"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    if args.dry_run:
        print(f"preflight passed: {output_dir / preflight_name}")
        return 0
    configuration = evaluation_config(
        args.model, args.setting, args.base_url, args.judge_model, args.max_tokens,
        {"rollout_run_sha256": rollout_run.get("run_sha256")},
    )
    signatures = {
        episode["sample_id"]: build_signature(episode, configuration)
        for episode in scorable_episodes
    }
    client = API_CLIENT or OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=REQUEST_TIMEOUT,
    ) if scorable_episodes else None
    results: dict[str, dict] = {
        sample_id: {
            "sample_id": sample_id,
            "complete": False,
            "failure_class": "model",
            "error": item["record"].get("error"),
            "failure_step": item["failure_step"],
        }
        for sample_id, item in model_failures.items()
    }
    started = time.time()

    def evaluate(episode: dict) -> tuple[str, dict]:
        sample_id = episode["sample_id"]
        result = evaluate_episode(
            client=client,
            episode=episode,
            signature=signatures[sample_id],
            result_dir=output_dir / sample_id,
            judge_model=args.judge_model,
            max_tokens=args.max_tokens,
            force=args.force,
        )
        return sample_id, result

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {
            pool.submit(evaluate, episode): episode["sample_id"]
            for episode in scorable_episodes
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                _, results[sample_id] = future.result()
                print(f"[{sample_id}] Overall={results[sample_id]['metrics']['Overall']}")
            except Exception as error:  # noqa: BLE001
                results[sample_id] = {
                    "sample_id": sample_id,
                    "complete": False,
                    "error": f"{type(error).__name__}: {str(error)[:500]}",
                }
                print(f"[{sample_id}] error: {error}")
            atomic_write_json(output_dir / results_name, results)

    aggregate = aggregate_results(
        args.model, args.setting, sample_ids, results, full=not is_partial,
        configuration=configuration, judge_model=args.judge_model,
    )
    atomic_write_json(output_dir / aggregate_name, aggregate)
    # 循环内的写盘只覆盖有 episode 被评测的情况；全是模型失败时这里是唯一写入点。
    atomic_write_json(output_dir / results_name, results)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"elapsed={time.time() - started:.0f}s output={output_dir}")
    return 0 if aggregate["complete"] else 1


def main() -> None:
    try:
        code = run(_arguments())
    except (FileNotFoundError, ValueError) as error:
        print(f"configuration error: {error}")
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()
