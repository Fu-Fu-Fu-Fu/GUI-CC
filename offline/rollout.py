"""Generate autoregressive predictions for the offline benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any

from utils.adapters.base import load_image_array
from utils.adapters.registry import create_adapter, get_model_spec, resolved_model_config
from utils.config import OFFLINE_CONFIG, load_project_json
from utils.failure import classify_failure
from offline.data import (
    OFFLINE_DATA_ROOT,
    OFFLINE_PREDICTIONS_ROOT,
    OFFLINE_SAMPLES_FILE,
    load_reference,
    load_sample_ids,
)
from offline.sharding import (
    partition_sample_ids,
    sample_output_present,
    shard_worker_root,
)
from utils.io import atomic_write_json
from utils.subset import subset_ids


ROOT = Path(__file__).resolve().parents[1]
# 本地图像模型共用一个 GPU pipeline，只能单线程；API 图像模型（gpt_image2 / gemini_image）
# 没有本地 pipeline，和 HTML 模型一样按 --workers 并发。
LOCAL_IMAGE_ADAPTERS = {"flux2", "mobileworld_diffusion", "qwen_image_edit", "vimo"}


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _git_commit() -> str | None:
    """当前 commit；工作树有未提交改动时加 -dirty，避免记录误导。"""
    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if dirty else commit


def _run_record(
    spec: dict[str, Any],
    setting: str,
    endpoint: str | None,
    device: str,
    history_window: int,
) -> dict[str, Any]:
    record = {
        "schema": "gui_cc_offline_rollout",
        "model": resolved_model_config(
            spec,
            setting,
            endpoint_override=endpoint,
            device=device,
            history_window=history_window,
        ),
        "commit": _git_commit(),
    }
    # run_sha256 只用作输出目录与 resume 的配置标识。
    record["run_sha256"] = _json_hash({"model": record["model"]})
    return record


def rollout_sample(
    adapter,
    sample_id: str,
    *,
    use_cache: bool,
    run_sha256: str,
) -> dict[str, Any]:
    sample_dir = OFFLINE_DATA_ROOT / sample_id
    trajectory = load_reference(sample_dir)
    current = load_image_array(sample_dir / "initial.png")
    native_h, native_w = current.shape[:2]
    history: list[dict[str, Any]] = []
    lineage_changed = not use_cache
    steps: list[dict[str, Any]] = []
    started = time.time()

    for row in trajectory:
        step = int(row["step_id"])
        action = row["semantic_action"]
        before = current
        prediction = adapter.predict(
            sample_id=sample_id,
            step_id=step,
            before_arr=before,
            semantic_action=action,
            history=history if adapter.history_setting == "WM-FullHist" else None,
            use_cache=use_cache and not lineage_changed,
        )
        if prediction.error:
            _fc = (prediction.failure_class
                   or classify_failure(prediction.error, "request")["class"])
            steps.append({"step": step, "error": prediction.error, "failure_class": _fc})
            return {
                "sample_id": sample_id,
                "run_sha256": run_sha256,
                "complete": False,
                "n_steps": len(trajectory),
                "steps": steps,
                "error": prediction.error,
                "failure_class": _fc,
            }
        if not prediction.cached:
            lineage_changed = True
        current = load_image_array(prediction.pred_png_path)
        # 下一步必须拿样本原始分辨率当参考：renderer 用它决定 CSS 视口，
        # 让预测尺寸漂移会改变排版（gWorld 的 scale 查表会掉进 fallback）。
        # online 侧 (online/rollout.py) 同样把预测帧缩回任务帧尺寸。
        resized = current.shape[:2] != (native_h, native_w)
        if resized:
            current = np.asarray(
                Image.fromarray(current).resize((native_w, native_h), Image.LANCZOS))
        history.append({"before_arr": before, "semantic_action": action})
        steps.append({"step": step, "cached": bool(prediction.cached),
                      **({"resized_to_native": True} if resized else {})})

    return {
        "sample_id": sample_id,
        "run_sha256": run_sha256,
        "complete": True,
        "n_steps": len(trajectory),
        "wallclock_s": round(time.time() - started, 2),
        "steps": steps,
    }


def _validate_output_namespace(config_dir: Path, run: dict[str, Any]) -> None:
    """防止不同模型配置的预测混进同一个输出目录。"""
    manifest_path = config_dir / "run.json"
    if not manifest_path.exists():
        atomic_write_json(manifest_path, run)
        return
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    if previous.get("run_sha256") != run["run_sha256"]:
        raise RuntimeError(
            f"{config_dir}  already holds output from a different configuration. "
            "Use a new --output-root; predictions from different runs must never be mixed."
        )


def main() -> None:
    config = load_project_json(OFFLINE_CONFIG)
    parser = argparse.ArgumentParser(description="Generate GUI-CC offline rollout predictions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--setting", choices=["WM-NoHist", "WM-FullHist"], required=True)
    parser.add_argument("--sample-ids", help="generate only these samples (comma separated); default is all 500")
    parser.add_argument("--subset", type=int, metavar="N",
                        help="generate only the fixed evenly spaced subset of N samples (identical across models; see utils/subset.py)")
    parser.add_argument("--output-root", default=str(OFFLINE_PREDICTIONS_ROOT))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--endpoint")
    parser.add_argument("--served-model",
                        help="override the served model name of an API world model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="ignore the world-model cache and rerun the selected samples")
    parser.add_argument("--shard-count", type=int, help="number of independent shards")
    parser.add_argument("--shard-index", type=int, help="index of this shard (0-based)")
    args = parser.parse_args()

    history_window = int(config["history_window"])
    if not 1 <= history_window <= 3:
        parser.error("history_window in utils/configs/offline.json must be between 1 and 3")
    spec = get_model_spec(config, args.model)
    if args.setting not in spec["settings"]:
        parser.error(f"{args.model} does not have the setting {args.setting}")
    if args.served_model:
        spec["served_model" if "served_model" in spec else "model"] = args.served_model
    output_root = Path(args.output_root).expanduser().resolve()
    all_ids = load_sample_ids()
    sample_ids = list(all_ids)
    shard_enabled = args.shard_count is not None or args.shard_index is not None
    if shard_enabled and (args.shard_count is None or args.shard_index is None):
        parser.error("--shard-count and --shard-index must be used together")
    if shard_enabled and (args.sample_ids or args.subset):
        parser.error("shard mode cannot be combined with --sample-ids/--subset; sharding must cover the full sample set")
    if args.sample_ids:
        wanted = {value.strip() for value in args.sample_ids.split(",") if value.strip()}
        sample_ids = [sample_id for sample_id in sample_ids if sample_id in wanted]
        if not sample_ids:
            parser.error(f"--sample-ids matched no sample: {args.sample_ids}")
    if args.subset:
        sample_ids = subset_ids(all_ids, args.subset)

    if shard_enabled:
        assert args.shard_count is not None and args.shard_index is not None
        try:
            sample_ids = partition_sample_ids(
                all_ids, args.shard_count, args.shard_index
            )
        except ValueError as error:
            parser.error(str(error))
        if not sample_ids:
            parser.error("this shard got no samples; reduce --shard-count")
        output_root = shard_worker_root(
            output_root, args.model, args.setting, args.shard_count, args.shard_index
        )

    run = _run_record(spec, args.setting, args.endpoint, args.device, history_window)
    if shard_enabled:
        run["execution"] = {"mode": "sharded", "shard_count": args.shard_count}
    history_dir = "fullhist" if args.setting == "WM-FullHist" else "nohist"
    config_dir = output_root / args.model / history_dir
    _validate_output_namespace(config_dir, run)
    if shard_enabled:
        print(
            f"Offline shard {args.shard_index}/{args.shard_count} output: {config_dir}",
            flush=True,
        )
    adapter = create_adapter(
        spec,
        output_root,
        args.setting,
        endpoint_override=args.endpoint,
        device=args.device,
        history_window=history_window,
    )
    if spec["adapter"] in LOCAL_IMAGE_ADAPTERS:
        # 本地图像模型共用同一个 GPU pipeline，并发访问会互相踩踏。
        if args.workers and args.workers != 1:
            print(f"{args.model}  is a local image model; ignoring --workers {args.workers} and running with 1", flush=True)
        workers = 1
    else:
        workers = args.workers or 8
    summary_path = config_dir / "summary.json"
    previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    results = {
        sample_id: record for sample_id, record in previous.get("samples", {}).items()
        if record.get("run_sha256") == run["run_sha256"]
        and sample_output_present(config_dir, record)
    }
    pending = [sample_id for sample_id in sample_ids if args.force or sample_id not in results]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                rollout_sample,
                adapter,
                sample_id,
                use_cache=not args.force,
                run_sha256=run["run_sha256"],
            ): sample_id
            for sample_id in pending
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                results[sample_id] = future.result()
            except Exception as error:  # noqa: BLE001
                # 未捕获异常是代码或环境问题，不是模型的能力问题。
                results[sample_id] = {
                    "sample_id": sample_id,
                    "run_sha256": run["run_sha256"],
                    "complete": False,
                    "error": str(error)[:500],
                    "failure_class": "infrastructure",
                }
            atomic_write_json(summary_path, {"run": run, "samples": results})
            print(f"[{sample_id}] complete={results[sample_id].get('complete')}", flush=True)

    failed = [sample_id for sample_id in sample_ids if not results.get(sample_id, {}).get("complete")]
    model_failed = [sid for sid in failed if results.get(sid, {}).get("failure_class") == "model"]
    infra_blocked = [sid for sid in failed if results.get(sid, {}).get("failure_class") != "model"]
    print(f"Offline rollout {args.model}/{history_dir}：{len(sample_ids) - len(failed)}/{len(sample_ids)} done"
          f" (model failures {len(model_failed)}, infrastructure failures {len(infra_blocked)}）")
    if infra_blocked:
        raise SystemExit(f"{len(infra_blocked)}  offline sample(s) hit an infrastructure failure (INFRA_BLOCKED); first: {infra_blocked[0]}")


if __name__ == "__main__":
    main()
