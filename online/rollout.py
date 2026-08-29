"""Run the fixed GUI agent on a world model."""
from __future__ import annotations

import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from utils.adapters.registry import create_adapter, get_model_spec, resolved_model_config
from utils.config import ONLINE_CONFIG, env, load_project_json
from online.actions import is_terminal_action
from online.agent import (
    MAX_RETRY_DEFAULT,
    PLANNER_TEMPERATURE_DEFAULT,
    REQUEST_TIMEOUT_DEFAULT,
    PlannerAgent,
)
from online.trajectory import (
    ONLINE_DATA_ROOT,
    ONLINE_OUTPUT_ROOT,
    ONLINE_SAMPLES_FILE,
    build_trajectory,
    load_task_definitions,
    select_task_ids,
)
from utils.failure import classify_failure
from utils.io import atomic_write_json
from utils.subset import subset_ids


ROOT = Path(__file__).resolve().parents[1]
LOCAL_IMAGE_ADAPTERS = {"flux2", "mobileworld_diffusion", "qwen_image_edit", "vimo"}

# 单机跑时为 None，PlannerAgent 自己建 OpenAI 客户端。GPU 与 API 分处两台机器时，
# 外部驱动可以在调用 main() 前把它换成自己的客户端；本仓库不依赖任何外部模块。
PLANNER_CLIENT = None



def _split_ids(value: str | None) -> list[str] | None:
    return [item.strip() for item in value.split(",") if item.strip()] if value else None


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def portable_path(path: str | os.PathLike[str] | None, root: Path) -> str:
    """若产物位于 ``root`` 下，则返回相对于该目录的路径。"""
    if not path:
        return ""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return str(candidate)


def _run_record(
    task_file: Path,
    task_defs: dict[str, dict],
    seeds_root: Path,
    model: dict[str, Any],
    planner: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "schema": "gui_cc_online_rollout",
        "tasks_file": task_file.relative_to(ROOT).as_posix() if task_file.is_relative_to(ROOT) else str(task_file),
        "model": model,
        "planner": planner,
    }
    # run_sha256 只用作输出目录与 resume 的配置标识。
    record["run_sha256"] = _json_hash(
        {"model": model, "planner": planner}
    )
    return record


def rollout_one_task(
    task_id: str,
    task_def: dict,
    agent: PlannerAgent,
    wm,
    output_dir: Path,
    seeds_root: Path,
    *,
    use_cache: bool,
    run_sha256: str,
    verbose: bool = True,
) -> dict[str, Any]:
    seed_path = seeds_root / task_id / "initial.png"
    with Image.open(seed_path) as image:
        current = image.convert("RGB")
    width, height = current.size
    task_dir = output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    agent.reset(task_def["instruction"])
    history: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    error: str | None = None
    failure_class: str | None = None
    terminal_status: str | None = None
    started = time.time()

    for step in range(task_def["step_budget"]):
        step_dir = task_dir / f"step_{step:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        frame_path = step_dir / "frame.png"
        current.save(frame_path)
        raw_response, action = agent.step(current)
        (step_dir / "agent_response.txt").write_text(raw_response or "", encoding="utf-8")
        agent_evidence = getattr(agent, "last_step_evidence", {})
        evidence_path = step_dir / "agent_evidence.json"
        atomic_write_json(evidence_path, agent_evidence)
        if action is None:
            failure = agent_evidence.get("failure") if isinstance(agent_evidence, dict) else None
            error = (
                failure.get("kind")
                if isinstance(failure, dict) and failure.get("kind")
                else "agent_failed_without_action"
            )
            fc = classify_failure(error, "agent")
            failure_class = fc["class"]
            trajectory.append({
                "step": step,
                "frame": portable_path(frame_path, output_dir),
                "agent_response": raw_response,
                "agent_evidence": portable_path(evidence_path, output_dir),
                "semantic_action": None,
                "failure": failure,
                "error": error,
                "failure_class": failure_class,
            })
            break

        atomic_write_json(step_dir / "action.json", {
            "plan": agent_evidence.get("plan") if isinstance(agent_evidence, dict) else None,
            "semantic_action": action,
        })
        record = {
            "step": step,
            "frame": portable_path(frame_path, output_dir),
            "agent_response": raw_response,
            "agent_evidence": portable_path(evidence_path, output_dir),
            "semantic_action": action,
        }
        if is_terminal_action(action):
            terminal_status = str(action.get("value") or "unknown")
            record["terminated"] = True
            trajectory.append(record)
            break

        before = np.asarray(current)
        prediction = wm.predict(
            sample_id=task_id,
            step_id=step,
            before_arr=before,
            semantic_action=action,
            history=history if wm.history_setting == "WM-FullHist" else None,
            use_cache=use_cache,
        )
        if prediction.error:
            error = prediction.error
            failure_class = getattr(prediction, "failure_class", None)
            if failure_class is None:
                fc = classify_failure(error, "wm")
                failure_class = fc["class"]
            record["wm_error"] = error
            record["failure_class"] = failure_class
            trajectory.append(record)
            break
        prediction_path = Path(prediction.pred_png_path)
        if not prediction_path.is_file():
            error = "world_model_returned_no_image"
            fc = classify_failure(error, "wm")
            failure_class = fc["class"]
            record["wm_error"] = error
            record["failure_class"] = failure_class
            trajectory.append(record)
            break
        with Image.open(prediction_path) as image:
            next_frame = image.convert("RGB")
        pred_original_size = list(next_frame.size)
        pred_resized_to_frame = next_frame.size != (width, height)
        if pred_resized_to_frame:
            next_frame = next_frame.resize((width, height), Image.Resampling.LANCZOS)
        if float(np.asarray(next_frame).std()) < 0.5:
            error = "world_model_returned_near_constant_image"
            fc = classify_failure(error, "wm")
            failure_class = fc["class"]
            record["wm_error"] = error
            record["failure_class"] = failure_class
            trajectory.append(record)
            break

        record.update({
            "pred_png": portable_path(prediction_path, output_dir),
            "pred_original_size": pred_original_size,
            "pred_resized_to_frame": pred_resized_to_frame,
            "pred_html": portable_path(getattr(prediction, "pred_html_path", ""), output_dir),
            "wm_cached": bool(prediction.cached),
        })
        trajectory.append(record)
        history.append({"before_arr": before, "semantic_action": action})
        current = next_frame
        if verbose:
            print(f"[{task_id}] step {step + 1}：{action.get('type')}", flush=True)

    final_frame = task_dir / "final_frame.png"
    current.save(final_frame)
    summary = {
        "task_id": task_id,
        "instruction": task_def["instruction"],
        "step_budget": task_def["step_budget"],
        "run_sha256": run_sha256,
        "complete": error is None,
        "error": error,
        "failure_class": failure_class,
        "terminal_status": terminal_status,
        "steps_taken": len(trajectory),
        "total_time_s": round(time.time() - started, 2),
        "trajectory": trajectory,
    }
    atomic_write_json(task_dir / "trajectory.json", summary)
    return summary


def _load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _merge_result(path: Path, run: dict[str, Any], task_id: str, result: dict[str, Any]) -> None:
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        payload = _load_results(path)
        if payload and payload.get("_RUN", {}).get("run_sha256") != run["run_sha256"]:
            raise RuntimeError("rollout_results.json changed to a different run while this job was running")
        payload["_RUN"] = run
        payload[task_id] = result
        atomic_write_json(path, payload)


def _ensure_namespace(output_dir: Path, results_path: Path, run: dict[str, Any]) -> dict[str, Any]:
    if not results_path.exists():
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(f"refusing to use an unrecorded output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(results_path, {"_RUN": run})
        return {"_RUN": run}
    results = _load_results(results_path)
    if results.get("_RUN", {}).get("run_sha256") != run["run_sha256"]:
        raise RuntimeError(f"{output_dir} holds output from a different configuration; use a new --output-dir")
    return results


def main() -> None:
    config = load_project_json(ONLINE_CONFIG)
    parser = argparse.ArgumentParser(description="Run a GUI-CC online rollout")
    parser.add_argument("--model", required=True)
    parser.add_argument("--setting", choices=["WM-NoHist", "WM-FullHist"], required=True)
    parser.add_argument("--task-ids", help="generate only these tasks (comma separated); default is all 200")
    parser.add_argument("--subset", type=int, metavar="N",
                        help="generate only the fixed evenly spaced subset of N samples (identical across models; see utils/subset.py)")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--output-root",
        help="root directory for online output; shard mode writes under its .shards namespace",
    )
    parser.add_argument("--shard-count", type=int, help="number of independent shards")
    parser.add_argument("--shard-index", type=int, help="index of this shard (0-based)")
    parser.add_argument("--planner-url", default=env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--planner-api-key", default=env("OPENAI_API_KEY"))
    parser.add_argument("--planner-model", default=config["planner_model"],
                        help="VLM driving the agent; defaults to the model reported in the paper")
    parser.add_argument("--endpoint", help="override the world-model API endpoint")
    parser.add_argument("--served-model",
                        help="override the served model name of an API world model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8,
                        help="tasks run in parallel; tasks are independent of each other")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    history_window = int(config["history_window"])
    if not 1 <= history_window <= 3:
        parser.error("history_window in utils/configs/online.json must be between 1 and 3")
    spec = get_model_spec(config, args.model)
    if args.setting not in spec["settings"]:
        parser.error(f"{args.model} does not have the setting {args.setting}")
    if args.served_model:
        spec["served_model" if "served_model" in spec else "model"] = args.served_model
    task_file = ONLINE_SAMPLES_FILE
    seeds_root = ONLINE_DATA_ROOT
    task_defs = load_task_definitions(task_file)
    all_ids = list(task_defs)
    shard_enabled = args.shard_count is not None or args.shard_index is not None
    if shard_enabled and (args.shard_count is None or args.shard_index is None):
        parser.error("--shard-count and --shard-index must be used together")
    if shard_enabled and (args.task_ids or args.subset):
        parser.error("shard mode cannot be combined with --task-ids/--subset; sharding must cover the full task set")
    if shard_enabled and args.output_dir:
        parser.error("shard mode uses --output-root, not --output-dir")
    task_ids = select_task_ids(task_defs, _split_ids(args.task_ids))
    if args.subset:
        task_ids = subset_ids(all_ids, args.subset)
    if shard_enabled:
        from online.sharding import partition_task_ids

        assert args.shard_count is not None and args.shard_index is not None
        try:
            task_ids = partition_task_ids(
                all_ids, args.shard_count, args.shard_index
            )
        except ValueError as error:
            parser.error(str(error))
        if not task_ids:
            parser.error("this online shard got no tasks; reduce --shard-count")
    history_dir = "fullhist" if args.setting == "WM-FullHist" else "nohist"
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else ONLINE_OUTPUT_ROOT
    )
    if shard_enabled:
        from online.sharding import shard_worker_root

        assert args.shard_count is not None and args.shard_index is not None
        worker_root = shard_worker_root(
            output_root, args.model, args.setting, args.shard_count, args.shard_index
        )
        output_dir = worker_root / args.model / history_dir
    else:
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else output_root / args.model / history_dir
        )
    model_config = resolved_model_config(
        spec, args.setting, endpoint_override=args.endpoint, device=args.device,
        history_window=history_window,
    )
    planner_config = {
        "model": args.planner_model, "endpoint": args.planner_url,
        "max_tokens": config["planner_max_tokens"],
        "temperature": PLANNER_TEMPERATURE_DEFAULT,
        "max_retry": MAX_RETRY_DEFAULT,
        "timeout_s": REQUEST_TIMEOUT_DEFAULT,
    }
    run = _run_record(task_file, task_defs, seeds_root, model_config, planner_config)
    if shard_enabled:
        run["execution"] = {"mode": "sharded", "shard_count": args.shard_count}
    results_path = output_dir / "rollout_results.json"
    results = _ensure_namespace(output_dir, results_path, run)
    if shard_enabled:
        print(
            f"Online shard {args.shard_index}/{args.shard_count} output: {output_dir}",
            flush=True,
        )
    wm = create_adapter(
        spec, output_dir / "wm_cache", args.setting, endpoint_override=args.endpoint,
        device=args.device,
        history_window=history_window,
    )
    def make_agent() -> PlannerAgent:
        """One agent per task: it holds that task's instruction and action history, so it cannot be shared."""
        return PlannerAgent(
            planner_base_url=args.planner_url,
            planner_model=args.planner_model,
            planner_api_key=args.planner_api_key,
            planner_max_tokens=config["planner_max_tokens"],
            client=PLANNER_CLIENT,
        )

    def already_done(task_id: str) -> bool:
        previous = results.get(task_id)
        if args.force or not isinstance(previous, dict) or previous.get("complete") is not True:
            return False
        if previous.get("run_sha256") != run["run_sha256"]:
            return False
        try:
            build_trajectory(output_dir, task_id, previous)
        except (OSError, ValueError):
            return False
        return True

    def run_task(task_id: str) -> dict[str, Any]:
        try:
            return rollout_one_task(
                task_id, task_defs[task_id], make_agent(), wm, output_dir, seeds_root,
                use_cache=not args.force, run_sha256=run["run_sha256"],
                verbose=not args.quiet,
            )
        except Exception as error:  # noqa: BLE001
            return {
                "task_id": task_id,
                "run_sha256": run["run_sha256"], "complete": False, "error": str(error)[:500],
                "failure_class": classify_failure(str(error), "agent")["class"],
            }

    if spec["adapter"] in LOCAL_IMAGE_ADAPTERS:
        # 本地图像模型共用同一个 GPU pipeline：只把 predict 串行化，任务保持并发，
        # 让 planner 调用（单次 60-90s）相互重叠，墙钟由 GPU 生成总时长决定。
        gpu_lock = threading.Lock()
        original_predict = wm.predict

        def locked_predict(*p_args, **p_kwargs):
            with gpu_lock:
                return original_predict(*p_args, **p_kwargs)

        wm.predict = locked_predict
        print(f"{args.model}  is a local image model: GPU generation is serialized while tasks stay parallel", flush=True)
    workers = args.workers

    failed: list[str] = []
    pending = [task_id for task_id in task_ids if not already_done(task_id)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(run_task, task_id): task_id for task_id in pending}
        for future in as_completed(futures):
            task_id = futures[future]
            result = future.result()
            _merge_result(results_path, run, task_id, result)
            results[task_id] = result
            if not result.get("complete"):
                failed.append(task_id)

    print(f"Online rollout {args.model}/{history_dir}：done {len(task_ids) - len(failed)}/{len(task_ids)}")
    model_failed = [
        tid for tid in failed
        if results.get(tid, {}).get("failure_class") == "model"
    ]
    infra_blocked = [
        tid for tid in failed
        if results.get(tid, {}).get("failure_class") != "model"
    ]
    if infra_blocked:
        raise SystemExit(
            f"{len(infra_blocked)}  online task(s) hit an infrastructure failure (INFRA_BLOCKED); "
            f"first: {infra_blocked[0]}"
        )
    if model_failed:
        print(f"  of which {len(model_failed)} task(s) failed on the model side (scored 0 in the fixed denominator, not blocking)")


if __name__ == "__main__":
    main()
