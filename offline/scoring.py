"""Input validation, caching, per-episode evaluation, and aggregation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from offline.judges import (
    IMAGE_ENCODING,
    JUDGE_MODEL,
    PROMPTS,
    DEFAULT_MAX_TOKENS,
    OfflineJudge,
)
from utils.io import atomic_write_json, load_json
from offline.visual_similarity import dino_cosine, siglip_cosine

SCHEMA = "gui_cc_offline_evaluation"
METRICS = (
    "S_ele", "S_lay", "S_sig", "S_dino", "S_ad",
    "S_id", "S_use", "S_cp", "S_rd", "S_rap",
)
STEP_METRICS = METRICS[:7]


def evaluation_config(wm: str, setting: str, base_url: str | None,
                      judge_model: str = JUDGE_MODEL,
                      max_tokens: int = DEFAULT_MAX_TOKENS,
                      extra_config: dict | None = None) -> dict:
    """本次评测的完整配置；写进聚合文件一次，并参与 signature 计算。"""
    return {
        "schema": SCHEMA,
        "metrics": METRICS,
        "wm": wm,
        "setting": setting,
        "judge_model": judge_model,
        "base_url": (base_url or "openai-default").rstrip("/"),
        "image_encoding": IMAGE_ENCODING,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "visual_models": {
            "siglip": os.environ.get("VISUAL_SIM_SIGLIP_MODEL", "google/siglip-so400m-patch14-384"),
            "dino": os.environ.get("VISUAL_SIM_DINO_MODEL", "facebook/dinov2-giant"),
            "siglip_revision": os.environ.get("VISUAL_SIM_SIGLIP_REVISION"),
            "dino_revision": os.environ.get("VISUAL_SIM_DINO_REVISION"),
        },
        **(extra_config or {}),
    }


def build_signature(episode: dict, configuration: dict) -> str:
    """per-episode 结果缓存的 key：配置或该 episode 的输入变了就失效。"""
    payload = {
        "task": episode["task"],
        "reference": episode["reference"],
        "prompts": PROMPTS,
        "configuration": configuration,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def overall(metrics: dict[str, Any]) -> float | None:
    values = [metrics.get(metric) for metric in METRICS]
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(float(value)) and 0 <= float(value) <= 1 for value in values):
        return None
    return _mean([float(value) for value in values])


def paper_scores(metrics: dict[str, Any]) -> dict[str, float | None]:
    return {
        metric: round(float(metrics[metric]) * 100, 1)
        if isinstance(metrics.get(metric), (int, float)) else None
        for metric in (*METRICS, "Overall")
    }


def _evaluate_transition(judge: OfflineJudge, episode: dict, transition: dict) -> dict:
    # 模型加载的线程安全由 visual_similarity 内部的加载锁保证。
    sig = siglip_cosine(transition["gt_after"], transition["pred_after"])
    dino = dino_cosine(transition["gt_after"], transition["pred_after"])
    if sig is None or dino is None:
        from offline.visual_similarity import backend_status

        raise RuntimeError(f"the visual similarity backends did not both produce a score: {backend_status()}")
    ele_lay = judge.s_ele_lay(transition["gt_after"], transition["pred_after"])
    ad = judge.s_ad(transition["pred_before"], transition["pred_after"],
                    episode["task"], transition["action"])
    inverse = judge.s_id(transition["pred_before"], transition["pred_after"], transition["action"])
    use = judge.s_use(transition["pred_after"])
    metrics = {"S_ele": ele_lay["S_ele"], "S_lay": ele_lay["S_lay"],
               "S_sig": sig, "S_dino": dino, "S_ad": ad["S_ad"],
               "S_id": inverse["S_id"], "S_use": use["S_use"]}
    # details 保留 judge 的完整返回：这是事后复查"某一分为什么这么打"的唯一依据。
    return {"step": transition["step"],
            "metrics": {key: float(value) for key, value in metrics.items()},
            "details": {"S_ele_S_lay": ele_lay, "S_ad": ad,
                        "S_id": inverse, "S_use": use}}


def _trajectory_metrics(judge: OfflineJudge, episode: dict) -> tuple[dict, dict]:
    actions = [transition["action"] for transition in episode["transitions"]]
    frames = [episode["initial"]] + [transition["pred_after"] for transition in episode["transitions"]]
    # 五条判据的键名必须与 utils/prompts/judge/s_{cp,rd}_system.md 里
    # 要求 judge 返回的字段一致；改 prompt 时必须同步改这里。
    cp_keys = ["C1_step_continuity", "C2_task_anchor_consistency", "C3_state_carryover",
               "C4_navigation_context_memory", "C5_long_horizon_history"]
    rd_keys = ["C1_action_responsiveness", "C2_action_change_synchronization",
               "C3_transition_order_coherence", "C4_change_scope_control",
               "C5_no_freeze_or_temporal_degradation"]
    jobs = {
        "S_cp": lambda: judge.trajectory("S_cp", PROMPTS["S_cp_system"], episode["task"],
                                          actions, frames, cp_keys),
        "S_rd": lambda: judge.trajectory("S_rd", PROMPTS["S_rd_system"], episode["task"],
                                          actions, frames, rd_keys),
        "S_rap": lambda: judge.s_rap(episode["task"], episode["transitions"]),
    }
    details = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(job): metric for metric, job in jobs.items()}
        for future in as_completed(futures):
            metric = futures[future]
            details[metric] = future.result()
    return {metric: details[metric][metric] for metric in ("S_cp", "S_rd", "S_rap")}, details


def cache_valid(cached: dict, signature: str, n_transitions: int) -> bool:
    rows = cached.get("per_step", [])
    return (
        cached.get("schema") == SCHEMA
        and cached.get("complete") is True
        and not cached.get("error")
        and cached.get("signature") == signature
        and len(rows) == n_transitions
        and all(all(metric in row.get("metrics", {}) for metric in STEP_METRICS) for row in rows)
        and isinstance(cached.get("trajectory"), dict)
        and overall(cached.get("metrics", {})) is not None
    )


def evaluate_episode(client: OpenAI, episode: dict, signature: str, result_dir: Path,
                     judge_model: str = JUDGE_MODEL,
                     max_tokens: int = DEFAULT_MAX_TOKENS, force: bool = False) -> dict:
    """评测一个 episode。并发由调用方在 episode 层提供，这里逐 transition 串行。"""
    cache = result_dir / "evaluation.json"
    if cache.is_file() and not force:
        cached = load_json(cache)
        if cache_valid(cached, signature, len(episode["transitions"])):
            return cached
    judge = OfflineJudge(client, judge_model, max_tokens)
    rows = [
        _evaluate_transition(judge, episode, transition)
        for transition in episode["transitions"]
    ]
    metrics = {
        metric: _mean([row["metrics"][metric] for row in rows])
        for metric in STEP_METRICS
    }
    trajectory_metrics, trajectory_details = _trajectory_metrics(judge, episode)
    metrics.update({key: float(value) for key, value in trajectory_metrics.items()})
    metrics["Overall"] = overall(metrics)
    record = {"schema": SCHEMA, "sample_id": episode["sample_id"], "complete": True,
              "n_transitions": len(rows), "metrics": metrics,
              "per_step": rows, "trajectory": trajectory_details, "signature": signature}
    atomic_write_json(cache, record)
    return record


def aggregate_results(wm: str, setting: str, requested: list[str],
                      results: dict[str, dict], full: bool,
                      configuration: dict | None = None,
                      judge_model: str = JUDGE_MODEL) -> dict:
    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    errors = []
    model_failed = 0
    for sample_id in requested:
        result = results.get(sample_id) or {}
        result_metrics = result.get("metrics", {})
        if result.get("complete") is True and overall(result_metrics) is not None:
            for metric in METRICS:
                values[metric].append(float(result_metrics[metric]))
        elif result.get("failure_class") == "model":
            # 模型失败是它的能力问题：计 0 分，留在固定分母里。
            model_failed += 1
            for metric in METRICS:
                values[metric].append(0.0)
        else:
            # 基础设施失败与机器有关而非模型：阻塞聚合，修好后重跑。
            errors.append({"sample_id": sample_id,
                           "error": result.get("error", "缺少结果")})
    n_scored = len(requested) - len(errors)
    metrics = {metric: _mean(values[metric]) if values[metric] else None
               for metric in METRICS}
    complete = not errors
    metrics["Overall"] = overall(metrics) if complete and full else None
    transition_count = sum(
        int(results[sample_id].get("n_transitions", 0))
        for sample_id in requested
        if isinstance(results.get(sample_id), dict)
        and results[sample_id].get("complete") is True
    )
    return {"schema": SCHEMA, "wm": wm, "setting": setting,
            "scope": "full" if full else "partial", "complete": complete,
            "judge_model": judge_model,
            "n_episodes_requested": len(requested),
            "n_episodes_scored": n_scored,
            "n_episodes_complete": n_scored - model_failed,
            "n_transitions_complete": transition_count,
            "failure_counts": {"model": model_failed, "infra_blocked": len(errors)},
            "metrics": metrics, "paper_scores": paper_scores(metrics), "errors": errors,
            "configuration": configuration}
