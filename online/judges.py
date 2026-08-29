"""VLM judges for the six online metrics reported in the paper."""
from __future__ import annotations

import base64
import io
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from utils.config import env
from utils.prompts.judge_prompts import (
    S_AD_SYSTEM_PROMPT,
    S_AD_USER_TEMPLATE,
    S_CP_SYSTEM_PROMPT,
    S_ID_CATEGORIES,
    S_ID_SYSTEM_PROMPT,
    S_ID_USER_PROMPT,
    S_MP_SYSTEM_PROMPT,
    S_MP_USER_TEMPLATE,
    S_RD_SYSTEM_PROMPT,
    S_USE_SYSTEM_PROMPT,
    S_USE_USER_PROMPT,
    TRAJ_USER_TEMPLATE,
)



def _judge_extra_body() -> dict:
    """Provider-specific request fields, empty unless JUDGE_EXTRA_BODY_JSON is set."""
    raw = (env("JUDGE_EXTRA_BODY_JSON", "") or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"JUDGE_EXTRA_BODY_JSON is not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError("JUDGE_EXTRA_BODY_JSON must be a JSON object")
    return parsed


REQUEST_TIMEOUT = 600.0
# 与 offline 保持同一套 judge 请求参数：max_tokens 是防跑飞的安全上限，不是预算，
# judge 只回一小段 JSON，截断才是真正的故障模式（推理模型的 reasoning token 也计入)。
DEFAULT_MAX_TOKENS = 4096  # see offline/judges.py
# 与 offline 一致：qwen judge 显式关闭思考模式（见 offline/judges.py)。
JUDGE_EXTRA_BODY = _judge_extra_body()


def parse_json_object(text: str) -> Optional[dict]:
    """解析模型回复中的第一个 JSON 对象。"""
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(cleaned)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        for match in re.finditer(r"\{", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def image_data_url(path: Path, longest_edge: int, jpeg_quality: int) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"找不到图像：{path}")
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = min(1.0, longest_edge / max(width, height))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def step_image_url(path: Path) -> str:
    return image_data_url(
        path,
        **IMAGE_ENCODING["step"],
    )


def trajectory_image_url(path: Path) -> str:
    return image_data_url(
        path,
        **IMAGE_ENCODING["trajectory"],
    )


def vlm_call(
    client,
    model: str,
    messages: list,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    retries: int = 3,
) -> dict:
    """调用 judge 模型并返回原始回复和解析后的 JSON。"""
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=REQUEST_TIMEOUT,
                extra_body=JUDGE_EXTRA_BODY,
            )
            raw = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            return {
                "raw": raw,
                "parsed": parse_json_object(raw) or {},
                "requested_model": model,
                "api_model": getattr(response, "model", None),
                # token 用量进结果文件：小样本试跑时据此估全量的 API 开销。
                "usage": usage.model_dump(exclude_none=True) if usage is not None else None,
            }
        except Exception as error:  # noqa: BLE001 - API 客户端可能抛出多种异常。
            last_error = error
            if attempt < retries:
                time.sleep(min(30, 2 ** attempt))
    return {
        "raw": "",
        "parsed": {},
        "error": str(last_error)[:300],
        "requested_model": model,
        "api_model": None,
    }


def parsed_with_meta(result: dict) -> dict:
    parsed = dict(result.get("parsed") or {})
    if result.get("api_model"):
        parsed["_api_model"] = result["api_model"]
    if result.get("requested_model"):
        parsed["_requested_model"] = result["requested_model"]
    parsed["_usage"] = result.get("usage")
    return parsed


def vlm_error_payload(result: dict, default_error: str = "json_parse_failed") -> Optional[dict]:
    if result.get("error"):
        return {
            "error": result["error"],
            "raw_response": result.get("raw", ""),
            "_api_model": result.get("api_model"),
            "_requested_model": result.get("requested_model"),
        }
    if not result.get("parsed"):
        return {
            "error": default_error,
            "raw_response": result.get("raw", ""),
            "_api_model": result.get("api_model"),
            "_requested_model": result.get("requested_model"),
        }
    return None


def action_description(action: Optional[dict]) -> str:
    if not action:
        return "(none)"
    instruction = str(action.get("low_level_instruction", "") or "").strip()
    if instruction:
        return instruction
    action_type = str(action.get("type", "unknown"))
    target = str(action.get("target", "") or "").strip()
    if action_type == "tap":
        return f'tap on "{target}"' if target else "tap"
    if action_type == "long_press":
        return f'long-press on "{target}"' if target else "long-press"
    if action_type == "type_text":
        text = str(action.get("text", "") or "")
        return f'type "{text}" into "{target}"' if target else f'type "{text}"'
    if action_type == "scroll":
        direction = str(action.get("direction", "") or "")
        return f'scroll {direction} on "{target}"' if target else f"scroll {direction}".strip()
    if action_type == "navigate_home":
        return "press Home"
    if action_type == "navigate_back":
        return "press Back"
    if action_type == "open_app":
        return f'open app "{action.get("app_name") or target}"'
    if action_type == "wait":
        return "wait"
    return f'{action_type} on "{target}"' if target else action_type

# 图片编码是评测协议的一部分：改这些值会改变 judge 看到的输入。
# trajectory 用更小的尺寸是因为一条消息里最多带 25 frames。
IMAGE_ENCODING = {
    "step": {"longest_edge": 1280, "jpeg_quality": 88},
    "trajectory": {"longest_edge": 768, "jpeg_quality": 82},
}

PAPER_METRICS = ("S_ad", "S_id", "S_use", "S_cp", "S_rd", "S_mp")
from online.trajectory import MAX_TRAJECTORY_FRAMES  # 单一事实来源（trajectory.py)

_S_USE_KEYS = (
    "C1_valid_mobile_gui",
    "C2_render_integrity",
    "C3_text_legibility",
    "C4_component_coherence",
    "C5_interaction_readiness",
)
_S_CP_KEYS = (
    "C1_step_continuity",
    "C2_task_anchor_consistency",
    "C3_state_carryover",
    "C4_navigation_context_memory",
    "C5_long_horizon_history",
)
_S_RD_KEYS = (
    "C1_action_responsiveness",
    "C2_action_change_synchronization",
    "C3_transition_order_coherence",
    "C4_change_scope_control",
    "C5_no_freeze_or_temporal_degradation",
)


def prompt_payload() -> dict[str, str]:
    """返回所有可能影响 Online 分数的 prompt 字节串。"""
    return {
        "S_ad_system": S_AD_SYSTEM_PROMPT,
        "S_ad_user": S_AD_USER_TEMPLATE,
        "S_id_system": S_ID_SYSTEM_PROMPT,
        "S_id_user": S_ID_USER_PROMPT,
        "S_use_system": S_USE_SYSTEM_PROMPT,
        "S_use_user": S_USE_USER_PROMPT,
        "S_cp_system": S_CP_SYSTEM_PROMPT,
        "S_rd_system": S_RD_SYSTEM_PROMPT,
        "trajectory_user": TRAJ_USER_TEMPLATE,
        "S_mp_system": S_MP_SYSTEM_PROMPT,
        "S_mp_user": S_MP_USER_TEMPLATE,
    }


def image_encoding_payload() -> dict:
    return IMAGE_ENCODING


def _error(kind: str, message: str, result: Optional[dict] = None) -> dict:
    error: dict[str, Any] = {"kind": kind, "message": message}
    if result:
        error["raw_response"] = result.get("raw", "")
        error["api_model"] = result.get("api_model")
    return {"score": None, "error": error}


def _parsed_result(result: dict) -> tuple[Optional[dict], Optional[dict]]:
    payload = vlm_error_payload(result)
    if payload:
        kind = "api_error" if result.get("error") else "json_parse_error"
        return None, _error(kind, str(payload.get("error", kind)), result)
    if result.get("api_model") not in (None, result.get("requested_model")):
        return None, _error(
            "model_identity_error",
            f"judge response model {result.get('api_model')!r} does not match the requested model "
            f"{result.get('requested_model')!r}",
            result,
        )
    parsed = parsed_with_meta(result)
    if not isinstance(parsed, dict):
        return None, _error("json_parse_error", "the judge reply is not a JSON object", result)
    if parsed.get("error"):
        return None, _error("judge_response_error", str(parsed["error"]), result)
    return parsed, None


def _strict_number(value: Any, low: float, high: float) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < low or number > high:
        return None
    return number


def _required_text(parsed: dict, key: str, result: dict) -> Optional[dict]:
    value = parsed.get(key)
    if not isinstance(value, str) or not value.strip():
        return _error("evidence_parse_error", f"{key}  must be a non-empty string", result)
    return None


def _strict_binary_score(
    parsed: dict,
    keys: tuple[str, ...],
    result: dict,
    *,
    require_failure_modes: bool = False,
) -> dict:
    values: list[int] = []
    for key in keys:
        value = parsed.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0, 1):
            return _error("binary_parse_error", f"{key}  must be the number 0 or 1", result)
        values.append(int(value))
    # score 与各二值字段冗余，以 criteria 之和为准；judge 偶写错冗余汇总（约 0.2%)，
    #时记录标记不作废（与 offline/judges.py 的 strict_binary 一致)。
    reported = parsed.get("score")
    if (
        isinstance(reported, bool)
        or not isinstance(reported, (int, float))
        or not math.isfinite(float(reported))
        or float(reported) != sum(values)
    ):
        parsed["_score_field_mismatch"] = reported
    evidence_error = _required_text(parsed, "reasoning", result)
    if evidence_error:
        return evidence_error
    if require_failure_modes and (
        not isinstance(parsed.get("failure_modes"), list)
        or not all(isinstance(item, str) for item in parsed["failure_modes"])
    ):
        return _error("evidence_parse_error", "failure_modes must be a list of strings", result)
    return {"score": sum(values) / len(keys), "details": parsed, "raw_response": result.get("raw", "")}


def _expected_action(action: dict) -> str:
    """judge 该推断出的动作类型；打开应用在数据里表现为一次 tap 加一句说明。"""
    action_type = str(action.get("type", "")).strip().lower()
    target = " ".join(
        str(action.get(key, "") or "") for key in ("target", "low_level_instruction", "app_name")
    ).strip().lower()
    if action_type == "tap" and re.search(r"\bopen\b.*\bapp\b", target):
        return "open_app"
    return action_type


def judge_s_ad(client, model: str, before: Path, after: Path, instruction: str, action: dict,
               max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    user_text = S_AD_USER_TEMPLATE.format(
        instruction=instruction,
        semantic_description=action_description(action),
        action_json=json.dumps(action, ensure_ascii=False),
    )
    result = vlm_call(client, model, max_tokens=max_tokens, messages=[
        {"role": "system", "content": S_AD_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": step_image_url(before)}},
            {"type": "image_url", "image_url": {"url": step_image_url(after)}},
            {"type": "text", "text": user_text},
        ]},
    ])
    parsed, error = _parsed_result(result)
    if error:
        return error
    score = _strict_number(parsed.get("score"), 0.0, 10.0)
    if score is None:
        return _error("score_parse_error", "score 必须是 [0, 10]的数值", result)
    evidence_error = _required_text(parsed, "reasoning", result)
    if evidence_error:
        return evidence_error
    return {"score": score / 10.0, "raw_score": score, "details": parsed, "raw_response": result.get("raw", "")}


def judge_s_id(client, model: str, before: Path, after: Path, action: dict,
               max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    result = vlm_call(client, model, max_tokens=max_tokens, messages=[
        {"role": "system", "content": S_ID_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": step_image_url(before)}},
            {"type": "image_url", "image_url": {"url": step_image_url(after)}},
            {"type": "text", "text": S_ID_USER_PROMPT},
        ]},
    ])
    parsed, error = _parsed_result(result)
    if error:
        return error
    inferred_raw = parsed.get("inferred_action")
    if not isinstance(inferred_raw, str):
        return _error("category_parse_error", "inferred_action 必须是字符串", result)
    inferred = inferred_raw.strip().lower()
    if inferred not in S_ID_CATEGORIES:
        return _error("category_parse_error", "inferred_action 不在允许的类别范围内", result)
    evidence_error = _required_text(parsed, "reasoning", result)
    if evidence_error:
        return evidence_error
    expected = _expected_action(action)
    return {
        "score": 1.0 if inferred == expected else 0.0,
        "expected": expected,
        "inferred": inferred,
        "details": parsed,
        "raw_response": result.get("raw", ""),
    }


def judge_s_use(client, model: str, after: Path,
                max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    result = vlm_call(client, model, max_tokens=max_tokens, messages=[
        {"role": "system", "content": S_USE_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": S_USE_USER_PROMPT},
            {"type": "image_url", "image_url": {"url": step_image_url(after)}},
        ]},
    ])
    parsed, error = _parsed_result(result)
    return error or _strict_binary_score(
        parsed, _S_USE_KEYS, result, require_failure_modes=True
    )


def _trajectory_messages(
    system_prompt: str,
    metric_name: str,
    instruction: str,
    actions: list[Optional[dict]],
    frames: list[Path],
) -> list:
    if len(frames) > MAX_TRAJECTORY_FRAMES:
        raise ValueError(f"an online trajectory may contain at most {MAX_TRAJECTORY_FRAMES} frames")
    if len(frames) != len(actions) + 1:
        raise ValueError("the frame count must be exactly one more than the action count")
    action_lines = "\n".join(
        f"- step {index + 1}: {action_description(action) if action else 'unavailable due to rollout error'}"
        for index, action in enumerate(actions)
    ) or "(none)"
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": TRAJ_USER_TEMPLATE.format(
            instruction=instruction,
            action_lines=action_lines,
            metric_name=metric_name,
        ),
    }]
    for index, frame in enumerate(frames):
        content.extend([
            {"type": "text", "text": f"Frame {index}."},
            {"type": "image_url", "image_url": {"url": trajectory_image_url(frame)}},
        ])
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]


def _judge_trajectory(
    client,
    model: str,
    instruction: str,
    actions: list[Optional[dict]],
    frames: list[Path],
    metric_name: str,
    system_prompt: str,
    keys: tuple[str, ...],
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    result = vlm_call(
        client,
        model,
        _trajectory_messages(system_prompt, metric_name, instruction, actions, frames),
        max_tokens=max_tokens,
    )
    parsed, error = _parsed_result(result)
    return error or _strict_binary_score(parsed, keys, result)


def judge_s_cp(client, model: str, instruction: str, actions: list[Optional[dict]], frames: list[Path],
               max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    return _judge_trajectory(
        client, model, instruction, actions, frames, "S_cp", S_CP_SYSTEM_PROMPT, _S_CP_KEYS,
        max_tokens=max_tokens,
    )


def judge_s_rd(client, model: str, instruction: str, actions: list[Optional[dict]], frames: list[Path],
               max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    return _judge_trajectory(
        client, model, instruction, actions, frames, "S_rd", S_RD_SYSTEM_PROMPT, _S_RD_KEYS,
        max_tokens=max_tokens,
    )


def _milestone_type(value: list) -> str:
    return "+".join(str(item) for item in value)


def _parse_milestone(result: dict, earliest: int, frame_count: int) -> dict:
    parsed, error = _parsed_result(result)
    if error:
        return error
    passed = parsed.get("passed")
    first = parsed.get("first_satisfied_frame")
    if isinstance(passed, bool) or not isinstance(passed, (int, float)) or passed not in (0, 1):
        return _error("binary_parse_error", "passed  must be the number 0 or 1", result)
    passed = int(passed)
    if type(first) is not int:
        return _error("frame_parse_error", "first_satisfied_frame must be an integer", result)
    evidence_error = _required_text(parsed, "evidence", result)
    if evidence_error:
        return evidence_error
    if passed == 0:
        if first != -1:
            return _error("frame_parse_error", "a failed milestone must use frame -1", result)
    elif first < earliest or first >= frame_count:
        return _error(
            "frame_parse_error",
            f"first_satisfied_frame {first} is outside the actual frame range [{earliest}, {frame_count - 1}]",
            result,
        )
    return {
        "score": float(passed),
        "passed": passed,
        "first_satisfied_frame": first,
        "details": parsed,
        "raw_response": result.get("raw", ""),
    }


def judge_s_mp(client, model: str, task: dict, frames: list[Path],
               max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    if not frames or len(frames) > MAX_TRAJECTORY_FRAMES:
        return _error("trajectory_error", f"S_mp needs 1 to {MAX_TRAJECTORY_FRAMES} actual frames")
    milestones = task.get("milestones")
    if not isinstance(milestones, list) or not milestones:
        return _error("milestone_error", "the task has no ordered milestones")
    per_milestone: list[dict] = []
    earliest = 0
    passed_prefix = 0
    for index, milestone in enumerate(milestones):
        if not isinstance(milestone, dict):
            return _error("milestone_error", f"milestone {index} is not an object")
        text = S_MP_USER_TEMPLATE.format(
            instruction=task["instruction"],
            milestone_id=milestone.get("id", index),
            milestone_type=_milestone_type(milestone.get("type", "")),
            milestone_assertion=milestone.get("assertion", ""),
            earliest_allowed_frame=earliest,
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for frame_index, frame in enumerate(frames):
            content.extend([
                {"type": "text", "text": f"Frame {frame_index}."},
                {"type": "image_url", "image_url": {"url": trajectory_image_url(frame)}},
            ])
        result = vlm_call(client, model, max_tokens=max_tokens, messages=[
            {"role": "system", "content": S_MP_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ])
        parsed = _parse_milestone(result, earliest, len(frames))
        parsed["milestone_id"] = str(milestone.get("id", index))
        per_milestone.append(parsed)
        if parsed.get("error"):
            return {"score": None, "error": parsed["error"], "milestones": per_milestone}
        if parsed["passed"] == 0:
            break
        passed_prefix += 1
        earliest = parsed["first_satisfied_frame"]
    return {
        "score": passed_prefix / len(milestones),
        "passed_prefix": passed_prefix,
        "milestone_count": len(milestones),
        "milestones": per_milestone,
        "frames_used": list(range(len(frames))),
    }
