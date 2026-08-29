"""VLM judges used by the offline evaluation."""
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

from openai import OpenAI
from utils.config import OFFLINE_CONFIG, load_project_json

from utils.prompts.judge_prompts import (
    S_AD_SYSTEM_PROMPT,
    S_AD_USER_TEMPLATE,
    S_CP_SYSTEM_PROMPT,
    S_ELE_LAY_SYSTEM_PROMPT,
    S_ELE_LAY_USER_PROMPT,
    S_ID_CATEGORIES,
    S_ID_SYSTEM_PROMPT,
    S_ID_USER_PROMPT,
    S_RAP_SYSTEM_PROMPT,
    S_RAP_USER_TEMPLATE,
    S_RD_SYSTEM_PROMPT,
    S_USE_SYSTEM_PROMPT,
    S_USE_USER_PROMPT,
    TRAJ_USER_TEMPLATE,
)


_JSON_RE = re.compile(r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}", re.DOTALL)


# 图片编码是评测协议的一部分：改这些值会改变 judge 看到的输入。
# trajectory 用更小的尺寸是因为一条消息里最多带 25 帧。
IMAGE_ENCODING = {
    "step": {"longest": 1280, "image_format": "JPEG", "jpeg_quality": 88},
    "trajectory": {"longest": 768, "image_format": "JPEG", "jpeg_quality": 82},
}
VLM_RETRIES = 3
# 单请求超时。带思考的 judge（qwen3.7/3.8 系）整段推理常超过 120s，
# 按 120s 掐断等于白花一次最贵的慢调用；快模型本来几秒返回，放宽无代价。
REQUEST_TIMEOUT = 600.0
# A safety ceiling, not a budget: the judge only returns a short JSON object, so
# truncation is the real failure mode. Reasoning tokens count against this too, and
# some gateways cut off long single requests, so keep it tight.
DEFAULT_MAX_TOKENS = 1536

# The judge only needs to return a short JSON object, so thinking mode is slower and
# more expensive for no gain. Providers that support it read this from extra_body;
# providers that do not simply ignore it.
JUDGE_EXTRA_BODY = {"enable_thinking": False}


def action_description(action: dict) -> str:
    action_type = action.get("type")
    target = action.get("target", "")
    if action_type == "tap":
        return f'tap on "{target}"' if target else "tap"
    if action_type == "long_press":
        return f'long-press on "{target}"' if target else "long-press"
    if action_type == "type_text":
        text = action.get("text", "")
        return f'type "{text}" into "{target}"' if target else f'type "{text}"'
    if action_type == "scroll":
        return (
            f'scroll {action.get("direction", "")} on "{target}"'
            if target
            else f'scroll {action.get("direction", "")}'
        )
    if action_type == "navigate_home":
        return "press Home"
    if action_type == "navigate_back":
        return "press Back"
    if action_type == "open_app":
        return f"open app: {action.get('app_name') or target}"
    if action_type == "wait":
        return "wait"
    return action_type or "?"


def _parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, json.JSONDecodeError):
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    best: tuple[dict, int] | None = None
    for match in _JSON_RE.finditer(cleaned):
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and (best is None or len(match.group(0)) > best[1]):
            best = (parsed, len(match.group(0)))
    return best[0] if best else None


def _is_rate_limit_error(error: object) -> bool:
    text = str(error).lower()
    return any(token in text for token in ("429", "throttling", "rate limit", "限流"))


def image_data_url(path: Path, longest: int, image_format: str, jpeg_quality: int) -> str:
    with Image.open(path) as image:
        converted = image.convert("RGB")
    width, height = converted.size
    if max(width, height) > longest:
        scale = longest / max(width, height)
        converted = converted.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    output_format = image_format.strip().upper().replace("JPG", "JPEG")
    if output_format == "PNG":
        converted.save(buffer, format="PNG")
        mime = "image/png"
    else:
        converted.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        mime = "image/jpeg"
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def step_image_url(path: Path) -> str:
    return image_data_url(path, **IMAGE_ENCODING["step"])


def trajectory_image_url(path: Path) -> str:
    return image_data_url(path, **IMAGE_ENCODING["trajectory"])


def call_vlm(
    client: OpenAI,
    model: str,
    messages: list,
    retries: Optional[int] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> dict:
    """调用已配置的 judge 模型，绝不替换为其他模型。"""
    retries = VLM_RETRIES if retries is None else retries
    request_timeout = REQUEST_TIMEOUT
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=request_timeout,
                extra_body=JUDGE_EXTRA_BODY,
            )
            raw = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            return {
                "raw": raw,
                "parsed": _parse_json(raw) or {},
                "requested_model": model,
                "api_model": getattr(response, "model", None),
                # token 用量进结果文件：小样本试跑时据此估全量的 API 开销。
                "usage": usage.model_dump(exclude_none=True) if usage is not None else None,
            }
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt < retries:
                time.sleep(min(60, 5 * (attempt + 1)) if _is_rate_limit_error(error) else min(30, 2**attempt))
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
    return parsed


def vlm_error_payload(result: dict, default_error: str = "json_parse_failed") -> Optional[dict]:
    if not result.get("error") and result.get("parsed"):
        return None
    return {
        "error": result.get("error") or default_error,
        "raw_response": result.get("raw", ""),
        "_api_model": result.get("api_model"),
        "_requested_model": result.get("requested_model"),
    }


JUDGE_MODEL = load_project_json(OFFLINE_CONFIG)["judge_model"]
PROMPTS = {
    "S_ele_lay_system": S_ELE_LAY_SYSTEM_PROMPT,
    "S_ele_lay_user": S_ELE_LAY_USER_PROMPT,
    "S_ad_system": S_AD_SYSTEM_PROMPT,
    "S_ad_user": S_AD_USER_TEMPLATE,
    "S_id_system": S_ID_SYSTEM_PROMPT,
    "S_id_user": S_ID_USER_PROMPT,
    "S_use_system": S_USE_SYSTEM_PROMPT,
    "S_use_user": S_USE_USER_PROMPT,
    "S_cp_system": S_CP_SYSTEM_PROMPT,
    "S_rd_system": S_RD_SYSTEM_PROMPT,
    "trajectory_user": TRAJ_USER_TEMPLATE,
    "S_rap_system": S_RAP_SYSTEM_PROMPT,
    "S_rap_user": S_RAP_USER_TEMPLATE,
}


class JudgeError(RuntimeError):
    """导致 episode 无法评分的 API 或响应错误。"""

    def __init__(self, metric: str, message: str):
        super().__init__(f"{metric}: {message}")
        self.metric = metric


def public_action(value: Any) -> Any:
    """Action 离开仓库前递归移除私有标注字段。"""
    if isinstance(value, dict):
        return {
            key: public_action(item)
            for key, item in value.items()
            if isinstance(key, str) and not key.startswith("_")
        }
    if isinstance(value, list):
        return [public_action(item) for item in value]
    return value


def strict_binary(parsed: dict[str, Any], keys: list[str], score_key: str | None) -> dict[str, int]:
    missing = [key for key in keys if key not in parsed]
    if missing:
        raise ValueError(f"缺少二值字段：{', '.join(missing)}")
    values: dict[str, int] = {}
    for key in keys:
        value = parsed[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0, 1):
            raise ValueError(f"{key} 必须是数值 0 或 1")
        values[key] = int(value)
    if score_key is not None:
        # score 字段与各二值字段冗余：以 criteria 之和为准。judge 偶尔会把
        # 冗余的汇总写错（约 0.2%），为此作废整个样本的全部已付费调用不成比例；
        # 不一致时记录标记供事后审计，不作废。
        score = parsed.get(score_key)
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or int(score) != score or int(score) != sum(values.values())):
            parsed["_score_field_mismatch"] = score
    return values


def _number(parsed: dict[str, Any], key: str, low: float, high: float) -> float:
    value = parsed.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} 必须是数值")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{key} 必须位于 [{low}, {high}] 范围内")
    return value


def _required_text(parsed: dict[str, Any], key: str) -> str:
    value = parsed.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} 必须是非空字符串")
    return value.strip()


def _expected_action(action: dict) -> str:
    """judge 该推断出的动作类型。

    数据集用 tap 加一句 "Open the Settings app." 表示打开应用，judge 只看图
    分不出这和普通 tap 的区别，所以这类动作按 open_app 对比。其余动作类型
    与 judge 的输出词汇一致，直接使用。
    """
    action_type = str(action.get("type", "")).strip().lower()
    target = " ".join(
        str(action.get(key, "") or "")
        for key in ("target", "low_level_instruction", "app_name")
    ).lower()
    if action_type == "tap" and re.search(r"\bopen\b.*\bapp\b", target):
        return "open_app"
    return action_type


class OfflineJudge:
    def __init__(self, client: OpenAI, model: str = JUDGE_MODEL,
                 max_tokens: int = DEFAULT_MAX_TOKENS):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def _request(self, metric: str, messages: list[dict]) -> dict:
        result = call_vlm(self.client, self.model, messages, max_tokens=self.max_tokens)
        error = vlm_error_payload(result)
        if error:
            raise JudgeError(metric, str(error.get("error", "响应无效")))
        # Some gateways do not echo the model name; only a present, mismatched echo is an error.
        if result.get("api_model") not in (None, self.model):
            raise JudgeError(
                metric,
                f"judge 响应模型 {result.get('api_model')!r} 与请求模型 {self.model!r} 不一致",
            )
        parsed = parsed_with_meta(result)
        parsed["api_model"] = parsed.pop("_api_model", None)
        parsed["requested_model"] = parsed.pop("_requested_model", self.model)
        parsed["raw_response"] = result.get("raw", "")
        parsed["usage"] = result.get("usage")
        if parsed.get("error"):
            raise JudgeError(metric, str(parsed["error"]))
        return parsed

    def s_ele_lay(self, reference: Path, prediction: Path) -> dict:
        messages = [{"role": "system", "content": S_ELE_LAY_SYSTEM_PROMPT}, {
            "role": "user", "content": [
                {"type": "text", "text": "Reference Image (Ground Truth, Image 1):"},
                {"type": "image_url", "image_url": {"url": step_image_url(reference)}},
                {"type": "text", "text": "Candidate Image (Prediction, Image 2):"},
                {"type": "image_url", "image_url": {"url": step_image_url(prediction)}},
                {"type": "text", "text": S_ELE_LAY_USER_PROMPT},
            ],
        }]
        parsed = self._request("S_ele/S_lay", messages)
        element = _number(parsed, "element_alignment_score", 1.0, 10.0)
        layout = _number(parsed, "structural_fidelity_score", 1.0, 10.0)
        _required_text(parsed, "reasoning")
        return {"S_ele": (element - 1.0) / 9.0, "S_lay": (layout - 1.0) / 9.0,
                "raw_scores": {"S_ele": element, "S_lay": layout}, "judge": parsed}

    def s_ad(self, before: Path, after: Path, task: str, action: dict) -> dict:
        action = public_action(action)
        user = S_AD_USER_TEMPLATE.format(
            instruction=task, semantic_description=action_description(action),
            action_json=json.dumps(action, ensure_ascii=False),
        )
        messages = [{"role": "system", "content": S_AD_SYSTEM_PROMPT}, {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": step_image_url(before)}},
            {"type": "image_url", "image_url": {"url": step_image_url(after)}},
            {"type": "text", "text": user},
        ]}]
        parsed = self._request("S_ad", messages)
        score = _number(parsed, "score", 0.0, 10.0)
        _required_text(parsed, "reasoning")
        return {"S_ad": score / 10.0, "raw_score": score, "action": action, "judge": parsed}

    def s_id(self, before: Path, after: Path, action: dict) -> dict:
        messages = [{"role": "system", "content": S_ID_SYSTEM_PROMPT}, {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": step_image_url(before)}},
            {"type": "image_url", "image_url": {"url": step_image_url(after)}},
            {"type": "text", "text": S_ID_USER_PROMPT},
        ]}]
        parsed = self._request("S_id", messages)
        inferred = str(parsed.get("inferred_action", "")).strip().lower()
        if inferred not in S_ID_CATEGORIES:
            raise JudgeError("S_id", f"inferred_action 不在允许的类别内：{inferred!r}")
        _required_text(parsed, "reasoning")
        expected = _expected_action(action)
        return {"S_id": float(inferred == expected), "expected": expected,
                "inferred": inferred, "judge": parsed}

    def s_use(self, prediction: Path) -> dict:
        messages = [{"role": "system", "content": S_USE_SYSTEM_PROMPT}, {"role": "user", "content": [
            {"type": "text", "text": S_USE_USER_PROMPT},
            {"type": "image_url", "image_url": {"url": step_image_url(prediction)}},
        ]}]
        parsed = self._request("S_use", messages)
        # 键名与 utils/prompts/judge/s_use_system.md 要求 judge 返回的字段一致。
        keys = ["C1_valid_mobile_gui", "C2_render_integrity", "C3_text_legibility",
                "C4_component_coherence", "C5_interaction_readiness"]
        values = strict_binary(parsed, keys, "score")
        _required_text(parsed, "reasoning")
        failure_modes = parsed.get("failure_modes")
        if not isinstance(failure_modes, list) or not all(
            isinstance(item, str) for item in failure_modes
        ):
            raise JudgeError("S_use", "failure_modes 必须是字符串列表")
        return {"S_use": sum(values.values()) / 5.0, "criteria": values, "judge": parsed}

    def trajectory(self, metric: str, system: str, task: str,
                   actions: list[dict], frames: list[Path], keys: list[str]) -> dict:
        lines = "\n".join(
            f"- step {index + 1}: {action_description(public_action(action))}"
            for index, action in enumerate(actions)
        )
        user = TRAJ_USER_TEMPLATE.format(
            instruction=task, action_lines=lines or "(none)", metric_name=metric
        )
        content: list[dict] = [{"type": "text", "text": user}]
        for index, frame in enumerate(frames):
            content.extend([
                {"type": "text", "text": f"Frame {index}."},
                {"type": "image_url", "image_url": {"url": trajectory_image_url(frame)}},
            ])
        parsed = self._request(metric, [{"role": "system", "content": system},
            {"role": "user", "content": content}])
        values = strict_binary(parsed, keys, "score")
        _required_text(parsed, "reasoning")
        return {metric: sum(values.values()) / 5.0, "criteria": values, "judge": parsed}

    def s_rap(self, task: str, transitions: list[dict]) -> dict:
        rows, prefix = [], 0
        for index, transition in enumerate(transitions):
            action = public_action(transition["action"])
            following = public_action(transitions[index + 1]["action"]) if index + 1 < len(transitions) else None
            terminal = "" if following else (
                "\nFor this final step, use the task instruction and Image 2 (GT final UI) "
                "as the semantic reference for task completion."
            )
            user = S_RAP_USER_TEMPLATE.format(
                instruction=task, step_index=index + 1, total_steps=len(transitions),
                action_desc=action_description(action),
                action_json=json.dumps(action, ensure_ascii=False),
                next_action_desc=action_description(following) if following else
                    "TERMINAL: no next reference action. Judge task completion for P3.",
                terminal_block=terminal,
            )
            content: list[dict] = [{"type": "text", "text": user}]
            for path in (transition["gt_before"], transition["gt_after"],
                         transition["pred_before"], transition["pred_after"]):
                content.append({"type": "image_url", "image_url": {"url": step_image_url(path)}})
            parsed = self._request("S_rap", [{"role": "system", "content": S_RAP_SYSTEM_PROMPT},
                {"role": "user", "content": content}])
            # 键名与 utils/prompts/judge/s_rap_system.md 要求 judge 返回的字段一致。
            keys = ["P1_precondition_supported", "P2_action_effect_supported",
                    "P3_next_action_supported_or_terminal"]
            values = strict_binary(parsed, keys + ["passed"], None)
            expected_pass = int(all(values[key] == 1 for key in keys))
            if values["passed"] != expected_pass:
                parsed["_passed_field_mismatch"] = values["passed"]
            _required_text(parsed, "evidence")
            allowed_failures = {
                "wrong_current_context", "action_target_missing", "action_not_reflected",
                "wrong_next_stage", "next_action_not_supported", "invalid_ui",
                "too_distorted", "terminal_not_completed", "terminal_wrong_app",
                "terminal_wrong_content", "none",
            }
            if parsed.get("failure_reason") not in allowed_failures:
                raise JudgeError("S_rap", "failure_reason 不在允许范围内")
            rows.append({"step": index + 1, "action": action, "next_action": following,
                         "result": values, "judge": parsed})
            if not expected_pass:
                break
            prefix += 1
        return {"S_rap": prefix / len(transitions), "supported_prefix": prefix,
                "n_steps": len(transitions),
                "first_failure_step": rows[-1]["step"] if prefix < len(transitions) else None,
                "per_step": rows}
