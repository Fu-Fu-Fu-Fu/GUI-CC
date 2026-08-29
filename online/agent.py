"""Single-stage GUI agent: the planner emits the action and absolute pixel coordinates directly.

The planner (called through an OpenAI-compatible API) sees the screenshot, the task instruction,
and the action history, and returns a structured action: the type comes from the dataset
vocabulary and the coordinates are absolute pixels on the screenshot, following the OpenAI
computer-use convention. No separate grounding model or coordinate-space conversion is needed.

Online rollout drives this through ``step(frame) -> (raw_response, semantic_action)``.
"""
from __future__ import annotations

import base64
import json
import time
from io import BytesIO
from typing import Optional

from openai import OpenAI
from PIL import Image

from utils.prompts.paper_prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_TEMPLATE,
)
from online.actions import (
    format_agent_response,
    plan_to_semantic_action,
    short_action_description,
)

# 协议级生成参数默认值（写入 run record，rollout 与此保持单一事实来源）
PLANNER_TEMPERATURE_DEFAULT = 1.0
MAX_RETRY_DEFAULT = 15
REQUEST_TIMEOUT_DEFAULT = 120.0


def _pil_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img = image.convert("RGB") if image.mode != "RGB" else image
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


# `computer` 工具：沿用 OpenAI computer-use 的约定（单动作对象、绝对像素坐标），
# 动作集按 Android 与数据集词表调整（tap/long_press/navigate_* 等）。
COMPUTER_TOOL = {
    "type": "function",
    "function": {
        "name": "computer",
        "description": "Perform one action on the Android phone screen.",
        "parameters": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["tap", "long_press", "scroll", "type_text", "navigate_home",
                             "navigate_back", "open_app", "wait", "terminate", "answer"],
                    "description": "The action to perform.",
                },
                "coordinate": {
                    "type": "array", "items": {"type": "integer"},
                    "description": "Absolute pixel [x, y] on the screenshot; "
                                   "required for tap/long_press/scroll.",
                },
                "target": {"type": "string", "description": "Short description of the element acted on."},
                "direction": {"type": "string", "enum": ["up", "down", "left", "right"],
                              "description": "Finger movement; required for scroll."},
                "text": {"type": "string", "description": "Text to type; required for type_text."},
                "app_name": {"type": "string", "description": "App to open; required for open_app."},
                "status": {"type": "string", "enum": ["success", "failure"],
                           "description": "Outcome; required for terminate."},
                "answer_text": {"type": "string", "description": "Final answer; required for answer."},
                "thinking": {"type": "string", "description": "One-sentence reasoning."},
            },
            "required": ["type"],
        },
    },
}


def _plan_from_tool_calls(tool_calls) -> tuple[Optional[dict], Optional[str]]:
    """从 `computer` 工具调用里取出动作参数并校验。"""
    if not tool_calls:
        return None, "planner 未调用 computer 工具"
    call = tool_calls[0]
    if call.function.name != "computer":
        return None, f"unexpected tool: {call.function.name}"
    try:
        plan = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as error:
        return None, f"工具参数不是合法 JSON：{error}"
    _normalise_coordinate(plan)
    error = _plan_error(plan)
    return (None, error) if error else (plan, None)


def _normalise_coordinate(plan: dict) -> None:
    """把坐标统一成 plan["x"], plan["y"] 两个数。

    schema 要求 coordinate: [x, y]（qwen 系 GUI 模型的原生习惯）；实测模型偶尔
    也会把数组塞进 x 字段。两种形状都收，其余交给 _plan_error 校验。
    """
    for key in ("coordinate", "x"):
        value = plan.get(key)
        if (isinstance(value, (list, tuple)) and len(value) == 2
                and all(isinstance(item, (int, float)) and not isinstance(item, bool)
                        for item in value)):
            plan["x"], plan["y"] = value
            return


def _failure_kind(validation_error: str) -> str:
    if validation_error.startswith("响应模型"):
        return "planner_model_identity_error"
    if validation_error.startswith("unsupported action type"):
        return "planner_unsupported_action"
    if validation_error.startswith("planner 未调用"):
        return "planner_no_tool_call"
    return "planner_validation_failed"


ACTION_TYPES = {
    "tap", "long_press", "scroll", "type_text", "navigate_home",
    "navigate_back", "open_app", "wait", "terminate", "answer",
}
POINT_ACTIONS = {"tap", "long_press", "scroll"}


def _plan_error(plan: dict) -> str | None:
    action = str(plan.get("type") or "").strip().lower()
    if action not in ACTION_TYPES:
        return f"unsupported action type: {action or '(empty)'}"
    if action in POINT_ACTIONS:
        for key in ("x", "y"):
            value = plan.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"{action} requires integer pixel {key}"
    if action == "scroll" and str(plan.get("direction") or "").lower() not in {
        "up", "down", "left", "right"
    }:
        return "scroll requires direction=up|down|left|right"
    if action == "type_text" and not isinstance(plan.get("text"), str):
        return "type_text requires a text string"
    if action == "open_app" and not str(plan.get("app_name") or "").strip():
        return "open_app requires app_name"
    if action == "terminate" and plan.get("status") not in {"success", "failure"}:
        return "terminate requires status=success|failure"
    if action == "answer" and not str(plan.get("answer_text") or "").strip():
        return "answer requires answer_text"
    return None


# ─── 智能体 ────────────────────────────────────────────────────────────

class PlannerAgent:
    """单阶段 GUI agent：planner 直接输出动作与绝对像素坐标。"""

    def __init__(self,
                 planner_base_url: str,
                 planner_model: str,
                 planner_api_key: str,
                 planner_temperature: float = PLANNER_TEMPERATURE_DEFAULT,
                 planner_max_tokens: int = 4096,
                 timeout: float = REQUEST_TIMEOUT_DEFAULT,
                 max_retry: int = MAX_RETRY_DEFAULT,
                 client=None):
        self.planner_model = planner_model
        self.planner_temperature = planner_temperature
        self.planner_max_tokens = planner_max_tokens
        self.max_retry = max_retry
        self.planner = client or OpenAI(
            api_key=planner_api_key, base_url=planner_base_url, timeout=timeout)
        self.instruction = ""
        self.action_history: list[str] = []
        self.last_step_evidence: dict = {}

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.action_history = []
        self.last_step_evidence = {}

    # ----- 公共接口 ----------------------------------------------------

    def step(self, frame: Image.Image) -> tuple[str, Optional[dict]]:
        """执行一步：planner 出 plan → 转成数据集格式的 semantic action。"""
        self.last_step_evidence = {"planner_attempts": []}
        plan = self._call_planner(frame)
        if plan is None:
            failure = self.last_step_evidence.get("failure") or {
                "kind": "planner_response_parse_failed",
                "message": "planner 未返回可用的 JSON plan",
            }
            self.last_step_evidence["failure"] = failure
            return f"[{failure['kind']}:{failure['message']}]", None
        self.last_step_evidence["plan"] = plan

        action = plan_to_semantic_action(plan, frame.size)
        if action is None:
            failure = {"kind": "planner_unsupported_action",
                       "message": str(plan.get("type") or "empty type")}
            self.last_step_evidence["failure"] = failure
            return f"[{failure['kind']}:{failure['message']}]", None
        self.last_step_evidence["semantic_action"] = action

        raw_response = format_agent_response(plan, action)
        self.last_step_evidence["formatted_response"] = raw_response
        self.action_history.append(short_action_description(plan))
        return raw_response, action

    # ----- planner -----------------------------------------------------

    def _planner_user_text(self) -> str:
        if self.action_history:
            hist = "\n".join(f"  Step {i+1}: {a}" for i, a in enumerate(self.action_history))
        else:
            hist = "  No previous action."
        return PLANNER_USER_TEMPLATE.format(
            instruction=self.instruction,
            hist_lines=hist,
        )

    def _call_planner(self, frame: Image.Image) -> Optional[dict]:
        """调用 planner 并取回 `computer` 工具调用的参数（即一个动作）。"""
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": self._planner_user_text()},
                {"type": "image_url", "image_url": {"url": _pil_to_data_url(frame)}},
            ]},
        ]
        for attempt in range(self.max_retry):
            try:
                # tool_choice 用 auto 而不是强制指定 computer：推理模型（DeepSeek 思考模式）
                # 对强制 tool_choice 直接 400；prompt 已要求调工具，没调到的情况由下面的
                # 校验 + 重试兜住（planner_no_tool_call）。
                resp = self.planner.chat.completions.create(
                    model=self.planner_model,
                    messages=messages,
                    tools=[COMPUTER_TOOL],
                    tool_choice="auto",
                    temperature=self.planner_temperature,
                    max_tokens=self.planner_max_tokens,
                )
            except Exception as error:
                message = f"{type(error).__name__}: {error}"[:500]
                self.last_step_evidence["planner_attempts"].append(
                    {"attempt": attempt + 1, "requested_model": self.planner_model,
                     "error": message})
                self.last_step_evidence["failure"] = {"kind": "planner_api_error",
                                                      "message": message}
                if attempt == self.max_retry - 1:
                    return None
                time.sleep(5)
                continue

            message = resp.choices[0].message
            response_model = getattr(resp, "model", None)
            plan, validation_error = _plan_from_tool_calls(message.tool_calls)
            if response_model is not None and response_model != self.planner_model:
                validation_error = (
                    f"响应模型 {response_model!r} 与请求模型 {self.planner_model!r} 不一致"
                )
            usage = getattr(resp, "usage", None)
            self.last_step_evidence["planner_attempts"].append({
                "attempt": attempt + 1,
                "tool_calls": [c.function.arguments for c in (message.tool_calls or [])],
                "content": message.content,
                "requested_model": self.planner_model,
                "response_model": response_model,
                "validation_error": validation_error,
                "usage": usage.model_dump(exclude_none=True) if usage is not None else None,
            })
            if validation_error is None:
                return plan
            self.last_step_evidence["failure"] = {
                "kind": _failure_kind(validation_error),
                "message": validation_error,
            }
            if attempt < self.max_retry - 1:
                time.sleep(2)
        return None
