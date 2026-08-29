"""Conversion between the planner's JSON plan and the dataset semantic action.

The planner emits absolute pixel coordinates directly, following the OpenAI computer-use
convention, and its action vocabulary already matches ``semantic_action.type`` in
``data/offline_data``. Only derived fields, such as the scroll endpoints, are filled in here.
"""
from __future__ import annotations

import json
from typing import Optional

# scroll 的滑动幅度：以落点为中心，沿指定方向各走屏幕的 30%。
SCROLL_SPAN = 0.3


def plan_to_semantic_action(plan: dict, frame_size: tuple[int, int]) -> Optional[dict]:
    """把 planner 的 plan 转成数据集格式的 semantic action。"""
    action = str(plan.get("type") or "").strip().lower()
    width, height = frame_size
    target = str(plan.get("target") or "").strip()
    common = {"target": target, "low_level_instruction": short_action_description(plan)}

    if action in {"tap", "long_press"}:
        return {"type": action, "source_coord": _clamp_point(plan, frame_size), **common}

    if action == "scroll":
        x, y = _clamp_point(plan, frame_size)
        direction = str(plan.get("direction") or "down").lower()
        dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}.get(
            direction, (0, 1))
        span_x, span_y = int(width * SCROLL_SPAN), int(height * SCROLL_SPAN)
        start = [x - dx * span_x // 2, y - dy * span_y // 2]
        end = [x + dx * span_x // 2, y + dy * span_y // 2]
        return {
            "type": "scroll", "source_coord": [x, y], "direction": direction,
            "start_norm": _to_norm(start, frame_size),
            "end_norm": _to_norm(end, frame_size),
            **common,
        }

    if action == "type_text":
        return {"type": "type_text", "text": plan.get("text", ""), **common}
    if action == "open_app":
        return {"type": "open_app", "app_name": plan.get("app_name", ""), **common}
    if action in {"navigate_home", "navigate_back", "wait"}:
        return {"type": action, **common}
    if action == "terminate":
        return {"type": "terminate", "value": plan.get("status", "success"), **common}
    if action == "answer":
        return {"type": "answer", "text": plan.get("answer_text", ""), **common}
    return None


def _clamp_point(plan: dict, frame_size: tuple[int, int]) -> list[int]:
    width, height = frame_size
    x = int(plan.get("x", width // 2))
    y = int(plan.get("y", height // 2))
    return [max(0, min(x, width - 1)), max(0, min(y, height - 1))]


def _to_norm(point: list[int], frame_size: tuple[int, int]) -> list[float]:
    """转成数据集使用的 0..1000 归一化坐标。"""
    width, height = frame_size
    return [
        max(0.0, min(point[0] / max(1, width) * 1000, 1000.0)),
        max(0.0, min(point[1] / max(1, height) * 1000, 1000.0)),
    ]


def short_action_description(plan: dict) -> str:
    """一句话动作摘要，进入动作历史与 semantic action 的 low_level_instruction。"""
    action = str(plan.get("type") or "?").strip().lower()
    target = str(plan.get("target") or "").strip()
    if action == "tap":
        return f"tap on {target}" if target else "tap"
    if action == "long_press":
        return f"long press {target}" if target else "long press"
    if action == "scroll":
        return f"scroll {plan.get('direction', '')} on {target or 'screen'}".strip()
    if action == "type_text":
        return f"type text: {plan.get('text', '')}"
    if action == "navigate_home":
        return "press home"
    if action == "navigate_back":
        return "press back"
    if action == "open_app":
        return f"open app: {plan.get('app_name', '')}"
    if action == "wait":
        return "wait"
    if action == "terminate":
        return f"terminate ({plan.get('status', '')})"
    if action == "answer":
        return f"answer: {plan.get('answer_text', '')}"
    return action


def format_agent_response(plan: dict, semantic_action: dict) -> str:
    """写进轨迹的可读记录：一行动作 + 完整 plan。"""
    return (
        f"Thinking: {plan.get('thinking', '')}\n"
        f"Action: {short_action_description(plan)}\n"
        f"Semantic action: {json.dumps(semantic_action, ensure_ascii=False)}\n"
    )


def is_terminal_action(semantic_action: dict) -> bool:
    return str((semantic_action or {}).get("type", "")).lower() in {"terminate", "answer"}
