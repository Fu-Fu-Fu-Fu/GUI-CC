"""Adapter aligned with the public Code2World input format.

The action description reproduces the ``T3A.step`` output of DiffSynth-Studio/Code2World:

  input_text    ->  "Type in {text}."
  navigate_home ->  "Press home button to go to home page."
  navigate_back ->  "Press the back button to go to last page."
  open_app      ->  "Open the app: {app_name}."
  click/scroll/long_press -> short natural language naming the target, for example
                             "click the + icon", "scroll up",
                             "long press the AliExpress icon"

Visual annotation follows Code2World's ``wm_utils.py``: a red circle for tap/long_press and a
red arrow for scroll.

The model is trained in a Markov fashion only; +H is an extension that prepends the previous
(image, action_text) turns as additional user/assistant messages.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from utils.prompts.paper_prompts import (
    CODE2WORLD_HISTORY_USER_TEMPLATE,
    CODE2WORLD_SYSTEM_PROMPT,
    CODE2WORLD_USER_TEMPLATE,
    HISTORY_SYSTEM_SUFFIX,
)
from utils.prompts.prompt_loader import trim_history_template

from .annotate_action import annotate
from .base import BaseWMAdapter, png_b64


def build_semantic_desc_official(sa: dict) -> str:
    """按照规范构造 Code2World 动作描述，复现官方 ``T3A.step`` 输出。"""
    t = sa.get("type")
    instruction = (sa.get("low_level_instruction") or "").strip()
    target = (instruction or sa.get("target") or "").strip()
    target_sentence = target.rstrip(".")
    if instruction and t in {"tap", "long_press"}:
        return target_sentence + "."
    already_action = bool(re.match(
        r"^(tap|click|press|open|select|choose|dismiss|confirm|submit|start|stop|toggle|long[- ]?press)\b",
        target_sentence,
        flags=re.IGNORECASE,
    ))
    if t == "type_text":
        return f"Type in {sa.get('text', '')}."
    if t == "navigate_home":
        return "Press home button to go to home page."
    if t == "navigate_back":
        return "Press the back button to go to last page."
    if t == "open_app":
        return f"Open the app: {sa.get('app_name') or sa.get('target') or ''}."
    if t == "wait":
        return "Wait for the screen to stabilize."
    if t == "tap":
        if already_action:
            return target_sentence + "."
        return f"Click {target_sentence}." if target_sentence else "Click the indicated element."
    if t == "long_press":
        if already_action:
            return target_sentence + "."
        return f"Long press {target_sentence}." if target_sentence else "Long press the indicated element."
    if t == "scroll":
        d = sa.get("direction", "down")
        return f"scroll {d}."
    return f"perform {t}."


class Code2WorldAdapter(BaseWMAdapter):
    name = "code2world"
    # 渲染分派给 html_to_png._render_code2world_html_to_png，
    # 其基础实现移植自 DiffSynth-Studio Code2World wm_utils.render_aligned_png。

    def build_messages(self, before_arr: np.ndarray, semantic_action: dict,
                       history: Optional[list[dict]] = None) -> list:
        annotated = annotate(before_arr, semantic_action)
        b64 = png_b64(annotated)
        desc = build_semantic_desc_official(semantic_action)
        user_text = CODE2WORLD_USER_TEMPLATE.format(semantic_desc=desc)

        # Markov 路径保持原始行为，不附加历史。
        if self.history_setting != "WM-FullHist" or not history:
            return [
                {"role": "system", "content": [{"type": "text", "text": CODE2WORLD_SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ]

        past = history[-self.hist_window:]
        sys_prompt = CODE2WORLD_SYSTEM_PROMPT + HISTORY_SYSTEM_SUFFIX
        descriptions = [
            build_semantic_desc_official(item.get("semantic_action") or {})
            for item in past
        ]
        template = trim_history_template(CODE2WORLD_HISTORY_USER_TEMPLATE, len(past))
        text = template.format(
            desc_0=descriptions[0] if len(descriptions) > 0 else "",
            desc_1=descriptions[1] if len(descriptions) > 1 else "",
            desc_2=descriptions[2] if len(descriptions) > 2 else "",
            desc_current=desc,
        )
        image_map = {
            f"[image_step{index}]": png_b64(item["before_arr"])
            for index, item in enumerate(past)
        }
        image_map["[annotated_current_image]"] = b64
        pattern = "(" + "|".join(re.escape(key) for key in image_map) + ")"
        content = []
        for part in re.split(pattern, text):
            if part in image_map:
                content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_map[part]}"
                }})
            elif part:
                content.append({"type": "text", "text": part})

        return [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": content},
        ]
