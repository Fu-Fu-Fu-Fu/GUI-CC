"""Adapter for closed HTML-generating world models.

Uses the paper's HTML-generation prompt and calls the model through an external
OpenAI-compatible API. This is also the generic entry point for plugging in your own model:
point it at any OpenAI-compatible endpoint with ``--endpoint`` and ``--served-model``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

from openai import OpenAI
from utils.prompts.paper_prompts import (
    HTML_WM_HISTORY_SYSTEM_SUFFIX,
    HTML_WM_HISTORY_USER_TEMPLATE,
    HTML_WM_SYSTEM_PROMPT,
    HTML_WM_USER_TEMPLATE,
)
from utils.prompts.prompt_loader import trim_history_template

# 默认为 None，直接建 OpenAI 客户端。API 与 GPU 分处两台机器时，外部驱动可以
# 在此换成自己的客户端；本仓库不依赖任何外部模块。
API_CLIENT = None

from .annotate_action import annotate
from .base import png_b64
from .code2world_adapter import (
    Code2WorldAdapter,
    build_semantic_desc_official,
)


class ClosedModelAdapter(Code2WorldAdapter):
    render_name = "code2world"

    def __init__(self, base_url: str | None, served_model_name: str,
                 persist_root: str, api_key: str | None = None,
                 name_override: str | None = None, **kwargs):
        if not api_key and API_CLIENT is None:
            raise RuntimeError("Set CLOSED_MODEL_API_KEY or OPENAI_API_KEY.")
        kwargs.setdefault("max_tokens", 16384)
        kwargs.setdefault("timeout", 120.0)
        kwargs.setdefault("temperature", 1.0)
        kwargs.setdefault("retries", 5)
        super().__init__(base_url=base_url, served_model_name=served_model_name,
                         persist_root=persist_root, **kwargs)
        if name_override:
            self.name = name_override
            history_dir = (
                "fullhist" if self.history_setting == "WM-FullHist" else "nohist")
            self.persist_root = Path(persist_root) / self.name / history_dir
            self.persist_root.mkdir(parents=True, exist_ok=True)
        client_kwargs = {"api_key": api_key, "timeout": kwargs.get("timeout", 120.0)}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = API_CLIENT or OpenAI(**client_kwargs)

    def build_messages(self, before_arr: np.ndarray, semantic_action: dict,
                       history: Optional[list[dict]] = None) -> list:
        ref_h, ref_w = before_arr.shape[:2]
        annotated = annotate(before_arr, semantic_action)
        b64 = png_b64(annotated)
        desc = build_semantic_desc_official(semantic_action)
        system_prompt = HTML_WM_SYSTEM_PROMPT.format(W=ref_w, H=ref_h)
        user_text = HTML_WM_USER_TEMPLATE.format(semantic_desc=desc)

        if self.history_setting != "WM-FullHist" or not history:
            return [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ]

        past = [
            item for item in history if item.get("before_arr") is not None
        ][-self.hist_window:]
        descriptions = [
            build_semantic_desc_official(item.get("semantic_action") or {})
            for item in past
        ]
        history_template = trim_history_template(
            HTML_WM_HISTORY_USER_TEMPLATE, len(past))
        formatted_user = history_template.format(
            desc_0=descriptions[0] if len(descriptions) > 0 else "",
            desc_1=descriptions[1] if len(descriptions) > 1 else "",
            desc_2=descriptions[2] if len(descriptions) > 2 else "",
            desc_current=desc,
        )
        image_map = {
            f"[image_step{i}]": png_b64(item["before_arr"])
            for i, item in enumerate(past)
        }
        image_map["[annotated_current_image]"] = b64
        content = []
        for part in re.split(r"(\[image_step[0-2]\]|\[annotated_current_image\])", formatted_user):
            if part in image_map:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_map[part]}"},
                })
            elif part:
                content.append({"type": "text", "text": part})

        return [
            {"role": "system", "content": [{
                "type": "text",
                "text": system_prompt + HTML_WM_HISTORY_SYSTEM_SUFFIX,
            }]},
            {"role": "user", "content": content},
        ]
