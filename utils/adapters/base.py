"""Base class for world models that emit HTML.

Provides one interface for code-generating world models. Each adapter is a thin wrapper that,
in order:
  1. builds the model-specific prompt from the canonical ``semantic_action`` and history;
  2. calls the world model's endpoint;
  3. parses the HTML out of the model output;
  4. renders that HTML to PNG through Playwright;
  5. reuses a cached result only when the full model request is identical.

The two history settings describe what the harness feeds the model at each step, not a property
of the model itself. Under ``WM-NoHist`` the adapter receives only the current screen and the
current action; under ``WM-FullHist`` it additionally receives the configured window of recent
(state, action) pairs. A model that maintains cross-step state on its own should be evaluated
under ``WM-NoHist`` and keep that state inside its adapter instance.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from openai import OpenAI

from utils.failure import classify_failure


def load_image_array(s) -> np.ndarray:
    if isinstance(s, np.ndarray):
        return s
    if isinstance(s, (str, os.PathLike)):
        with Image.open(s) as im:
            return np.array(im.convert("RGB"))
    if isinstance(s, Image.Image):
        return np.array(s.convert("RGB"))
    raise TypeError(f"unsupported screenshot type {type(s)}")


def png_b64(arr: np.ndarray) -> str:
    pil = Image.fromarray(arr).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_html(text: str) -> str:
    """移除 ``<think>`` 和 Markdown fence，并提取完整 HTML 文档。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:html)?\s*(<!DOCTYPE.*?</html>)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"(<!DOCTYPE.*?</html>)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"(<html.*?</html>)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    raise ValueError("world model response does not contain a complete HTML document")


@dataclass
class WMPrediction:
    html: str
    pred_png_path: str          # 渲染结果路径
    pred_html_path: str         # 保存的 HTML 路径
    raw_response: str
    cached: bool
    error: Optional[str] = None
    failure_class: Optional[str] = None  # 'infrastructure' | 'model'


class BaseWMAdapter:
    name: str = "base"
    render_name: str | None = None
    served_model_name: str = "base"
    base_url: str = ""
    history_setting: str = "WM-NoHist"  # 也可以是 "WM-FullHist"
    # `name` 标识配置中的模型及输出；`render_name` 标识兼容的 HTML renderer，
    # 其取值为 code2world、gworld 或 mobileworld。

    def __init__(self, base_url: str, served_model_name: str,
                 persist_root: str, history_setting: str = "WM-NoHist",
                 max_tokens: int = 8192, temperature: float = 0.0,
                 retries: int = 2, timeout: float = 600.0,
                 hist_window: int = 3, output_name: str | None = None):
        self.base_url = base_url
        self.served_model_name = served_model_name
        self.history_setting = history_setting
        self.render_name = self.render_name or self.name
        if output_name:
            self.name = output_name
        history_dir = "fullhist" if history_setting == "WM-FullHist" else "nohist"
        self.persist_root = Path(persist_root) / self.name / history_dir
        self.persist_root.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.hist_window = hist_window
        self._client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)

    # --- 由子类实现 ---

    def build_messages(self, before_arr: np.ndarray, semantic_action: dict,
                       history: Optional[list[dict]] = None) -> list:
        raise NotImplementedError

    def parse_output(self, raw: str) -> str:
        return parse_html(raw)

    # --- 公共执行流程 ---

    def _step_dir(self, sample_id: str, step_id: int) -> Path:
        d = self.persist_root / sample_id / f"step_{step_id:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def predict(self, sample_id: str, step_id: int,
                before_arr: np.ndarray, semantic_action: dict,
                history: Optional[list[dict]] = None,
                use_cache: bool = True) -> WMPrediction:
        from .html_to_png import render_html_string_to_image
        d = self._step_dir(sample_id, step_id)
        png_p = d / "pred.png"
        html_p = d / "pred.html"
        raw_p = d / "raw_response.txt"
        meta_p = d / "meta.json"

        messages = self.build_messages(before_arr, semantic_action, history)
        request_payload = {
            "adapter": f"{type(self).__module__}.{type(self).__name__}",
            "name": self.name,
            "served_model_name": self.served_model_name,
            "base_url": self.base_url,
            "history_setting": self.history_setting,
            "history_window": self.hist_window,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
            "resolved_config": getattr(self, "resolved_config", None),
        }
        request_sha256 = hashlib.sha256(
            json.dumps(request_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if (
            use_cache
            and png_p.is_file()
            and html_p.is_file()
            and raw_p.is_file()
            and meta_p.is_file()
        ):
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                with Image.open(png_p) as image:
                    image.verify()
                if meta.get("request_sha256") == request_sha256:
                    return WMPrediction(
                        html=html_p.read_text(encoding="utf-8"),
                        pred_png_path=str(png_p),
                        pred_html_path=str(html_p),
                        raw_response=raw_p.read_text(encoding="utf-8") if raw_p.exists() else "",
                        cached=True,
                    )
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        for stale in (png_p, html_p, raw_p, meta_p):
            stale.unlink(missing_ok=True)

        last_err = None
        raw = ""
        response_model = None
        usage = None
        for attempt in range(self.retries + 1):
            try:
                r = self._client.chat.completions.create(
                    model=self.served_model_name, messages=messages,
                    max_tokens=self.max_tokens, temperature=self.temperature,
                )
                raw = r.choices[0].message.content or ""
                response_model = getattr(r, "model", None)
                usage = getattr(r, "usage", None)
                usage = usage.model_dump(exclude_none=True) if usage is not None else None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                _fc = classify_failure(str(e), "request")
                if _fc["class"] == "infrastructure" and attempt < self.retries:
                    time.sleep(2 ** attempt)
                else:
                    break  # model failure or last attempt — no retry
        if not raw:
            error = (
                "empty_completion" if last_err is None else str(last_err)[:300]
            )
            fc = classify_failure(error, "request")
            meta_p.write_text(json.dumps({
                "stage": "request",
                "error": error,
                "failure_class": fc["class"],
                "request_sha256": request_sha256,
                "response_model": response_model,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            fc = classify_failure(error, "request")
            return WMPrediction(html="", pred_png_path="", pred_html_path="",
                                raw_response="", cached=False, error=error,
                                failure_class=fc["class"])

        raw_p.write_text(raw, encoding="utf-8")
        try:
            html = self.parse_output(raw)
        except Exception as error:  # noqa: BLE001
            meta_p.write_text(json.dumps({
                "wm": self.name,
                "served_model_name": self.served_model_name,
                "request_sha256": request_sha256,
                "stage": "parse",
                "error": str(error)[:500],
                "failure_class": "model",
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            return WMPrediction(
                html="", pred_png_path="", pred_html_path="",
                raw_response=raw, cached=False, error=f"parse_fail: {error}",
                failure_class="model",
            )
        # 按模型专用协议将 HTML 渲染为 PNG。
        # ref_w/ref_h 是样本原始像素尺寸，三个 renderer 都用它统一输出尺寸。
        ref_h, ref_w = before_arr.shape[:2]   # NumPy shape = (H, W, 3)
        try:
            render_html_string_to_image(html, work_dir=str(d), idx=0,
                                         wm=self.render_name,
                                         ref_w=ref_w, ref_h=ref_h)
            old_png = d / "sample_0.png"
            old_html = d / "sample_0.html"
            if old_png.exists():
                os.replace(old_png, png_p)
            if old_html.exists():
                os.replace(old_html, html_p)
        except Exception as e:  # noqa: BLE001
            html_p.write_text(html, encoding="utf-8")
            _rfc = classify_failure(str(e), "render")
            meta_p.write_text(json.dumps({
                "wm": self.name,
                "served_model_name": self.served_model_name,
                "request_sha256": request_sha256,
                "stage": "render",
                "error": str(e)[:500],
                "failure_class": _rfc["class"],
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            _rfc = classify_failure(str(e), "render")
            return WMPrediction(html=html, pred_png_path="", pred_html_path=str(html_p),
                                raw_response=raw, cached=False, error=f"render_fail: {e}",
                                failure_class=_rfc["class"])

        meta = {
            "wm": self.name,
            "served_model_name": self.served_model_name,
            "response_model": response_model,
            # token 用量：API 世界模型小样本试跑时据此估全量开销。
            "usage": usage,
            "history_setting": self.history_setting,
            "n_history_pairs": len(history) if history else 0,
            "semantic_action": semantic_action,
            "request_sha256": request_sha256,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return WMPrediction(html=html, pred_png_path=str(png_p), pred_html_path=str(html_p),
                            raw_response=raw, cached=False)
