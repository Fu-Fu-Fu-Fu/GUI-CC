"""Base class for world models that generate images directly.

A diffusion world model produces the next-state image without an HTML intermediate. Each adapter:
  1. builds a text prompt from the canonical ``semantic_action`` and optional history;
  2. preprocesses the input screenshot to the resolution the model expects;
  3. runs the model's inference pipeline to generate the next-state prediction;
  4. saves the PNG under ``persist_root/<wm>/<setting>/<sample_id>/step_NNN/``.

Unlike the code-generating adapters there is no rendering step: the generated image is the
final output.
"""
from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.failure import classify_failure

import numpy as np
from PIL import Image


@dataclass
class DiffusionPrediction:
    pred_png_path: str
    cached: bool
    error: Optional[str] = None
    failure_class: Optional[str] = None


class BaseDiffusionAdapter:
    name: str = "diffusion_base"
    history_setting: str = "WM-NoHist"

    def __init__(self, persist_root: str, history_setting: str = "WM-NoHist",
                 device: str = "cuda", seed: int = 42, num_steps: int = 40,
                 flat_output: bool = False):
        self.history_setting = history_setting
        self.flat_output = flat_output
        if flat_output:
            self.persist_root = Path(persist_root) / self.name
        else:
            history_dir = "fullhist" if history_setting == "WM-FullHist" else "nohist"
            self.persist_root = Path(persist_root) / self.name / history_dir
        self.persist_root.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.seed = seed
        self.num_steps = num_steps
        self._pipe = None

    def load_model(self):
        raise NotImplementedError

    def build_prompt(self, semantic_action: dict,
                     history: Optional[list[dict]] = None) -> str:
        raise NotImplementedError

    def generate(self, input_image: Image.Image, prompt: str,
                 height: int, width: int) -> Image.Image:
        raise NotImplementedError

    def target_hw(self, src_w: int, src_h: int,
                  target_area: int = 1024 * 1024,
                  divisor: int = 16) -> tuple[int, int]:
        import math
        ratio = src_w / src_h
        w = math.sqrt(target_area * ratio)
        h = w / ratio
        w = max(divisor, round(w / divisor) * divisor)
        h = max(divisor, round(h / divisor) * divisor)
        return int(h), int(w)

    def _step_dir(self, sample_id: str, step_id: int) -> Path:
        d = self.persist_root / sample_id / f"step_{step_id:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def request_sha256(
        self,
        before_arr: np.ndarray,
        semantic_action: dict,
        history: Optional[list[dict]],
        prompt: str,
    ) -> str:
        history_rows = []
        for row in history or []:
            image = row.get("before_arr")
            history_rows.append({
                "image_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
                if image is not None else None,
                "semantic_action": row.get("semantic_action"),
            })
        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
            if key not in {"persist_root", "_pipe"}
            and isinstance(value, (str, int, float, bool, type(None), Path))
        }
        payload = {
            "adapter": f"{type(self).__module__}.{type(self).__name__}",
            "config": config,
            "before_sha256": hashlib.sha256(np.asarray(before_arr).tobytes()).hexdigest(),
            "semantic_action": semantic_action,
            "history": history_rows,
            "prompt": prompt,
            "resolved_config": getattr(self, "resolved_config", None),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def cache_matches(png_path: Path, meta_path: Path, request_sha256: str) -> bool:
        if not png_path.is_file() or not meta_path.is_file():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            with Image.open(png_path) as image:
                image.verify()
            return meta.get("request_sha256") == request_sha256
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def predict(self, sample_id: str, step_id: int,
                before_arr: np.ndarray, semantic_action: dict,
                history: Optional[list[dict]] = None,
                use_cache: bool = True) -> DiffusionPrediction:
        d = self._step_dir(sample_id, step_id)
        png_p = d / "pred.png"
        meta_p = d / "meta.json"

        prompt = self.build_prompt(semantic_action, history)
        request_sha256 = self.request_sha256(before_arr, semantic_action, history, prompt)
        if use_cache and self.cache_matches(png_p, meta_p, request_sha256):
            return DiffusionPrediction(pred_png_path=str(png_p), cached=True)
        png_p.unlink(missing_ok=True)
        meta_p.unlink(missing_ok=True)

        if self._pipe is None:
            self.load_model()

        input_image = Image.fromarray(before_arr).convert("RGB")
        src_w, src_h = input_image.size
        h, w = self.target_hw(src_w, src_h)

        try:
            out_image = self.generate(input_image, prompt, height=h, width=w)
            out_resized = out_image.resize((src_w, src_h), Image.LANCZOS)
            out_resized.save(png_p)
        except Exception as e:
            fc = classify_failure(str(e), "generation")
            return DiffusionPrediction(pred_png_path="", cached=False,
                                       error=f"generation_fail: {e}",
                                       failure_class=fc["class"])

        meta = {
            "wm": self.name,
            "history_setting": self.history_setting,
            "prompt": prompt,
            "gen_h": h, "gen_w": w,
            "src_h": src_h, "src_w": src_w,
            "seed": self.seed,
            "num_steps": self.num_steps,
            "semantic_action": semantic_action,
            "request_sha256": request_sha256,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return DiffusionPrediction(pred_png_path=str(png_p), cached=False)
