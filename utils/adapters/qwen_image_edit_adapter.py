"""Zero-shot Qwen-Image-Edit-2511 adapter, without fine-tuning.

Uses the same ``QwenImagePipeline`` from DiffSynth-Studio but loads only the original base
weights, with no checkpoint override. This baseline tests whether the general image-editing
ability of Qwen-Image-Edit-2511 can carry out interface state transitions without any
GUI-specific training.

The prompt is a natural-language editing instruction describing the UI state change the action
is expected to produce.

Key inference parameters follow the official ``code2world_eval/inference.py``:
  - edit_image=[pil_img] (must be a list)
  - edit_image_auto_resize=True
  - zero_cond_t=True
  - max_pixels=1048576 (caps total pixels and aligns to a multiple of 16)
  - num_inference_steps=40, seed=42
"""
from __future__ import annotations

import os
import sys
import types
from typing import Optional

import numpy as np
from PIL import Image

from utils.config import required_path
from utils.prompts.paper_prompts import QWEN_IMAGE_EDIT_PROMPT_TEMPLATE

from .diffusion_base import BaseDiffusionAdapter, DiffusionPrediction

DEFAULT_MAX_PIXELS = 1048576


def _build_action_description(sa: dict) -> str:
    """构造填入论文 Figure 13 模板的动作短语。"""
    t = sa.get("type")
    target = (sa.get("target") or "").strip()
    instruction = (sa.get("low_level_instruction") or "").strip().rstrip(".")
    if instruction:
        return instruction
    if t == "tap":
        return f'tapping on "{target}"' if target else "tapping on the indicated element"
    if t == "long_press":
        return f'long pressing "{target}"' if target else "long pressing the indicated element"
    if t == "scroll":
        d = sa.get("direction", "down")
        return f"scrolling {d}"
    if t == "type_text":
        text = sa.get("text", "")
        return f'typing "{text}" into the {target} field' if target else f'typing "{text}" into the focused text field'
    if t == "navigate_home":
        return "pressing the home button"
    if t == "navigate_back":
        return "pressing the back button"
    if t == "open_app":
        app = sa.get("app_name") or sa.get("target") or "the app"
        return f"opening {app}"
    return f"the user performs a {t} action"


def _build_edit_prompt(sa: dict) -> str:
    """使用论文 Figure 13 模板构造 Qwen-Image-Edit prompt。"""
    return QWEN_IMAGE_EDIT_PROMPT_TEMPLATE.format(
        action_description=_build_action_description(sa)
    )


def _align_size(w: int, h: int, max_pixels: int = DEFAULT_MAX_PIXELS) -> tuple[int, int]:
    """将宽高按 16 对齐，并限制输出总像素。"""
    h_aligned = (h // 16) * 16
    w_aligned = (w // 16) * 16
    if h_aligned * w_aligned > max_pixels:
        scale = (max_pixels / (h_aligned * w_aligned)) ** 0.5
        h_aligned = int(h * scale) // 16 * 16
        w_aligned = int(w * scale) // 16 * 16
    return w_aligned, h_aligned


def _patch_siglip_import():
    """兼容 transformers>=5.x 中缺失的 SiglipVisionTransformer import。"""
    if 'transformers.models.siglip.modeling_siglip' not in sys.modules:
        _mock = types.ModuleType('transformers.models.siglip.modeling_siglip')
        _mock.SiglipVisionTransformer = type('SiglipVisionTransformer', (object,), {'__init__': lambda s, *a, **k: None})
        _mock.SiglipVisionConfig = type('SiglipVisionConfig', (object,), {'__init__': lambda s, *a, **k: None})
        sys.modules['transformers.models.siglip.modeling_siglip'] = _mock


class QwenImageEditAdapter(BaseDiffusionAdapter):
    name = "qwen_image_edit"

    def __init__(self, persist_root: str, history_setting: str = "WM-NoHist",
                 device: str = "cuda", seed: int = 42, num_steps: int = 40,
                 low_vram: bool = False, max_pixels: int = DEFAULT_MAX_PIXELS,
                 flat_output: bool = False):
        super().__init__(persist_root, history_setting, device, seed, num_steps, flat_output)
        self.low_vram = low_vram
        self.max_pixels = max_pixels

    def load_model(self):
        _patch_siglip_import()
        diffsynth_dir = required_path("DIFFSYNTH_DIR")
        model_dir = str(required_path("QWEN_IMAGE_EDIT_2511_DIR"))
        if str(diffsynth_dir) not in sys.path:
            sys.path.insert(0, str(diffsynth_dir))
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/"

        import torch
        from diffsynth.pipelines.qwen_image import QwenImagePipeline
        from diffsynth.core import ModelConfig

        extra = {}
        if self.low_vram:
            extra = dict(
                offload_dtype=torch.bfloat16,
                offload_device="cpu",
                onload_dtype=torch.bfloat16,
                onload_device="cpu",
                preparing_dtype=torch.bfloat16,
                preparing_device=self.device,
                computation_dtype=torch.bfloat16,
                computation_device=self.device,
            )

        self._pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=self.device,
            model_configs=[
                ModelConfig(model_id=model_dir,
                            origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                            **extra),
                ModelConfig(model_id=model_dir,
                            origin_file_pattern="text_encoder/model*.safetensors",
                            **extra),
                ModelConfig(model_id=model_dir,
                            origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                            **extra),
            ],
            tokenizer_config=None,
            processor_config=ModelConfig(model_id=model_dir,
                                         origin_file_pattern="processor/"),
        )

    def target_hw(self, src_w: int, src_h: int, **kwargs) -> tuple[int, int]:
        w_aligned, h_aligned = _align_size(src_w, src_h, self.max_pixels)
        return h_aligned, w_aligned

    def build_prompt(self, semantic_action: dict,
                     history: Optional[list[dict]] = None) -> str:
        return _build_edit_prompt(semantic_action)

    def generate(self, input_image: Image.Image, prompt: str,
                 height: int, width: int) -> Image.Image:
        return self._pipe(
            prompt=prompt,
            edit_image=[input_image],
            seed=self.seed,
            num_inference_steps=self.num_steps,
            height=height, width=width,
            edit_image_auto_resize=True,
            zero_cond_t=True,
        )
