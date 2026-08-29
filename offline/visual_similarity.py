"""SigLIP and DINOv2 cosine similarity for offline next-state evaluation."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

_HF_ENDPOINT = os.environ.get("VISUAL_SIM_HF_ENDPOINT")
if _HF_ENDPOINT:
    os.environ.setdefault("HF_ENDPOINT", _HF_ENDPOINT)
    os.environ.setdefault("HF_HUB_ENDPOINT", _HF_ENDPOINT)

ImageInput = Union[str, Path, Image.Image, np.ndarray]


def _load_pil(source: ImageInput) -> Image.Image:
    if isinstance(source, np.ndarray):
        return Image.fromarray(source).convert("RGB")
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (str, Path)):
        with Image.open(source) as image:
            return image.convert("RGB")
    raise TypeError(type(source))


_SIGLIP_LOCK = threading.Lock()
_SIGLIP = {
    "tried": False,
    "ok": False,
    "model": None,
    "processor": None,
    "device": None,
    "model_name": None,
    "error": None,
}


def _try_load_siglip() -> bool:
    try:
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        model_name = os.environ.get("VISUAL_SIM_SIGLIP_MODEL", "google/siglip-so400m-patch14-384")
        revision = os.environ.get("VISUAL_SIM_SIGLIP_REVISION")
        device = os.environ.get("VISUAL_SIM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        model = SiglipVisionModel.from_pretrained(model_name, revision=revision).to(device).eval()
        processor = AutoImageProcessor.from_pretrained(model_name, revision=revision)
    except Exception as error:  # noqa: BLE001
        _SIGLIP.update({"tried": True, "error": str(error)[:200]})
        return False
    _SIGLIP.update(
        {
            "tried": True,
            "ok": True,
            "model": model,
            "processor": processor,
            "device": device,
            "model_name": model_name,
        }
    )
    return True


def _ensure_siglip() -> bool:
    if _SIGLIP["tried"]:
        return bool(_SIGLIP["ok"])
    with _SIGLIP_LOCK:
        return bool(_SIGLIP["ok"]) if _SIGLIP["tried"] else _try_load_siglip()


def siglip_cosine(first: ImageInput, second: ImageInput) -> Optional[float]:
    if not _ensure_siglip():
        return None
    import torch

    with torch.no_grad():
        inputs = _SIGLIP["processor"](
            images=[_load_pil(first), _load_pil(second)], return_tensors="pt"
        ).to(_SIGLIP["device"])
        output = _SIGLIP["model"](**inputs)
        features = getattr(output, "pooler_output", None)
        if features is None:
            features = output.last_hidden_state[:, 0]
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cosine = (features[0] @ features[1]).item()
    return float(max(0.0, min(1.0, cosine)))


_DINO_LOCK = threading.Lock()
_DINO = {
    "tried": False,
    "ok": False,
    "model": None,
    "processor": None,
    "device": None,
    "model_name": None,
    "error": None,
}


def _try_load_dino() -> bool:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        model_name = os.environ.get("VISUAL_SIM_DINO_MODEL", "facebook/dinov2-giant")
        revision = os.environ.get("VISUAL_SIM_DINO_REVISION")
        local_model = os.environ.get("VISUAL_SIM_DINO_LOCAL_MODEL")
        if local_model:
            if not Path(local_model).is_dir():
                raise RuntimeError(
                    f"VISUAL_SIM_DINO_LOCAL_MODEL 指向的目录不存在：{local_model}"
                )
            if not model_name.startswith("/"):
                model_name = local_model
        device = os.environ.get("VISUAL_SIM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
        processor = AutoImageProcessor.from_pretrained(model_name, revision=revision)
    except Exception as error:  # noqa: BLE001
        _DINO.update({"tried": True, "error": str(error)[:200]})
        return False
    _DINO.update(
        {
            "tried": True,
            "ok": True,
            "model": model,
            "processor": processor,
            "device": device,
            "model_name": model_name,
        }
    )
    return True


def _ensure_dino() -> bool:
    if _DINO["tried"]:
        return bool(_DINO["ok"])
    with _DINO_LOCK:
        return bool(_DINO["ok"]) if _DINO["tried"] else _try_load_dino()


def dino_cosine(first: ImageInput, second: ImageInput) -> Optional[float]:
    if not _ensure_dino():
        return None
    import torch

    with torch.no_grad():
        inputs = _DINO["processor"](
            images=[_load_pil(first), _load_pil(second)], return_tensors="pt"
        ).to(_DINO["device"])
        output = _DINO["model"](**inputs)
        features = output.last_hidden_state[:, 0]
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cosine = (features[0] @ features[1]).item()
    return float(max(0.0, min(1.0, cosine)))


def backend_status() -> dict:
    return {
        "siglip": {
            "ok": _SIGLIP["ok"],
            "model": _SIGLIP["model_name"],
            "device": _SIGLIP["device"],
            "error": _SIGLIP["error"],
        },
        "dino": {
            "ok": _DINO["ok"],
            "model": _DINO["model_name"],
            "device": _DINO["device"],
            "error": _DINO["error"],
        },
    }
