"""World-model registry shared by the offline and online runners."""
from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Any

from utils.config import env

ADAPTER_CLASSES = {
    "closed_html": ("utils.adapters.closedmodel_adapter", "ClosedModelAdapter"),
    "code2world": ("utils.adapters.code2world_adapter", "Code2WorldAdapter"),
    "qwen_image_edit": ("utils.adapters.qwen_image_edit_adapter", "QwenImageEditAdapter"),
}

DEFAULT_ENDPOINTS = {
    "CODE2WORLD_URL": "http://localhost:4244/v1",
}

CHECKPOINT_ENV = {
    "code2world": ("CODE2WORLD_CKPT",),
    "qwen_image_edit": ("QWEN_IMAGE_EDIT_2511_DIR", "DIFFSYNTH_DIR"),
}


def get_model_spec(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    for spec in config.get("models", []):
        if spec.get("id") == model_id:
            return dict(spec)
    available = ", ".join(str(item.get("id")) for item in config.get("models", []))
    raise ValueError(f"Unknown model {model_id!r}; available: {available}")


def _portable(value):
    """把本机绝对路径换成可移植的标识（HF 仓库名或文件名）。

    run 记录要随包开源，不能带任何本机或历史服务器的路径；而且它进 run_sha256，
    带路径会让同一份配置在不同机器上得到不同的 run 标识。
    """
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        parts = [part for part in value.strip("/").split("/") if part]
        return "/".join(parts[-2:]) if parts else value
    return value


def resolved_model_config(
    spec: dict[str, Any],
    history_setting: str,
    *,
    endpoint_override: str | None = None,
    device: str = "cuda",
    seed_override: int | None = None,
    num_steps_override: int | None = None,
    history_window: int = 3,
    max_tokens_override: int | None = None,
) -> dict[str, Any]:
    endpoint_env = spec.get("endpoint_env")
    endpoint = endpoint_override
    if not endpoint and endpoint_env:
        endpoint = env(str(endpoint_env), DEFAULT_ENDPOINTS.get(str(endpoint_env)))
    if spec.get("adapter") == "closed_html":
        endpoint = endpoint_override or env("CLOSED_MODEL_BASE_URL") or env("OPENAI_BASE_URL")
    
    model_id = str(spec["id"])
    # 只记哪些 checkpoint 环境变量被设过，不记本机绝对路径：路径没有消费方，
    # 且会进 run_sha256，让同一配置在不同机器上得到不同的 run 标识。
    checkpoint_env_set = sorted(
        name for name in CHECKPOINT_ENV.get(model_id, ()) if os.environ.get(name)
    )
    identity_fields = {
        "id", "adapter", "served_model", "model", "endpoint_env",
        "settings", "seed", "num_steps", "revision",
    }
    generation = {
        key: value for key, value in spec.items() if key not in identity_fields
    }
    if spec.get("adapter") in {"code2world", "closed_html"}:
        generation["max_tokens"] = (
            max_tokens_override
            if max_tokens_override is not None
            else spec.get("max_tokens",
                          16384 if spec.get("adapter") == "closed_html" else 8192)
        )
        # 记录实际采样用的值；spec 里显式给了就用它（mobileworld 是 0.7）。
        generation["temperature"] = spec.get(
            "temperature", 1.0 if spec.get("adapter") == "closed_html" else 0.0)
    return {
        "model_id": model_id,
        "adapter": spec["adapter"],
        "served_model": spec.get("served_model") or spec.get("model"),
        "history_setting": history_setting,
        "history_window": history_window,
        "endpoint": endpoint,
        "device": device,
        "seed": seed_override if seed_override is not None else spec.get("seed"),
        "num_steps": num_steps_override if num_steps_override is not None else spec.get("num_steps"),
        "generation": _portable(generation),
        "checkpoint_env_set": checkpoint_env_set,
        "model_revision": env(f"{model_id.upper()}_REVISION") or spec.get("revision"),
    }


def create_adapter(
    spec: dict[str, Any],
    persist_root: str | Path,
    history_setting: str,
    *,
    endpoint_override: str | None = None,
    device: str = "cuda",
    seed_override: int | None = None,
    num_steps_override: int | None = None,
    max_tokens: int | None = None,
    history_window: int = 3,
):
    """实例化一个已配置的适配器，同时保留模型自身默认参数。"""
    resolved = resolved_model_config(
        spec,
        history_setting,
        endpoint_override=endpoint_override,
        device=device,
        seed_override=seed_override,
        num_steps_override=num_steps_override,
        history_window=history_window,
        max_tokens_override=max_tokens,
    )
    adapter_name = str(spec["adapter"])
    module_name, class_name = ADAPTER_CLASSES[adapter_name]
    cls = getattr(import_module(module_name), class_name)
    common = {
        "persist_root": str(persist_root),
        "history_setting": history_setting,
    }

    if adapter_name in {"code2world", "closed_html"}:
        kwargs = {
            **common,
            "base_url": resolved["endpoint"],
            "served_model_name": spec["served_model"],
            "hist_window": history_window,
            "output_name": spec["id"],
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        elif "max_tokens" in spec:
            kwargs["max_tokens"] = int(spec["max_tokens"])
        if "temperature" in spec:
            kwargs["temperature"] = spec["temperature"]
        if adapter_name == "closed_html":
            kwargs.update({
                "api_key": env("CLOSED_MODEL_API_KEY") or env("OPENAI_API_KEY"),
                "name_override": spec["id"],
            })
        adapter = cls(**kwargs)
    else:
        kwargs = {**common, "device": device}
        if resolved["seed"] is not None:
            kwargs["seed"] = int(resolved["seed"])
        if resolved["num_steps"] is not None:
            kwargs["num_steps"] = int(resolved["num_steps"])
        if adapter_name == "qwen_image_edit":
            kwargs["max_pixels"] = int(spec.get("max_pixels", 1024 * 1024))
            kwargs["low_vram"] = bool(spec.get("low_vram", False))
        adapter = cls(**kwargs)
        if spec.get("model"):
            adapter.served_model_name = str(spec["model"])

    model_id = str(spec["id"])
    history_dir = "fullhist" if history_setting == "WM-FullHist" else "markov"
    adapter.name = model_id
    adapter.persist_root = Path(persist_root) / model_id / history_dir
    adapter.persist_root.mkdir(parents=True, exist_ok=True)
    adapter.resolved_config = resolved
    return adapter
