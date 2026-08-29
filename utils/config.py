"""Machine-local configuration: load utils/configs/paths.env and read values from it.

Process environment variables already set by the caller win; ``paths.env`` only fills in what
is missing. Keep ``paths.env`` on your machine and out of version control.
"""
from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_CONFIG = REPO_ROOT / "utils" / "configs" / "offline.json"
ONLINE_CONFIG = REPO_ROOT / "utils" / "configs" / "online.json"


def _load_env_file() -> None:
    env_file = Path(os.environ.get("GUI_CC_ENV_FILE", REPO_ROOT / "utils" / "configs" / "paths.env"))
    if not env_file.exists():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        if not key or key in os.environ:
            continue
        try:
            parts = shlex.split(value, comments=False, posix=True)
            parsed = parts[0] if parts else ""
        except ValueError:
            parsed = value.strip().strip('"').strip("'")
        os.environ[key] = parsed


_load_env_file()


def env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    configured = os.environ.get(name)
    value = configured if configured not in (None, "") else default
    if required and not value:
        raise RuntimeError(
            f"Missing required config {name}. Set it in utils/configs/paths.env or the environment."
        )
    return value


def required_path(name: str) -> Path:
    return Path(env(name, required=True)).expanduser()  # type: ignore[arg-type]


def load_project_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
