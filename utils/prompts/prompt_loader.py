from __future__ import annotations

from pathlib import Path
import re

PROMPT_ROOT = Path(__file__).resolve().parent


def load_prompt(relative_path: str) -> str:
    """加载相对于本目录的提示词模板。"""
    path = PROMPT_ROOT / relative_path
    return path.read_text(encoding="utf-8").strip("\n")


def trim_history_template(template: str, count: int) -> str:
    """从论文的三步模板中移除未使用的步骤和图片占位符。"""
    if not 0 <= count <= 3:
        raise ValueError(f"history count must be in [0, 3], got {count}")
    output = template
    for index in range(count, 3):
        output = re.sub(rf"^step {index}:.*\n", "", output, flags=re.MULTILINE)
        output = re.sub(
            rf"^\[image_step{index}\]\n\^ State BEFORE step {index}\..*\n",
            "",
            output,
            flags=re.MULTILINE,
        )
    return output
