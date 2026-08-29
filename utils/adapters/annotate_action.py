"""Draw the action cue on the screenshot, following the Code2World input convention:

  - tap / long_press -> a translucent red circle at the touch point
  - scroll           -> a red arrow showing the finger movement direction
  - every other type -> no visual cue; the text description carries the action

A Code2World input consists of the annotated screenshot plus the natural-language
``semantic_desc``.
"""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw


def _bbox_center(bbox):
    if not bbox or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    except (TypeError, ValueError):
        return None
    return (round((x1 + x2) / 2), round((y1 + y2) / 2))


def coordinates_from_action(action: dict) -> tuple[int, int] | None:
    """从 semantic action 中提取点击坐标。"""
    if action is None:
        return None
    sc = action.get("source_coord")
    if sc and len(sc) >= 2:
        try:
            return (round(float(sc[0])), round(float(sc[1])))
        except (TypeError, ValueError):
            pass
    return _bbox_center((action.get("grounding") or {}).get("bbox"))


def _normalized_point_to_pixel(point, width: int, height: int) -> tuple[int, int] | None:
    """将 GUI-Odyssey 的 [0, 1000] 坐标转换为截图像素坐标。"""
    if not point or len(point) < 2:
        return None
    try:
        x = round(float(point[0]) / 1000 * width)
        y = round(float(point[1]) / 1000 * height)
    except (TypeError, ValueError):
        return None
    return (
        min(max(int(x), 0), max(width - 1, 0)),
        min(max(int(y), 0), max(height - 1, 0)),
    )


def scroll_coordinates_from_action(
    action: dict, width: int, height: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """返回滑动起止像素坐标，并优先使用数据中记录的端点。"""
    if action is None:
        return None
    start = _normalized_point_to_pixel(action.get("start_norm"), width, height)
    end = _normalized_point_to_pixel(action.get("end_norm"), width, height)
    if start is not None and end is not None:
        return start, end

    direction = (action.get("direction") or "").lower()
    delta = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }.get(direction)
    if delta is None:
        return None
    center = (width // 2, height // 2)
    length = min(400, max(1, round(min(width, height) * 0.4)))
    end = (
        min(max(center[0] + delta[0] * length, 0), max(width - 1, 0)),
        min(max(center[1] + delta[1] * length, 0), max(height - 1, 0)),
    )
    return (center, end) if end != center else None


def annotate(image: np.ndarray, action: dict) -> np.ndarray:
    """返回叠加动作视觉提示后的新 RGB 图像。"""
    pil = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    fill = (255, 0, 0, 100)
    outline = (255, 0, 0, 255)

    t = (action or {}).get("type")
    if t in ("tap", "click", "long_press"):
        c = coordinates_from_action(action)
        if c is not None:
            x, y = c
            r = 40
            draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=5)
    elif t == "scroll":
        h, w = image.shape[:2]
        points = scroll_coordinates_from_action(action, w, h)
        if points is None or points[0] == points[1]:
            return np.array(pil.convert("RGB"))
        (cx, cy), (ex, ey) = points
        draw.line([(cx, cy), (ex, ey)], fill=outline, width=15)
        ang = math.atan2(ey - cy, ex - cx)
        al = 50
        p1 = (ex - al * math.cos(ang - math.pi / 6), ey - al * math.sin(ang - math.pi / 6))
        p2 = (ex - al * math.cos(ang + math.pi / 6), ey - al * math.sin(ang + math.pi / 6))
        draw.polygon([(ex, ey), p1, p2], fill=outline)
    # type_text / navigate_* / wait / status / open_app 不添加视觉标记
    return np.array(Image.alpha_composite(pil, overlay).convert("RGB"))
