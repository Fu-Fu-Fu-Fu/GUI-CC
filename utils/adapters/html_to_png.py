"""Render world-model HTML output to PNG.

The Code2World rendering path is a strict port of the renderer released with the
model, including details that look improvable. Changing it would make the output
differ from what the baseline authors produced themselves, so the logic is frozen
and only follows upstream changes.

  render_for_code2world  <- DiffSynth-Studio/Code2World/android_world/agents/wm_utils.py
                            (render_aligned_png): fixed 1080x2400 viewport, DPR=1,
                            omit_background=True, two-stage font loading fallback

Typical use:
  png_arr = render_html_string_to_image(html_str, work_dir, idx, ref_w=..., ref_h=...)
"""
from __future__ import annotations

import os
import time
import re
import numpy as np

from utils.failure import classify_failure
from pathlib import Path
from PIL import Image

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
except ModuleNotFoundError:
    sync_playwright = None

    class PlaywrightTimeoutError(Exception):
        pass


def _playwright_context():
    if sync_playwright is None:
        raise RuntimeError(
            "Playwright is required for HTML rendering; install requirements.txt "
            "and run `playwright install chromium`."
        )
    return sync_playwright()


# =====================================================================
# Code2World 官方 renderer 的基础实现
# 来源：DiffSynth-Studio/Code2World/android_world/agents/wm_utils.py:151
#       函数 `render_aligned_png(html_path, output_path)`
# =====================================================================

# 浏览器偶发故障（截图协议错误等）重开浏览器即可恢复；
# 渲染约占单步耗时 3.5%，重试成本可忽略。
RENDER_ATTEMPTS = 3

C2W_TARGET_W = 1080
C2W_TARGET_H = 2400


def _detect_render_target_size(html_path: str) -> tuple[int, int]:
    """从 HTML 的 CSS 或 inline style 中检测 ``#render-target`` 尺寸。

    检测成功时返回 ``(width, height)``，否则返回默认目标尺寸。
    """
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        # 尝试解析 CSS block：#render-target { width: ...px; height: ...px }
        m = re.search(
            r'#render-target\s*\{[^}]*width:\s*(\d+)px[^}]*height:\s*(\d+)px',
            html, re.DOTALL)
        if m:
            return int(m.group(1)), int(m.group(2))
        m2 = re.search(
            r'#render-target\s*\{[^}]*height:\s*(\d+)px[^}]*width:\s*(\d+)px',
            html, re.DOTALL)
        if m2:
            return int(m2.group(2)), int(m2.group(1))
        # 尝试解析 inline style：id="render-target" style="...width: Npx...height: Npx..."
        m3 = re.search(
            r'id="render-target"[^>]*style="[^"]*width:\s*(\d+)px[^"]*height:\s*(\d+)px',
            html, re.DOTALL)
        if m3:
            return int(m3.group(1)), int(m3.group(2))
        m4 = re.search(
            r'id="render-target"[^>]*style="[^"]*height:\s*(\d+)px[^"]*width:\s*(\d+)px',
            html, re.DOTALL)
        if m4:
            return int(m4.group(2)), int(m4.group(1))
    except Exception:
        pass
    return C2W_TARGET_W, C2W_TARGET_H


def _render_code2world_html_to_png(html_path: str, output_path: str,
                                   ref_w: int = C2W_TARGET_W,
                                   ref_h: int = C2W_TARGET_H) -> bool:
    """Code2World 自适应 renderer。

    1. 先按 HTML 声明的 ``#render-target`` 尺寸渲染；
    2. 检测内容是否溢出，即 scrollWidth/scrollHeight 是否大于容器；
    3. 若溢出，则移除 ``overflow:hidden`` 并按实际内容尺寸重新渲染；
    4. 最终输出统一缩放为 ``ref_w × ref_h``。
    """
    abs_html_path = os.path.abspath(html_path)
    html_w, html_h = _detect_render_target_size(html_path)

    with _playwright_context() as p:
        browser = p.chromium.launch(headless=True)
        success = False
        last_error = ""

        # 第一阶段：渲染并检测 overflow。
        context = browser.new_context(
            viewport={'width': html_w, 'height': html_h},
            device_scale_factor=1.0,
        )
        page = context.new_page()
        try:
            page.goto(f"file://{abs_html_path}", wait_until="load", timeout=30000)
            page.wait_for_load_state("networkidle", timeout=2000)

            # 取得实际内容尺寸，用于检查 overflow。
            content_size = page.evaluate('''() => {
                const el = document.getElementById('render-target');
                if (!el) return null;
                return {
                    scrollW: el.scrollWidth,
                    scrollH: el.scrollHeight,
                    clientW: el.clientWidth,
                    clientH: el.clientHeight
                };
            }''')

            overflow = False
            if content_size:
                overflow = (content_size['scrollW'] > content_size['clientW'] + 2 or
                            content_size['scrollH'] > content_size['clientH'] + 2)

            if not overflow:
                page.screenshot(path=output_path, omit_background=True)
                success = True
            else:
                # 令 overflow 可见，并按实际内容尺寸重新渲染。
                actual_w = content_size['scrollW']
                actual_h = content_size['scrollH']
                page.evaluate('''() => {
                    const el = document.getElementById('render-target');
                    if (el) {
                        el.style.overflow = 'visible';
                        el.style.width = el.scrollWidth + 'px';
                        el.style.height = el.scrollHeight + 'px';
                    }
                }''')
                page.set_viewport_size({'width': actual_w, 'height': actual_h})
                page.wait_for_timeout(100)
                page.screenshot(path=output_path, omit_background=True)
                success = True
        except PlaywrightTimeoutError as e:
            last_error = f"timeout: {e}"
        except Exception as e:
            last_error = str(e)
        finally:
            page.close()
            context.close()

        # 第二阶段：禁用外部字体并快速加载，作为回退路径。
        if not success:
            context_fb = browser.new_context(
                viewport={'width': html_w, 'height': html_h},
                device_scale_factor=1.0,
            )
            page_fb = context_fb.new_page()
            try:
                page_fb.route("**/*.{woff,woff2,ttf,otf,eot}", lambda r: r.abort())
                page_fb.route("**/*fonts.googleapis.com*", lambda r: r.abort())
                page_fb.route("**/*fonts.gstatic.com*", lambda r: r.abort())
                page_fb.add_init_script("""
                    const style = document.createElement('style');
                    style.innerHTML = `* { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important; }`;
                    document.head.appendChild(style);
                """)
                page_fb.goto(f"file://{abs_html_path}", wait_until="domcontentloaded", timeout=15000)
                page_fb.wait_for_timeout(500)
                page_fb.screenshot(path=output_path, omit_background=True)
                success = True
            except Exception as e:
                last_error = str(e)
            finally:
                page_fb.close()
                context_fb.close()
        browser.close()

    # 后处理：内容未填满画布时先按边界裁剪，再缩放到 ref 尺寸。
    if success and os.path.exists(output_path):
        with Image.open(output_path) as img:
            arr = np.array(img)
            # 检测 RGBA 或 RGB 图像中的非透明内容边界。
            if arr.shape[2] == 4:
                mask = arr[:, :, 3] > 0
            else:
                mask = arr.mean(axis=2) > 5
            if mask.any():
                cols = np.where(mask.any(axis=0))[0]
                rows = np.where(mask.any(axis=1))[0]
                content_w = cols[-1] + 1
                content_h = rows[-1] + 1
                img_w, img_h = img.size
                # 任一维度的内容占比低于 85% 时先裁剪。
                if content_w < img_w * 0.85 or content_h < img_h * 0.85:
                    img = img.crop((0, 0, content_w, content_h))
            if img.size != (ref_w, ref_h):
                resized = img.resize((ref_w, ref_h), Image.LANCZOS)
                resized.save(output_path)
            elif img.size == (ref_w, ref_h):
                img.save(output_path)

    if not success and last_error:
        raise RuntimeError(f"code2world render failed: {last_error}")
    return success



def _require_valid_png(output_path: str, success: bool, renderer: str) -> None:
    """仅当 renderer 明确成功并写出有效 PNG 时正常返回，否则抛出异常。"""
    path = Path(output_path)
    if not success:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"{renderer} renderer returned failure")
    if not path.is_file() or path.stat().st_size == 0:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"{renderer} renderer did not create a PNG")
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.width <= 0 or image.height <= 0:
                raise ValueError("output is not a non-empty PNG")
            image.verify()
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"{renderer} renderer produced an invalid PNG: {exc}") from exc

def render_html_string_to_image(html_code: str, work_dir: str, idx: int = 0,
                                wm: str = "code2world",
                                ref_w: int | None = None,
                                ref_h: int | None = None) -> np.ndarray:
    """Render world-model HTML with the model-specific protocol.

    Args:
        html_code: HTML string emitted by the world model
        work_dir:  directory for sample_<idx>.html and sample_<idx>.png
        idx:       index suffix used in the file names
        wm:        renderer to use; only "code2world" ships with this release
        ref_w, ref_h: native pixel size of the reference image, used to keep the
                      output size aligned with the input screenshot

    Returns: the PNG as a NumPy RGB array.
    """
    os.makedirs(work_dir, exist_ok=True)
    html_path = os.path.join(work_dir, f"sample_{idx}.html")
    png_path = os.path.join(work_dir, f"sample_{idx}.png")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_code)
    Path(png_path).unlink(missing_ok=True)

    if wm != "code2world":
        raise ValueError(f"unknown wm: {wm}")
    c2w_ref_w = ref_w if ref_w else C2W_TARGET_W
    c2w_ref_h = ref_h if ref_h else C2W_TARGET_H
    render = lambda: _render_code2world_html_to_png(
        html_path, png_path, c2w_ref_w, c2w_ref_h)

    for attempt in range(RENDER_ATTEMPTS):
        try:
            _require_valid_png(png_path, render(), wm)
            break
        except Exception as error:
            Path(png_path).unlink(missing_ok=True)
            transient = classify_failure(str(error), "render")["class"] == "infrastructure"
            if not transient or attempt == RENDER_ATTEMPTS - 1:
                raise
            time.sleep(2)

    with Image.open(png_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)
