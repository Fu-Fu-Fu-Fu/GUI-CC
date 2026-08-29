from __future__ import annotations

import json
import io
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from utils.adapters.annotate_action import (
    annotate,
    coordinates_from_action,
    scroll_coordinates_from_action,
)
from utils.adapters.closedmodel_adapter import ClosedModelAdapter
from utils.adapters.code2world_adapter import Code2WorldAdapter, build_semantic_desc_official
from utils.adapters.base import BaseWMAdapter, parse_html
from utils.adapters.html_to_png import render_html_string_to_image
from utils.adapters.qwen_image_edit_adapter import _align_size, _build_action_description
from utils.prompts.paper_prompts import (
    HTML_WM_HISTORY_SYSTEM_SUFFIX,
    HTML_WM_SYSTEM_PROMPT,
)


class ActionAnnotationTest(unittest.TestCase):
    def test_tap优先使用源坐标而非bbox(self) -> None:
        action = {
            "type": "tap",
            "source_coord": [50, 50],
            "grounding": {"bbox": [220, 220, 240, 240]},
        }
        self.assertEqual(coordinates_from_action(action), (50, 50))
        rendered = annotate(np.zeros((300, 300, 3), dtype=np.uint8), action)
        self.assertGreater(rendered[50, 50, 0], 0)
        self.assertTrue(np.array_equal(rendered[230, 230], [0, 0, 0]))

    def test_long_press缺少源坐标时使用bbox(self) -> None:
        action = {"type": "long_press", "grounding": {"bbox": [120, 100, 160, 140]}}
        self.assertEqual(coordinates_from_action(action), (140, 120))
        rendered = annotate(np.zeros((260, 260, 3), dtype=np.uint8), action)
        self.assertGreater(rendered[120, 140, 0], 0)

    def test_源坐标格式错误时回退到bbox(self) -> None:
        action = {
            "type": "tap",
            "source_coord": ["invalid", None],
            "grounding": {"bbox": [20, 40, 60, 80]},
        }
        self.assertEqual(coordinates_from_action(action), (40, 60))

    def test_scroll使用记录的归一化端点(self) -> None:
        action = {
            "type": "scroll",
            "direction": "down",
            "start_norm": [250, 200],
            "end_norm": [750, 800],
        }
        self.assertEqual(
            scroll_coordinates_from_action(action, width=200, height=100),
            ((50, 20), (150, 80)),
        )

    def test_scroll缺少端点时仅按方向回退(self) -> None:
        start, end = scroll_coordinates_from_action(
            {"type": "scroll", "direction": "up"}, width=200, height=300)
        self.assertEqual(start, (100, 150))
        self.assertLess(end[1], start[1])



class ClosedModelPromptTest(unittest.TestCase):
    @staticmethod
    def _adapter(setting: str) -> ClosedModelAdapter:
        adapter = ClosedModelAdapter.__new__(ClosedModelAdapter)
        adapter.history_setting = setting
        adapter.hist_window = 3
        return adapter

    def test_nohist使用figure9_prompt和原始尺寸(self) -> None:
        adapter = self._adapter("WM-NoHist")
        messages = adapter.build_messages(
            np.zeros((120, 80, 3), dtype=np.uint8),
            {"type": "tap", "target": "Save", "source_coord": [10, 20]},
        )
        self.assertEqual(
            messages[0]["content"][0]["text"],
            HTML_WM_SYSTEM_PROMPT.format(W=80, H=120),
        )
        self.assertIn("Click Save.", messages[1]["content"][1]["text"])

    def test_full_history使用figure10且仅保留最近三张图(self) -> None:
        adapter = self._adapter("WM-FullHist")
        history = [
            {
                "before_arr": np.full((60, 40, 3), i, dtype=np.uint8),
                "semantic_action": {"type": "tap", "target": f"target-{i}"},
            }
            for i in range(5)
        ]
        messages = adapter.build_messages(
            np.zeros((120, 80, 3), dtype=np.uint8),
            {"type": "tap", "target": "current", "source_coord": [10, 20]},
            history=history,
        )
        self.assertEqual(
            messages[0]["content"][0]["text"],
            HTML_WM_SYSTEM_PROMPT.format(W=80, H=120) + HTML_WM_HISTORY_SYSTEM_SUFFIX,
        )
        content = messages[1]["content"]
        self.assertEqual(sum(item["type"] == "image_url" for item in content), 4)
        text = "".join(item["text"] for item in content if item["type"] == "text")
        self.assertNotIn("target-0", text)
        self.assertNotIn("target-1", text)
        for index in (2, 3, 4):
            self.assertIn(f"target-{index}", text)
        self.assertIn("current", text)


class OpenModelHistoryPromptTest(unittest.TestCase):
    @staticmethod
    def _history() -> list[dict]:
        return [
            {
                "before_arr": np.full((60, 40, 3), index, dtype=np.uint8),
                "semantic_action": {
                    "type": "tap", "target": f"target-{index}",
                    "source_coord": [10, 20],
                },
            }
            for index in range(3)
        ]

    def test_code2world使用figure5历史文本(self) -> None:
        adapter = Code2WorldAdapter.__new__(Code2WorldAdapter)
        adapter.history_setting = "WM-FullHist"
        adapter.hist_window = 3
        messages = adapter.build_messages(
            np.zeros((120, 80, 3), dtype=np.uint8),
            {"type": "tap", "target": "current", "source_coord": [10, 20]},
            self._history(),
        )
        content = messages[1]["content"]
        self.assertEqual(sum(item["type"] == "image_url" for item in content), 4)
        text = "".join(item["text"] for item in content if item["type"] == "text")
        self.assertIn("ACTION HISTORY, oldest first:", text)
        self.assertIn("^ CURRENT state. The red action cue is drawn on top.", text)



class ActionPromptTest(unittest.TestCase):

    def test_qwen_max_pixels属于运行时配置(self) -> None:
        width, height = _align_size(1080, 2400, max_pixels=200_000)
        self.assertLessEqual(width * height, 200_000)



class HtmlRendererFailureTest(unittest.TestCase):
    def test_renderer返回失败时删除旧png并抛错(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stale = Path(tmp) / "sample_0.png"
            Image.new("RGB", (4, 4), "gray").save(stale)
            with patch(
                "utils.adapters.html_to_png._render_code2world_html_to_png",
                return_value=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "returned failure"):
                    render_html_string_to_image("<html></html>", tmp)
            self.assertFalse(stale.exists())

    def test_成功但无png时抛错而不创建灰色回退图(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "utils.adapters.html_to_png._render_code2world_html_to_png",
                return_value=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "did not create a PNG"):
                    render_html_string_to_image("<html></html>", tmp)
            self.assertFalse((Path(tmp) / "sample_0.png").exists())

    def test_返回有效renderer输出(self) -> None:
        def render(_html, output, *_dims):
            Image.new("RGB", (7, 9), "white").save(output)
            return True

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "utils.adapters.html_to_png._render_code2world_html_to_png",
                side_effect=render,
            ):
                result = render_html_string_to_image("<html></html>", tmp)
            self.assertEqual(result.shape, (9, 7, 3))


class HtmlCacheIntegrityTest(unittest.TestCase):
    def test_html缓存按request标识命中(self) -> None:
        class Adapter(BaseWMAdapter):
            name = "code2world"

            def build_messages(self, before_arr, semantic_action, history=None):
                return [{"role": "user", "content": "render"}]

        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                model="code2world",
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content="<!DOCTYPE html><html><body>ok</body></html>"
                ))],
            )

        def render(html, work_dir, **_kwargs):
            root = Path(work_dir)
            (root / "sample_0.html").write_text(html, encoding="utf-8")
            Image.new("RGB", (16, 32), "white").save(root / "sample_0.png")

        with tempfile.TemporaryDirectory() as tmp:
            adapter = Adapter("http://localhost/v1", "code2world", tmp)
            adapter._client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            before = np.zeros((32, 16, 3), dtype=np.uint8)
            with patch("utils.adapters.html_to_png.render_html_string_to_image", side_effect=render):
                first = adapter.predict("sample", 0, before, {"type": "wait"})
                cached = adapter.predict("sample", 0, before, {"type": "wait"})
                Path(first.pred_png_path).unlink()
                regenerated = adapter.predict("sample", 0, before, {"type": "wait"})
        self.assertFalse(first.cached)
        self.assertTrue(cached.cached)
        self.assertFalse(regenerated.cached)
        self.assertEqual(len(calls), 2)



if __name__ == "__main__":
    unittest.main()
