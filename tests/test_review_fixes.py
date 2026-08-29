"""Regression tests for historical fixes."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.cli import _output_names  # noqa: E402
from offline.data import gt_after_path, gt_before_path  # noqa: E402
from online.trajectory import MAX_TRAJECTORY_FRAMES, load_task_definitions  # noqa: E402


class BudgetLimitTest(unittest.TestCase):
    def test_budget超出judge帧上限在加载时报错(self) -> None:
        task = {
            "task_id": "t1",
            "template_id": "x",
            "category": "System & Utility",
            "apps": ["Settings"],
            "step_budget": MAX_TRAJECTORY_FRAMES,
            "instruction": "do",
            "initial_state": "s",
            "milestones": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "online_samples.jsonl"
            path.write_text(json.dumps(task) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "judge frame limit"):
                load_task_definitions(path)


class GtPathTest(unittest.TestCase):
    def test_GT路径按约定派生(self) -> None:
        sample_dir = Path("/d")
        self.assertEqual(gt_after_path(sample_dir, 3), sample_dir / "step_003_after.png")
        self.assertEqual(gt_before_path(sample_dir, 0), sample_dir / "initial.png")
        self.assertEqual(gt_before_path(sample_dir, 1), sample_dir / "step_000_after.png")


class OutputNamesTest(unittest.TestCase):
    def test_partial结果写入独立文件名(self) -> None:
        self.assertEqual(
            _output_names(is_partial=True),
            ("_aggregate_partial.json", "_results_partial.json", "_preflight_partial.json"),
        )
        self.assertEqual(
            _output_names(is_partial=False),
            ("_aggregate.json", "_results.json", "_preflight.json"),
        )



if __name__ == "__main__":
    unittest.main()
