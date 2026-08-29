from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.collect_results import _matrix, _offline_rows, _online_rows


ROOT = Path(__file__).resolve().parents[1]


class ProjectLayoutTest(unittest.TestCase):
    def test_结果汇总脚本可从文档命令直接启动(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/collect_results.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("收集完整的 GUI-CC 结果矩阵", result.stdout)

    def test_offline数据是完整的500个样本(self) -> None:
        if not (ROOT / "data/offline_samples.jsonl").is_file():
            self.skipTest("dataset not downloaded; see README for the hf download command")
        rows = [
            json.loads(line)
            for line in (ROOT / "data/offline_samples.jsonl").read_text().splitlines()
            if line.strip()
        ]
        sample_ids = [row["sample_id"] for row in rows]
        self.assertEqual(len(sample_ids), 500)
        self.assertEqual(len(set(sample_ids)), 500)
        self.assertEqual(sum(row["n_steps"] for row in rows), 4905)
        dirs = {p.name for p in (ROOT / "data/offline_data").iterdir() if p.is_dir()}
        self.assertEqual(dirs, set(sample_ids))

    def test_online任务与seed一致(self) -> None:
        if not (ROOT / "data/online_samples.jsonl").is_file():
            self.skipTest("dataset not downloaded; see README for the hf download command")
        rows = [
            json.loads(line)
            for line in (ROOT / "data/online_samples.jsonl").read_text().splitlines()
            if line.strip()
        ]
        task_ids = [row["task_id"] for row in rows]
        seeds = {
            path.name
            for path in (ROOT / "data/online_data").iterdir()
            if path.is_dir()
        }
        self.assertEqual(len(task_ids), 200)
        self.assertEqual(set(task_ids), seeds)

    def test_实验矩阵包含十八种配置(self) -> None:
        for name in ("offline", "online"):
            config = json.loads((ROOT / f"utils/configs/{name}.json").read_text())
            rows = sum(len(model["settings"]) for model in config["models"])
            self.assertEqual(rows, 5, name)
            self.assertEqual(len(_matrix(name)), 5)

    def test_结果汇总拒绝目录与offline模型身份不一致(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "offline/evaluation/code2world/markov/_aggregate.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "schema": "gui_cc_offline_evaluation",
                "complete": True,
                "scope": "full",
                "wm": "gworld",
            }), encoding="utf-8")
            with patch("scripts.collect_results._matrix", return_value=[("code2world", "WM-Markov")]):
                with self.assertRaisesRegex(ValueError, "模型与目录不一致"):
                    _offline_rows(root)

    def test_结果汇总拒绝目录与online模型身份不一致(self) -> None:
        if not (ROOT / "data/online_samples.jsonl").is_file():
            self.skipTest("dataset not downloaded; see README for the hf download command")
        task_ids = [
            json.loads(line)["task_id"]
            for line in (ROOT / "data/online_samples.jsonl").read_text().splitlines()
            if line.strip()
        ]
        rollout = {
            "schema": "gui_cc_online_rollout",
            "model": {"model_id": "gworld", "history_setting": "WM-Markov"},
            "planner": {"model": "gpt-5.5-0424-global"},
        }
        rollout["run_sha256"] = hashlib.sha256(
            json.dumps(rollout, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "online/code2world/markov/evaluation_results.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "run": {
                    "schema": "gui_cc_online_evaluation",
                    "mode": "full",
                    "is_partial": False,
                    "status": "complete",
                    "n_requested": 200,
                    "task_ids": task_ids,
                    "judge_model": "gpt-5.5-0424-global",
                    "rollout": rollout,
                },
                "tasks": {task_id: {} for task_id in task_ids},
                "aggregate": {"status": "complete"},
            }), encoding="utf-8")
            with patch("scripts.collect_results._matrix", return_value=[("code2world", "WM-Markov")]):
                with self.assertRaisesRegex(ValueError, "模型与目录不一致"):
                    _online_rows(root)


if __name__ == "__main__":
    unittest.main()
