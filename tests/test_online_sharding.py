from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from online.sharding import (
    merge_shards,
    partition_task_ids,
    shard_worker_root,
    task_output_present,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _task_def(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "instruction": f"do {task_id}",
        "step_budget": 5,
        "milestones": [],
    }


class PartitionTest(unittest.TestCase):
    def test_round_robin按任务顺序稳定分片(self) -> None:
        ids = [f"task{index}" for index in range(5)]
        self.assertEqual(partition_task_ids(ids, 2, 0), ["task0", "task2", "task4"])
        self.assertEqual(partition_task_ids(ids, 2, 1), ["task1", "task3"])

    def test_worker_root路径包含模型setting与序号(self) -> None:
        root = shard_worker_root(Path("/out"), "gworld", "WM-NoHist", 4, 2)
        self.assertEqual(
            root, Path("/out/.shards/gworld-nohist/shard-00002-of-00004")
        )


class MergeTest(unittest.TestCase):
    def _build_shard(
        self, output_root: Path, index: int, shard_count: int,
        task_ids: list[str], run: dict,
    ) -> None:
        output_dir = (
            shard_worker_root(output_root, "wm", "WM-NoHist", shard_count, index)
            / "wm" / "nohist"
        )
        results: dict = {"_RUN": run}
        for task_id in task_ids:
            task_dir = output_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            record = {"task_id": task_id, "complete": True}
            _write_json(task_dir / "trajectory.json", record)
            (task_dir / "final_frame.png").write_bytes(b"png")
            results[task_id] = record
        _write_json(output_dir / "rollout_results.json", results)

    def test_merge按任务文件顺序产出rollout_results(self) -> None:
        task_ids = ["task_a", "task_b", "task_c"]
        run = {"model": {"model_id": "wm", "history_setting": "WM-NoHist"}, "run_sha256": "r1"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_file = root / "online_samples.jsonl"
            tasks_file.write_text(
                "".join(json.dumps(_task_def(task_id)) + "\n" for task_id in task_ids),
                encoding="utf-8",
            )
            output_root = root / "out"
            self._build_shard(output_root, 0, 2, ["task_a", "task_c"], run)
            self._build_shard(output_root, 1, 2, ["task_b"], run)
            final = merge_shards(
                model="wm", setting="WM-NoHist", tasks_file=tasks_file,
                output_root=output_root, shard_count=2,
            )
            merged = json.loads((final / "rollout_results.json").read_text())
            self.assertEqual(list(merged), ["_RUN", *task_ids])
            for task_id in task_ids:
                self.assertTrue((final / task_id / "final_frame.png").is_file())

    def test_merge拒绝缺失任务(self) -> None:
        task_ids = ["task_a", "task_b"]
        run = {"model": {"model_id": "wm", "history_setting": "WM-NoHist"}, "run_sha256": "r1"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_file = root / "online_samples.jsonl"
            tasks_file.write_text(
                "".join(json.dumps(_task_def(task_id)) + "\n" for task_id in task_ids),
                encoding="utf-8",
            )
            output_root = root / "out"
            self._build_shard(output_root, 0, 2, ["task_a"], run)
            self._build_shard(output_root, 1, 2, [], run)
            with self.assertRaisesRegex(ValueError, "is missing or unfinished"):
                merge_shards(
                    model="wm", setting="WM-NoHist", tasks_file=tasks_file,
                    output_root=output_root, shard_count=2,
                )

    def test_task_output_present要求终结记录与盘上产物(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            record = {"task_id": "t1", "complete": True}
            self.assertFalse(task_output_present(output_dir, record))
            task_dir = output_dir / "t1"
            task_dir.mkdir()
            _write_json(task_dir / "trajectory.json", record)
            (task_dir / "final_frame.png").write_bytes(b"png")
            self.assertTrue(task_output_present(output_dir, record))
            failed = {"task_id": "t1", "complete": False,
                      "failure_class": "model", "error": "parse_fail"}
            self.assertTrue(task_output_present(output_dir, failed))
            infra = dict(failed, failure_class="infrastructure")
            self.assertFalse(task_output_present(output_dir, infra))


if __name__ == "__main__":
    unittest.main()
