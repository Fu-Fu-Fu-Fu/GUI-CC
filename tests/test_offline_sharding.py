from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from offline.sharding import (
    merge_shards,
    partition_sample_ids,
    sample_output_present,
    shard_worker_root,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class PartitionTest(unittest.TestCase):
    def test_round_robin按样本顺序稳定分片(self) -> None:
        ids = [f"{index:03d}" for index in range(1, 8)]
        self.assertEqual(partition_sample_ids(ids, 3, 0), ["001", "004", "007"])
        self.assertEqual(partition_sample_ids(ids, 3, 1), ["002", "005"])
        self.assertEqual(partition_sample_ids(ids, 3, 2), ["003", "006"])

    def test_非法shard参数直接失败(self) -> None:
        with self.assertRaises(ValueError):
            partition_sample_ids(["001"], 0, 0)
        with self.assertRaises(ValueError):
            partition_sample_ids(["001"], 2, 2)

    def test_worker_root路径包含模型setting与序号(self) -> None:
        root = shard_worker_root(Path("/out"), "gworld", "WM-FullHist", 4, 1)
        self.assertEqual(
            root, Path("/out/.shards/gworld-fullhist/shard-00001-of-00004")
        )


class MergeTest(unittest.TestCase):
    def _build_shard(
        self, output_root: Path, index: int, shard_count: int,
        sample_ids: list[str], run: dict,
    ) -> None:
        config_dir = (
            shard_worker_root(output_root, "wm", "WM-Markov", shard_count, index)
            / "wm" / "markov"
        )
        samples = {}
        for sample_id in sample_ids:
            step_dir = config_dir / sample_id / "step_000"
            step_dir.mkdir(parents=True, exist_ok=True)
            (step_dir / "pred.png").write_bytes(b"png")
            samples[sample_id] = {
                "sample_id": sample_id, "complete": True, "steps": [{"step": 0}],
            }
        _write_json(config_dir / "run.json", run)
        _write_json(config_dir / "summary.json", {"run": run, "samples": samples})

    def _samples_fixture(self, root: Path, sample_ids: list[str]) -> Path:
        samples_file = root / "offline_samples.jsonl"
        samples_file.write_text(
            "".join(json.dumps({"sample_id": sid}) + "\n" for sid in sample_ids),
            encoding="utf-8",
        )
        return samples_file

    def test_merge按样本顺序产出单一目录(self) -> None:
        run = {"model": {"model_id": "wm", "history_setting": "WM-Markov"}, "run_sha256": "r1"}
        sample_ids = ["001", "002", "003"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples_file = self._samples_fixture(root, sample_ids)
            output_root = root / "out"
            self._build_shard(output_root, 0, 2, ["001", "003"], run)
            self._build_shard(output_root, 1, 2, ["002"], run)
            final = merge_shards(
                model="wm", setting="WM-Markov",
                output_root=output_root, shard_count=2, samples_file=samples_file,
            )
            summary = json.loads((final / "summary.json").read_text())
            self.assertEqual(list(summary["samples"]), sample_ids)
            for sample_id in sample_ids:
                self.assertTrue((final / sample_id / "step_000" / "pred.png").is_file())
            # 合并目录已存在时拒绝覆盖
            with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
                merge_shards(
                    model="wm", setting="WM-Markov",
                    output_root=output_root, shard_count=2, samples_file=samples_file,
                )

    def test_merge拒绝run配置不一致或样本缺失(self) -> None:
        run_a = {"model": {"model_id": "wm", "history_setting": "WM-Markov"}, "run_sha256": "ra"}
        run_b = {"model": {"model_id": "wm", "history_setting": "WM-Markov"}, "run_sha256": "rb"}
        sample_ids = ["001", "002"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples_file = self._samples_fixture(root, sample_ids)
            output_root = root / "out"
            self._build_shard(output_root, 0, 2, ["001"], run_a)
            self._build_shard(output_root, 1, 2, ["002"], run_b)
            with self.assertRaisesRegex(ValueError, "不一致"):
                merge_shards(
                    model="wm", setting="WM-Markov",
                    output_root=output_root, shard_count=2, samples_file=samples_file,
                )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples_file = self._samples_fixture(root, sample_ids)
            output_root = root / "out"
            self._build_shard(output_root, 0, 2, ["001"], run_a)
            self._build_shard(output_root, 1, 2, [], run_a)
            with self.assertRaisesRegex(ValueError, "缺失或未终结"):
                merge_shards(
                    model="wm", setting="WM-Markov",
                    output_root=output_root, shard_count=2, samples_file=samples_file,
                )

    def test_pred缺失的完成样本不可合并(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            record = {"sample_id": "001", "complete": True, "steps": [{"step": 0}]}
            self.assertFalse(sample_output_present(config_dir, record))


if __name__ == "__main__":
    unittest.main()
