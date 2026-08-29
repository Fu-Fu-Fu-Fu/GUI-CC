from __future__ import annotations

import copy
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from offline.scoring import (
    METRICS,
    SCHEMA,
    STEP_METRICS,
    aggregate_results,
    build_signature,
    cache_valid,
    evaluation_config,
    overall,
)
from offline.judges import JudgeError, OfflineJudge, public_action, strict_binary
from offline.cli import _select_samples


def complete_result(value: float) -> dict:
    metrics = {metric: value for metric in METRICS}
    metrics["Overall"] = value
    return {"complete": True, "metrics": metrics}


class OfflineMetricTest(unittest.TestCase):
    def test_论文metric名称和表格公式(self) -> None:
        self.assertEqual(METRICS, (
            "S_ele", "S_lay", "S_sig", "S_dino", "S_ad",
            "S_id", "S_use", "S_cp", "S_rd", "S_rap",
        ))
        metrics = {name: index / 10 for index, name in enumerate(METRICS, 1)}
        self.assertEqual(overall(metrics), 0.55)

    def test_overall要求所有metric齐全(self) -> None:
        metrics = {name: 1.0 for name in METRICS}
        metrics.pop("S_rap")
        self.assertIsNone(overall(metrics))
        metrics["S_rap"] = 1.1
        self.assertIsNone(overall(metrics))

    def test_aggregate分母必须匹配请求的episode(self) -> None:
        aggregate = aggregate_results(
            "wm", "WM-NoHist", ["a", "b"], {"a": complete_result(0.5)}, full=True
        )
        self.assertFalse(aggregate["complete"])
        self.assertEqual(aggregate["n_episodes_requested"], 2)
        self.assertEqual(aggregate["n_episodes_scored"], 1)
        self.assertEqual(aggregate["failure_counts"]["infra_blocked"], 1)
        self.assertIsNone(aggregate["metrics"]["Overall"])

    def test_partial评测不会生成formal_overall(self) -> None:
        aggregate = aggregate_results(
            "wm", "WM-NoHist", ["a", "b"],
            {"a": complete_result(0.0), "b": complete_result(1.0)}, full=False,
        )
        self.assertTrue(aggregate["complete"])
        self.assertEqual(aggregate["scope"], "partial")
        self.assertEqual(aggregate["metrics"]["S_ele"], 0.5)
        self.assertIsNone(aggregate["metrics"]["Overall"])

    def test_全量评测的表格分数同时包含原值和百分数(self) -> None:
        aggregate = aggregate_results(
            "wm", "WM-NoHist", ["a", "b"],
            {"a": complete_result(0.0), "b": complete_result(1.0)}, full=True,
        )
        self.assertEqual(aggregate["metrics"]["Overall"], 0.5)
        self.assertEqual(aggregate["paper_scores"]["Overall"], 50.0)
        self.assertEqual(aggregate["n_episodes_scored"], 2)

    def test_内部聚合不提前舍入(self) -> None:
        first = 0.12344
        second = 0.12347
        aggregate = aggregate_results(
            "wm", "WM-NoHist", ["a", "b"],
            {"a": complete_result(first), "b": complete_result(second)},
            full=True,
        )
        self.assertAlmostEqual(
            aggregate["metrics"]["S_ele"],
            (first + second) / 2,
            places=12,
        )


class OfflineInputTest(unittest.TestCase):
    def test_offline_judge响应模型必须与请求一致(self) -> None:
        judge = OfflineJudge(object())
        with patch("offline.judges.call_vlm", return_value={
            "raw": '{"score":10,"reasoning":"ok"}',
            "parsed": {"score": 10, "reasoning": "ok"},
            "requested_model": "gpt-5.5-0424-global",
            "api_model": "another-model",
        }):
            with self.assertRaisesRegex(JudgeError, "does not match the requested model"):
                judge._request("S_ad", [])

    def test_二值parser拒绝缺失值和非二值(self) -> None:
        with self.assertRaises(ValueError):
            strict_binary({"a": 1, "score": 1}, ["a", "b"], "score")
        with self.assertRaises(ValueError):
            strict_binary({"a": 2, "score": 2}, ["a"], "score")
        # score 与 criteria 之和不一致：以 criteria 为准，记录标记，不作废
        mismatched = {"a": 1, "b": 0, "score": 0}
        self.assertEqual(strict_binary(mismatched, ["a", "b"], "score"), {"a": 1, "b": 0})
        self.assertEqual(mismatched["_score_field_mismatch"], 0)

    def test_递归移除动作私有字段(self) -> None:
        action = {"type": "tap", "_raw": {"secret": 1},
                  "grounding": {"bbox": [1, 2], "_source": "raw"}}
        self.assertEqual(public_action(action), {
            "type": "tap", "grounding": {"bbox": [1, 2]}
        })

    def test_缓存要求当前signature和完整step_metrics(self) -> None:
        metrics = {metric: 0.5 for metric in METRICS}
        metrics["Overall"] = 0.5
        record = {
            "schema": SCHEMA, "complete": True, "signature": "current",
            "metrics": metrics,
            "per_step": [{"metrics": {metric: 0.5 for metric in STEP_METRICS}}],
            "trajectory": {"S_cp": {}, "S_rd": {}, "S_rap": {}},
        }
        self.assertTrue(cache_valid(record, "current", 1))

        unsigned = copy.deepcopy(record)
        unsigned.pop("signature")
        self.assertFalse(cache_valid(unsigned, "current", 1))

        self.assertFalse(cache_valid(record, "other", 1))

        missing = copy.deepcopy(record)
        missing["per_step"][0]["metrics"].pop("S_use")
        self.assertFalse(cache_valid(missing, "current", 1))

    def test_sample_ids选择子集即为partial(self) -> None:
        all_samples = [f"{index:03d}" for index in range(1, 501)]
        args = SimpleNamespace(sample_ids="001,002", subset=None)
        with patch("offline.cli.load_sample_ids", return_value=all_samples):
            selected, is_partial = _select_samples(args)
        self.assertEqual(selected, ["001", "002"])
        self.assertTrue(is_partial)

    def test_不传sample_ids即为全量500个样本(self) -> None:
        all_samples = [f"{index:03d}" for index in range(1, 501)]
        args = SimpleNamespace(sample_ids=None, subset=None)
        with patch("offline.cli.load_sample_ids", return_value=all_samples):
            selected, is_partial = _select_samples(args)
        self.assertEqual(selected, all_samples)
        self.assertFalse(is_partial)

    def test_sample_ids无匹配时报错(self) -> None:
        args = SimpleNamespace(sample_ids="999", subset=None)
        with patch("offline.cli.load_sample_ids", return_value=["001"]):
            with self.assertRaisesRegex(ValueError, "matched no sample"):
                _select_samples(args)

    def test_内容endpoint_prompt或配置变化会改变signature(self) -> None:
        episode = {
            "task": "do task",
            "reference": [{"semantic_action": {"type": "tap"}}],
        }

        def config(base_url="https://a", extra=None):
            return evaluation_config("wm", "WM-NoHist", base_url,
                                     extra_config=extra or {"x": 1})

        first = build_signature(episode, config())
        self.assertNotEqual(first, build_signature(episode, config(base_url="https://b")))
        self.assertNotEqual(first, build_signature(episode, config(extra={"x": 2})))

        changed_task = copy.deepcopy(episode)
        changed_task["task"] = "another task"
        self.assertNotEqual(first, build_signature(changed_task, config()))

        with patch.dict("offline.scoring.PROMPTS", {"changed": "prompt"}, clear=True):
            self.assertNotEqual(first, build_signature(episode, config()))

    def test_rap使用从一开始的整数step和公开动作(self) -> None:
        class RecordingJudge(OfflineJudge):
            def __init__(self) -> None:
                self.messages = []

            def _request(self, metric: str, messages: list[dict]) -> dict:
                self.messages.append(messages)
                return {"P1_precondition_supported": 1, "P2_action_effect_supported": 1,
                        "P3_next_action_supported_or_terminal": 1, "passed": 1,
                        "failure_reason": "none", "evidence": "visible"}

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "screen.png"
            from PIL import Image
            Image.new("RGB", (2, 2)).save(image)
            judge = RecordingJudge()
            result = judge.s_rap("task", [{
                "action": {"type": "tap", "_raw": {"hidden": True}},
                "gt_before": image, "gt_after": image,
                "pred_before": image, "pred_after": image,
            }])
        text = judge.messages[0][1]["content"][0]["text"]
        self.assertIn("Step:\n1 of 1", text)
        self.assertNotIn("_raw", text)
        self.assertEqual(result["per_step"][0]["step"], 1)
        self.assertEqual(result["per_step"][0]["action"], {"type": "tap"})

    def test_rap_api错误不会转换为零分(self) -> None:
        class FailingJudge(OfflineJudge):
            def __init__(self) -> None:
                pass

            def _request(self, metric: str, messages: list[dict]) -> dict:
                raise JudgeError(metric, "API unavailable")

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "screen.png"
            from PIL import Image
            Image.new("RGB", (2, 2)).save(image)
            transition = {"action": {"type": "tap"}, "gt_before": image,
                          "gt_after": image, "pred_before": image, "pred_after": image}
            with self.assertRaises(JudgeError):
                FailingJudge().s_rap("task", [transition])


if __name__ == "__main__":
    unittest.main()

class GatewayWithoutModelEchoTest(unittest.TestCase):
    """When a gateway omits the model echo, the judge must still score and record api_model=None."""

    def test_model回显缺席时正常出分(self) -> None:
        import json as _json
        from types import SimpleNamespace
        from offline.judges import OfflineJudge

        from tempfile import TemporaryDirectory
        from PIL import Image

        payload = _json.dumps({"element_alignment_score": 7,
                               "structural_fidelity_score": 8, "reasoning": "ok"})
        response = SimpleNamespace(
            model=None,
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content=payload))],
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: response)))
        judge = OfflineJudge(client, model="qwen3.7-plus")
        with TemporaryDirectory() as work:
            image = Path(work) / "x.png"
            Image.new("RGB", (8, 8)).save(image)
            result = judge.s_ele_lay(image, image)
        self.assertAlmostEqual(result["S_ele"], 6 / 9)
        self.assertIsNone(result["judge"]["api_model"])

