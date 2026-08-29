"""Failure attribution and fixed-denominator scoring policy tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils.failure import classify_failure
from offline.scoring import METRICS, aggregate_results
from online import judges
from online.scoring import _mean_step_metric, aggregate_results as online_aggregate


def _complete_offline_result(value: float) -> dict:
    metrics = {metric: value for metric in METRICS}
    metrics["Overall"] = value
    return {"complete": True, "metrics": metrics}


def _complete_online_record(task_id: str, value: float) -> dict:
    metrics = {metric: value for metric in (*judges.PAPER_METRICS, "Overall")}
    return {
        "task_id": task_id,
        "signature": {"sha256": f"sig-{task_id}", "inputs": {}},
        "status": "complete",
        "metrics": metrics,
        "errors": [],
    }


class ClassifyFailureTest(unittest.TestCase):
    def test_infrastructure_failures(self) -> None:
        for error in ["timeout", "connection refused", "503 Service Unavailable",
                       "out of memory", "CUDA error", "browser crashed",
                       "socket closed", "dns resolution failed"]:
            fc = classify_failure(error, "request")
            self.assertEqual(fc["class"], "infrastructure", f"Expected infra: {error}")

    def test_model_failures(self) -> None:
        fc = classify_failure("parse_fail: bad JSON", "parse")
        self.assertEqual(fc["class"], "model")
        fc = classify_failure("empty_completion", "request")
        self.assertEqual(fc["class"], "model")
        fc = classify_failure("world_model_returned_near_constant_image", "wm")
        self.assertEqual(fc["class"], "model")

    def test_generation_oom_is_infrastructure(self) -> None:
        fc = classify_failure("CUDA out of memory", "generation")
        self.assertEqual(fc["class"], "infrastructure")

    def test_render_infra_vs_model(self) -> None:
        fc = classify_failure("chromium target closed", "render")
        self.assertEqual(fc["class"], "infrastructure")
        fc = classify_failure("bad HTML structure", "render")
        self.assertEqual(fc["class"], "model")


class OfflineModelFailureTest(unittest.TestCase):
    def test_model_failure_fixed_denominator(self) -> None:
        """Model failure contributes 0.0; aggregate still complete over fixed denominator."""
        model_failed = {"complete": False, "failure_class": "model", "error": "parse_fail"}
        results = {"a": _complete_offline_result(0.5), "b": model_failed}
        agg = aggregate_results("wm", "WM-Markov", ["a", "b"], results, full=True)
        self.assertTrue(agg["complete"])
        self.assertEqual(agg["failure_counts"]["model"], 1)
        self.assertEqual(agg["failure_counts"]["infra_blocked"], 0)
        self.assertAlmostEqual(agg["metrics"]["Overall"], 0.25)
        self.assertEqual(agg["n_episodes_complete"], 1)

    def test_infra_blocked_no_aggregate(self) -> None:
        """Infrastructure failure blocks formal aggregate."""
        infra = {"complete": False, "failure_class": "infrastructure", "error": "timeout"}
        results = {"a": _complete_offline_result(0.5), "b": infra}
        agg = aggregate_results("wm", "WM-Markov", ["a", "b"], results, full=True)
        self.assertFalse(agg["complete"])
        self.assertEqual(agg["failure_counts"]["infra_blocked"], 1)
        self.assertIsNone(agg["metrics"]["Overall"])


class OnlineModelFailureTest(unittest.TestCase):
    def test_model_failure_fixed_denominator(self) -> None:
        """Model failure contributes 0.0 to 200-denominator; aggregate still complete."""
        complete = _complete_online_record("task1", 0.5)
        model_failed = {
            "task_id": "task2", "status": "error",
            "failure_class": "model", "metrics": {}, "errors": [],
        }
        results = {"task1": complete, "task2": model_failed}
        agg = online_aggregate(["task1", "task2"], results, full=True)
        self.assertEqual(agg["status"], "complete")
        self.assertEqual(agg["failure_counts"]["model"], 1)
        self.assertEqual(agg["n_complete_tasks"], 1)
        self.assertAlmostEqual(agg["metrics"]["S_ad"], 0.25)
        self.assertAlmostEqual(agg["metrics"]["Overall"], 0.25)

    def test_infra_blocked_no_aggregate(self) -> None:
        """Infrastructure failure blocks formal aggregate (online)."""
        complete = _complete_online_record("task1", 0.5)
        infra = {
            "task_id": "task2", "status": "error",
            "failure_class": "infrastructure", "metrics": {}, "errors": [],
        }
        results = {"task1": complete, "task2": infra}
        agg = online_aggregate(["task1", "task2"], results, full=True)
        self.assertEqual(agg["status"], "error")
        self.assertEqual(agg["failure_counts"]["infra_blocked"], 1)
        self.assertIsNone(agg["metrics"]["Overall"])


class OnlineStepZeroTerminateTest(unittest.TestCase):
    def test_empty_steps_produce_zero_not_error(self) -> None:
        """Step-0 terminate: no considered steps → 0.0, not None/error."""
        score, errors = _mean_step_metric([], "S_ad")
        self.assertEqual(score, 0.0)
        self.assertEqual(errors, [])

        score, errors = _mean_step_metric([{"step": 0, "skipped": True}], "S_ad")
        self.assertEqual(score, 0.0)
        self.assertEqual(errors, [])


class OfflineRolloutResumeTest(unittest.TestCase):
    def test_sample_output_present(self) -> None:
        """终结记录（完成或 model failure）且盘上产物齐全时可跳过重跑。"""
        from offline.sharding import sample_output_present
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            step_dir = config_dir / "s1" / "step_000"
            step_dir.mkdir(parents=True)
            (step_dir / "pred.png").write_bytes(b"png")

            complete = {"sample_id": "s1", "complete": True, "steps": [{"step": 0}]}
            self.assertTrue(sample_output_present(config_dir, complete))

            failed = {
                "sample_id": "s1", "complete": False, "failure_class": "model",
                "error": "parse_fail",
                "steps": [{"step": 0, "error": "parse_fail"}],
            }
            self.assertTrue(sample_output_present(config_dir, failed))

            infra = dict(failed, failure_class="infrastructure")
            self.assertFalse(sample_output_present(config_dir, infra))

            (step_dir / "pred.png").unlink()
            self.assertFalse(sample_output_present(config_dir, complete))


if __name__ == "__main__":
    unittest.main()
