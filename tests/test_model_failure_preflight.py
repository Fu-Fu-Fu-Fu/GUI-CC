from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import ExitStack as _ExitStack
from types import SimpleNamespace
from unittest.mock import patch


class ExitStack(_ExitStack):
    @classmethod
    def from_patches(cls, patches):
        stack = cls()
        for item in patches:
            stack.enter_context(item)
        return stack


from PIL import Image

from utils.adapters.base import parse_html
from offline.cli import run


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HtmlFenceParsingTest(unittest.TestCase):
    def test_complete_html_document_inside_outer_fence_is_accepted(self) -> None:
        document = "<!DOCTYPE html><html><body>ok</body></html>"
        response = f"```html\n{document}\n```"
        self.assertEqual(parse_html(response), document)

    def test_incomplete_fenced_html_remains_invalid(self) -> None:
        response = "```html\n<!DOCTYPE html><html><body>truncated\n```"
        for parser in (parse_html,):
            with self.assertRaises(ValueError):
                parser(response)


class OfflineEvaluatorFailurePolicyTest(unittest.TestCase):
    def _fixture(self, root: Path, outcome: str) -> SimpleNamespace:
        tasks_root = root / "tasks"
        pred_root = root / "predictions"
        result_root = root / "evaluation"
        sample_dir = tasks_root / "sample"
        initial = sample_dir / "initial.png"
        n_steps = 2 if outcome == "model" else 1
        screenshots = [(initial, "white")]
        for step in range(n_steps):
            screenshots.append((sample_dir / f"step_{step:03d}_after.png", "black"))
        for path, color in screenshots:
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color).save(path)
        samples_file = root / "offline_samples.jsonl"
        samples_file.write_text(json.dumps({
            "sample_id": "sample",
            "task_instruction": "exercise the evaluator failure policy",
            "n_steps": n_steps,
        }) + "\n", encoding="utf-8")
        references = [
            {
                "step_id": step,
                "semantic_action": {"type": "tap", "target": "button"},
            }
            for step in range(n_steps)
        ]
        (sample_dir / "reference_trajectory.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in references),
            encoding="utf-8",
        )

        config_dir = pred_root / "mobileworld" / "nohist"
        run_sha256 = "run-sha256"
        _write_json(config_dir / "run.json", {
            "schema": "gui_cc_offline_rollout",
            "run_sha256": run_sha256,
            "model": {
                "model_id": "mobileworld",
                "history_setting": "WM-NoHist",
            },
        })
        step_dir = config_dir / "sample" / "step_000"
        if outcome == "success":
            prediction = step_dir / "pred.png"
            prediction.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), "blue").save(prediction)
            prediction_sha256 = _sha256(prediction)
            _write_json(step_dir / "meta.json", {
                "request_sha256": "request-sha256",
                "prediction_sha256": prediction_sha256,
            })
            record = {
                "sample_id": "sample",
                "run_sha256": run_sha256,
                "complete": True,
                "n_steps": 1,
                "steps": [{
                    "step": 0,
                    "request_sha256": "request-sha256",
                    "prediction_sha256": prediction_sha256,
                }],
            }
        elif outcome == "model":
            prediction = step_dir / "pred.png"
            prediction.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), "blue").save(prediction)
            prediction_sha256 = _sha256(prediction)
            _write_json(step_dir / "meta.json", {
                "request_sha256": "request-sha256",
                "prediction_sha256": prediction_sha256,
            })
            failure_dir = config_dir / "sample" / "step_001"
            _write_json(failure_dir / "meta.json", {
                "stage": "parse",
                "error": "response does not contain a complete HTML document",
                "failure_class": "model",
                "request_sha256": "failed-request-sha256",
            })
            record = {
                "sample_id": "sample",
                "run_sha256": run_sha256,
                "complete": False,
                "n_steps": 2,
                "steps": [
                    {
                        "step": 0,
                        "request_sha256": "request-sha256",
                        "prediction_sha256": prediction_sha256,
                    },
                    {
                        "step": 1,
                        "error": "parse_fail: response does not contain a complete HTML document",
                        "failure_class": "model",
                    },
                ],
                "error": "parse_fail: response does not contain a complete HTML document",
                "failure_class": "model",
            }
        elif outcome == "infra":
            record = {
                "sample_id": "sample",
                "run_sha256": run_sha256,
                "complete": False,
                "n_steps": 1,
                "steps": [{
                    "step": 0,
                    "error": "connection refused",
                    "failure_class": "infrastructure",
                }],
                "error": "connection refused",
                "failure_class": "infrastructure",
            }
        else:  # pragma: no cover - test helper contract
            raise AssertionError(outcome)
        _write_json(config_dir / "summary.json", {"samples": {"sample": record}})
        return SimpleNamespace(
            model="mobileworld",
            setting="WM-NoHist",
            sample_ids="sample",
            subset=None,
            dry_run=True,
            api_key="",
            base_url="",
            parallel=1,
            judge_model="mock-judge",
            max_tokens=16384,
            force=False,
        ), tasks_root, pred_root, result_root, samples_file

    def _run_fixture(self, outcome: str) -> tuple[int, dict]:
        with tempfile.TemporaryDirectory() as directory:
            args, tasks_root, pred_root, result_root, samples_file = self._fixture(
                Path(directory), outcome
            )
            with self._patched_roots(tasks_root, pred_root, result_root, samples_file):
                exit_code = run(args)
            reports = list(result_root.rglob("_preflight_partial.json"))
            self.assertEqual(len(reports), 1)
            return exit_code, json.loads(reports[0].read_text(encoding="utf-8"))

    @staticmethod
    def _patched_roots(tasks_root, pred_root, result_root, samples_file):
        """把 offline.cli 的仓库内路径常量指向测试用临时目录。"""
        return ExitStack.from_patches([
            patch("offline.cli.OFFLINE_DATA_ROOT", tasks_root),
            patch("offline.cli.OFFLINE_PREDICTIONS_ROOT", pred_root),
            patch("offline.cli.OFFLINE_EVALUATION_ROOT", result_root),
            # 全集含两个样本，只评一个 → is_partial=True（写 _*_partial.json）
            patch("offline.cli.load_sample_ids", return_value=["sample", "other"]),
            patch("offline.cli.load_samples",
                  return_value={"sample": json.loads(samples_file.read_text())}),
        ])

    def test_success_dry_run_passes(self) -> None:
        exit_code, report = self._run_fixture("success")
        self.assertEqual(exit_code, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["n_model_failure_transitions_skipped"], 0)

    def test_valid_model_failure_dry_run_passes_without_prediction(self) -> None:
        exit_code, report = self._run_fixture("model")
        self.assertEqual(exit_code, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["n_model_failure_transitions_skipped"], 1)

    def test_infrastructure_failure_still_blocks_dry_run(self) -> None:
        exit_code, report = self._run_fixture("infra")
        self.assertEqual(exit_code, 2)
        self.assertFalse(report["passed"])
        self.assertTrue(any(
            "INFRA_BLOCKED" in item.get("error", "") for item in report["errors"]
        ))

    def test_model_failure_is_aggregated_without_calling_judge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args, tasks_root, pred_root, result_root, samples_file = self._fixture(
                Path(directory), "model"
            )
            args.dry_run = False
            with self._patched_roots(tasks_root, pred_root, result_root, samples_file), \
                    patch("offline.cli.OpenAI", side_effect=AssertionError("judge called")):
                self.assertEqual(run(args), 0)
            result_files = list(result_root.rglob("_results_partial.json"))
            aggregate_files = list(result_root.rglob("_aggregate_partial.json"))
            self.assertEqual(len(result_files), 1)
            self.assertEqual(len(aggregate_files), 1)
            results = json.loads(result_files[0].read_text(encoding="utf-8"))
            aggregate = json.loads(aggregate_files[0].read_text(encoding="utf-8"))
            self.assertEqual(results["sample"]["failure_class"], "model")
            self.assertTrue(aggregate["complete"])
            self.assertEqual(aggregate["failure_counts"]["model"], 1)


if __name__ == "__main__":
    unittest.main()
