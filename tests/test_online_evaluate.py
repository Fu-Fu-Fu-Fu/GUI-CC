from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from online import judges
from online.scoring import (
    JudgeCache,
    PreparedTask,
    _task_signature,
    aggregate_results,
    evaluate_task,
    overall,
    prepare_tasks,
    reusable_result,
)


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected judge call")
        content = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            model="gpt-5.5-0424-global",
        )


def _complete_record(task_id: str, value: float) -> dict:
    metrics = {metric: value for metric in (*judges.PAPER_METRICS, "Overall")}
    return {
        "task_id": task_id,
        "signature": {"sha256": f"signature-{task_id}"},
        "status": "complete",
        "metrics": metrics,
        "errors": [],
    }


class OnlineEvaluateTest(unittest.TestCase):
    def test_完整真实结果可以断点复用(self) -> None:
        record = _complete_record("task", 0.5)
        self.assertTrue(reusable_result(record, "signature-task"))
        aggregate = aggregate_results(["task"], {"task": record})
        self.assertEqual(aggregate["n_complete_tasks"], 1)

    def test_完整任务使用论文六项metric公式(self) -> None:
        use = {key: 1 for key in judges._S_USE_KEYS}
        use.update({"score": 5, "failure_modes": [], "reasoning": "usable"})
        cp = {key: 1 for key in judges._S_CP_KEYS}
        cp.update({"score": 5, "reasoning": "persistent"})
        rd = {key: 0 for key in judges._S_RD_KEYS}
        rd.update({"score": 0, "reasoning": "frozen"})
        client = _FakeClient([
            json.dumps({"score": 8.0, "reasoning": "good"}),
            json.dumps({"inferred_action": "wait", "reasoning": "unchanged"}),
            json.dumps(use),
            json.dumps(cp),
            json.dumps(rd),
            json.dumps({"passed": 1, "first_satisfied_frame": 1, "evidence": "done"}),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before.png"
            after = root / "after.png"
            Image.new("RGB", (8, 8), "black").save(before)
            Image.new("RGB", (8, 8), "white").save(after)
            prepared = PreparedTask(
                task_id="task",
                task={
                    "task_id": "task",
                    "instruction": "Wait until done.",
                    "step_budget": 1,
                    "milestones": [{"id": "m1", "type": ["terminal_success"], "assertion": "Done."}],
                },
                rollout={"task_id": "task"},
                frames=(before, after),
                actions=({"type": "wait"},),
                signature="signature",
            )
            result = evaluate_task(
                prepared,
                client,
                "gpt-5.5-0424-global",
                JudgeCache(root / "cache"),
            )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(set(result["metrics"]), set((*judges.PAPER_METRICS, "Overall")))
        self.assertAlmostEqual(result["metrics"]["Overall"], 0.8)
        self.assertEqual(result["paper_scores"]["Overall"], 80.0)
        self.assertEqual(result["errors"], [])

    def test_overall要求六项metric齐全且无错误(self) -> None:
        metrics = {metric: 0.5 for metric in judges.PAPER_METRICS}
        self.assertEqual(overall(metrics, []), 0.5)
        metrics["S_mp"] = None
        self.assertIsNone(overall(metrics, []))
        metrics["S_mp"] = 0.5
        self.assertIsNone(overall(metrics, [{"kind": "judge_error"}]))

    def test_overall不在生成论文分数前舍入(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.600001]
        metrics = dict(zip(judges.PAPER_METRICS, values))
        self.assertAlmostEqual(overall(metrics, []), sum(values) / len(values), places=12)

    def test_缺失metric会使aggregate和分母无效(self) -> None:
        record = _complete_record("task", 0.5)
        record["status"] = "error"
        record["metrics"]["S_mp"] = None
        record["metrics"]["Overall"] = None
        aggregate = aggregate_results(["task"], {"task": record})
        self.assertEqual(aggregate["status"], "error")
        self.assertEqual(aggregate["n_requested"], 1)
        self.assertEqual(aggregate["n"]["S_mp"], 0)
        self.assertIsNone(aggregate["metrics"]["Overall"])
        self.assertGreater(aggregate["errors"]["count"], 0)

    def test_subset_aggregate忽略未请求的恢复记录(self) -> None:
        results = {
            "selected": _complete_record("selected", 0.25),
            "unrelated": _complete_record("unrelated", 1.0),
        }
        aggregate = aggregate_results(["selected"], results, full=False)
        self.assertEqual(aggregate["status"], "complete")
        self.assertEqual(aggregate["scope"], "partial")
        self.assertEqual(aggregate["n_requested"], 1)
        self.assertEqual(aggregate["n"]["S_ad"], 1)
        self.assertEqual(aggregate["metrics"]["S_ad"], 0.25)
        self.assertEqual(aggregate["paper_scores"]["S_ad"], 25.0)
        self.assertIsNone(aggregate["metrics"]["Overall"])

    def test_二十五张实际frame全部发送给trajectory_judge(self) -> None:
        parsed = {key: 1 for key in judges._S_CP_KEYS}
        parsed.update({"score": 5, "reasoning": "all frames are coherent"})
        client = _FakeClient([json.dumps(parsed)])
        with tempfile.TemporaryDirectory() as tmp:
            frames = []
            for index in range(25):
                path = Path(tmp) / f"frame_{index}.png"
                Image.new("RGB", (8, 8), (index, 0, 0)).save(path)
                frames.append(path)
            result = judges.judge_s_cp(
                client,
                "gpt-5.5-0424-global",
                "Do the task.",
                [{"type": "wait"} for _ in range(24)],
                frames,
            )
        self.assertEqual(result["score"], 1.0)
        content = client.calls[0]["messages"][1]["content"]
        self.assertEqual(sum(part.get("type") == "image_url" for part in content), 25)
        labels = [part["text"] for part in content if part.get("type") == "text"]
        self.assertIn("Frame 24.", labels)

    def test_milestone_frame必须引用实际frame(self) -> None:
        client = _FakeClient([json.dumps({
            "passed": 1,
            "first_satisfied_frame": 25,
            "evidence": "outside the trajectory",
        })])
        with tempfile.TemporaryDirectory() as tmp:
            frames = []
            for index in range(25):
                path = Path(tmp) / f"frame_{index}.png"
                Image.new("RGB", (8, 8)).save(path)
                frames.append(path)
            result = judges.judge_s_mp(
                client,
                "gpt-5.5-0424-global",
                {
                    "instruction": "Do the task.",
                    "milestones": [{"id": "m1", "type": ["terminal_success"], "assertion": "Done."}],
                },
                frames,
            )
        self.assertIsNone(result["score"])
        self.assertEqual(result["error"]["kind"], "frame_parse_error")

    def test_judge错误不会转换成milestone失败(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "frame.png"
            Image.new("RGB", (8, 8)).save(frame)
            with patch("online.judges.vlm_call", return_value={
                "raw": "",
                "parsed": {},
                "error": "service unavailable",
                "api_model": "gpt-5.5-0424-global",
            }):
                result = judges.judge_s_mp(
                    object(),
                    "gpt-5.5-0424-global",
                    {
                        "instruction": "Do the task.",
                        "milestones": [{"id": "m1", "type": ["terminal_success"], "assertion": "Done."}],
                    },
                    [frame],
                )
        self.assertIsNone(result["score"])
        self.assertEqual(result["error"]["kind"], "api_error")
        self.assertNotEqual(result.get("passed_prefix"), 0)

    def test_judge响应模型必须与请求的gpt55一致(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "frame.png"
            Image.new("RGB", (8, 8)).save(frame)
            with patch("online.judges.vlm_call", return_value={
                "raw": '{"score":10,"reasoning":"ok"}',
                "parsed": {"score": 10, "reasoning": "ok"},
                "requested_model": "gpt-5.5-0424-global",
                "api_model": "another-model",
            }):
                result = judges.judge_s_ad(
                    object(), "gpt-5.5-0424-global", frame, frame, "Do it.", {"type": "wait"}
                )
        self.assertEqual(result["error"]["kind"], "model_identity_error")

    def test_二值字段不接受字符串值(self) -> None:
        parsed = {key: 1 for key in judges._S_USE_KEYS}
        parsed[judges._S_USE_KEYS[0]] = "1"
        parsed["score"] = 5
        client = _FakeClient([json.dumps(parsed)])
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "frame.png"
            Image.new("RGB", (8, 8)).save(frame)
            result = judges.judge_s_use(client, "gpt-5.5-0424-global", frame)
        self.assertIsNone(result["score"])
        self.assertEqual(result["error"]["kind"], "binary_parse_error")

    def test_judge要求rubric声明的解释证据(self) -> None:
        use = {key: 1 for key in judges._S_USE_KEYS}
        use.update({"score": 5, "failure_modes": []})
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "frame.png"
            Image.new("RGB", (8, 8)).save(frame)
            use_result = judges.judge_s_use(
                _FakeClient([json.dumps(use)]),
                "gpt-5.5-0424-global",
                frame,
            )
            milestone_result = judges.judge_s_mp(
                _FakeClient([json.dumps({"passed": 1, "first_satisfied_frame": 0})]),
                "gpt-5.5-0424-global",
                {
                    "instruction": "Do the task.",
                    "milestones": [{"id": "m1", "type": ["terminal_success"], "assertion": "Done."}],
                },
                [frame],
            )
        self.assertEqual(use_result["error"]["kind"], "evidence_parse_error")
        self.assertEqual(milestone_result["error"]["kind"], "evidence_parse_error")

    def test_signature覆盖任务rollout_prompt_endpoint和编码(self) -> None:
        task = {"task_id": "task", "instruction": "Do it", "step_budget": 1, "milestones": []}
        rollout = {"task_id": "task", "step_budget": 1, "trajectory": []}

        def signature(**overrides):
            arguments = {
                "rollout_run_sha256": "run-a",
                "judge_model": "gpt-5.5-0424-global",
                "base_url": "https://judge.example/v1",
            }
            arguments.update(overrides)
            return _task_signature(task, rollout, **arguments)

        first = signature()
        run_changed = signature(rollout_run_sha256="run-b")
        endpoint = signature(base_url="https://other.example/v1")
        with patch("online.judges.prompt_payload", return_value={"changed": "prompt"}):
            prompt = signature()
        with patch.dict("online.judges.IMAGE_ENCODING",
                        {"step": {"longest_edge": 777, "jpeg_quality": 88}}):
            encoding = signature()
        changed_task = dict(task, instruction="Do something else")
        task_changed = _task_signature(
            changed_task, rollout,
            rollout_run_sha256="run-a",
            judge_model="gpt-5.5-0424-global",
            base_url="https://judge.example/v1",
        )
        changed_rollout = dict(rollout, trajectory=[{"step": 0}])
        rollout_changed = _task_signature(
            task, changed_rollout,
            rollout_run_sha256="run-a",
            judge_model="gpt-5.5-0424-global",
            base_url="https://judge.example/v1",
        )
        for changed in (run_changed, endpoint, prompt, encoding, task_changed, rollout_changed):
            self.assertNotEqual(first, changed)

    def test_缓存要求签名匹配且拒绝错误结果(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = JudgeCache(Path(tmp))
            path = Path(tmp) / "task" / "S_cp.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"score": 1.0}), encoding="utf-8")
            self.assertIsNone(cache.get("task", "S_cp", "signature"))
            cache.put("task", "S_cp", "signature", {"score": 1.0, "error": {"kind": "api_error"}})
            self.assertIsNone(cache.get("task", "S_cp", "signature"))
            cache.put("task", "S_cp", "signature", {"score": 1.0, "reasoning": "ok"})
            self.assertEqual(cache.get("task", "S_cp", "signature")["score"], 1.0)
            self.assertIsNone(cache.get("task", "S_cp", "different"))

    def test_prepare校验rollout齐全且run标识一致(self) -> None:
        task_id = "task"
        task = {
            "task_id": task_id,
            "instruction": "Stop.",
            "step_budget": 1,
            "milestones": [{"id": "m1", "type": ["terminal_success"], "assertion": "Stopped."}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = root / task_id / "step_000" / "frame.png"
            frame.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "black").save(frame)
            run = {
                "schema": "gui_cc_online_rollout",
                "model": {"model_id": "test"},
                "planner": {"model": "gpt-5.5-0424-global"},
                    "run_sha256": "run-a",
            }
            rollout = {
                "task_id": task_id,
                "step_budget": 1,
                "run_sha256": "run-a",
                "complete": True,
                "error": None,
                "trajectory": [{
                    "step": 0,
                    "frame": f"{task_id}/step_000/frame.png",
                    "agent_response": "Action: Finish.\n",
                    "agent_evidence": f"{task_id}/step_000/agent_evidence.json",
                    "semantic_action": {"type": "terminate", "value": "success"},
                    "terminated": True,
                }],
            }
            prepared = prepare_tasks(
                {task_id: task},
                {"_RUN": run, task_id: rollout},
                [task_id],
                root,
                judge_model="gpt-5.5-0424-global",
                base_url="https://judge.example/v1",
            )
            self.assertIn(task_id, prepared)
            mismatched = dict(rollout, run_sha256="run-b")
            with self.assertRaisesRegex(ValueError, "运行标识与 _RUN 不一致"):
                prepare_tasks(
                    {task_id: task},
                    {"_RUN": run, task_id: mismatched},
                    [task_id],
                    root,
                    judge_model="gpt-5.5-0424-global",
                    base_url="https://judge.example/v1",
                )


if __name__ == "__main__":
    unittest.main()
