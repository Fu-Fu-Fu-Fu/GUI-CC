from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from online.rollout import rollout_one_task
from online.trajectory import (
    build_trajectory,
    load_task_definitions,
    validate_rollouts,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeAgent:
    def __init__(self) -> None:
        self.step_index = 0

    def reset(self, instruction: str) -> None:
        self.step_index = 0

    def step(self, frame: Image.Image) -> tuple[str, dict]:
        if self.step_index == 0:
            self.step_index += 1
            response = "Action: tap on center\n"
            plan = {"type": "tap", "x": 16, "y": 32, "target": "center"}
            action = {"type": "tap", "source_coord": [16, 32], "target": "center",
                      "low_level_instruction": "tap on center"}
        else:
            response = "Action: terminate (success)\n"
            plan = {"type": "terminate", "status": "success"}
            action = {"type": "terminate", "value": "success", "target": "",
                      "low_level_instruction": "terminate (success)"}
        self.last_step_evidence = {
            "planner_attempts": [{
                "attempt": 1,
                "raw": json.dumps(plan),
                "requested_model": "gpt-5.5-0424-global",
                "response_model": "gpt-5.5-0424-global",
                "validation_error": None,
            }],
            "plan": plan,
            "semantic_action": action,
            "formatted_response": response,
        }
        return response, action


class _FakeWorldModel:
    history_setting = "WM-NoHist"

    def __init__(self, prediction_path: Path) -> None:
        self.prediction_path = prediction_path
        self.cache_values: list[bool] = []

    def predict(self, **kwargs):
        self.cache_values.append(kwargs["use_cache"])
        image = Image.new("RGB", (32, 64), "white")
        for x in range(16):
            for y in range(64):
                image.putpixel((x, y), (32, 32, 32))
        image.save(self.prediction_path)
        return SimpleNamespace(
            error=None,
            pred_png_path=str(self.prediction_path),
            pred_html_path="",
            cached=False,
        )


class OnlinePipelineTest(unittest.TestCase):
    def test_formal任务定义可加载(self) -> None:
        if not (ROOT / "data/online_samples.jsonl").is_file():
            self.skipTest("dataset not downloaded; see README for the hf download command")
        tasks = load_task_definitions()
        self.assertEqual(len(tasks), 200)

    def test_rollout遵守禁用缓存并写入可移植路径(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs"
            seeds = root / "seeds"
            task_id = "test_task"
            (seeds / task_id).mkdir(parents=True)
            Image.new("RGB", (32, 64), "black").save(seeds / task_id / "initial.png")
            wm = _FakeWorldModel(output / "wm_prediction.png")

            summary = rollout_one_task(
                task_id,
                {
                    "task_id": task_id,
                    "instruction": "Tap once, then finish.",
                    "step_budget": 2,
                    "milestones": [],
                },
                _FakeAgent(),
                wm,
                output,
                seeds_root=seeds,
                use_cache=False,
                run_sha256="test-run",
                verbose=False,
            )

            self.assertEqual(wm.cache_values, [False])
            self.assertEqual(summary["trajectory"][0]["frame"], "test_task/step_000/frame.png")
            self.assertFalse(Path(summary["trajectory"][0]["frame"]).is_absolute())
            trajectory = build_trajectory(output, task_id, summary)
            self.assertEqual(len(trajectory["frames"]), 2)
            self.assertEqual(len(trajectory["actions"]), 1)


    def test_拒绝历史绝对frame路径(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rollout_dir = Path(tmp)
            task_id = "migrated_task"
            step_dir = rollout_dir / task_id / "step_000"
            step_dir.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(step_dir / "frame.png")
            rollout = {
                "task_id": task_id,
                "complete": True,
                "trajectory": [{
                    "step": 0,
                    "frame": f"/old/run/results/{task_id}/step_000/frame.png",
                    "semantic_action": {"type": "terminate", "value": "success"},
                    "terminated": True,
                }],
            }
            with self.assertRaisesRegex(ValueError, "must be relative to the rollout directory"):
                build_trajectory(rollout_dir, task_id, rollout)

    def test_缺失rollout会导致硬失败(self) -> None:
        task_id = "required_task"
        task_defs = {
            task_id: {
                "task_id": task_id,
                "instruction": "Required.",
                "step_budget": 1,
                "milestones": [],
            }
        }
        with self.assertRaisesRegex(ValueError, "missing 1 required rollout"):
            validate_rollouts({}, task_defs, [task_id], ROOT / "outputs/online")


if __name__ == "__main__":
    unittest.main()
