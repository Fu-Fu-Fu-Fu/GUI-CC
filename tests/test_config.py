from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.adapters.registry import create_adapter, get_model_spec, resolved_model_config
from utils.config import OFFLINE_CONFIG, ONLINE_CONFIG, REPO_ROOT, load_project_json
from offline.data import OFFLINE_DATA_ROOT, OFFLINE_SAMPLES_FILE


class ExperimentConfigTest(unittest.TestCase):
    def test_offline和online使用相同十八种模型配置(self) -> None:
        offline = load_project_json(OFFLINE_CONFIG)
        online = load_project_json(ONLINE_CONFIG)
        offline_rows = [
            (model["id"], setting)
            for model in offline["models"]
            for setting in model["settings"]
        ]
        online_rows = [
            (model["id"], setting)
            for model in online["models"]
            for setting in model["settings"]
        ]
        self.assertEqual(offline_rows, online_rows)
        self.assertEqual(len(offline_rows), 5)

    def test_直接生成模型保留formal生成参数(self) -> None:
        config = load_project_json(OFFLINE_CONFIG)
        expected = {
            "qwen_image_edit": (42, 40),
        }
        for model_id, (seed, steps) in expected.items():
            resolved = resolved_model_config(
                get_model_spec(config, model_id), "WM-Markov"
            )
            self.assertEqual((resolved["seed"], resolved["num_steps"]), (seed, steps))

    def test_judge_and_planner_default_to_the_paper_model(self) -> None:
        # The released defaults match the models reported in the paper; both are
        # overridable with --judge-model / --planner-model.
        offline = load_project_json(OFFLINE_CONFIG)
        online = load_project_json(ONLINE_CONFIG)
        self.assertEqual(offline["judge_model"], "gpt-5.5")
        self.assertEqual(online["judge_model"], "gpt-5.5")
        self.assertEqual(online["planner_model"], "gpt-5.5")
        self.assertTrue(1 <= offline["history_window"] <= 3)
        self.assertEqual(offline["history_window"], online["history_window"])

    def test_offline默认路径解析到仓库自带数据集(self) -> None:
        self.assertEqual(OFFLINE_DATA_ROOT, REPO_ROOT / "data/offline_data")
        self.assertEqual(OFFLINE_SAMPLES_FILE, REPO_ROOT / "data/offline_samples.jsonl")
        if not OFFLINE_SAMPLES_FILE.is_file():
            self.skipTest("dataset not downloaded; see README for the hf download command")
        self.assertTrue(OFFLINE_DATA_ROOT.is_dir())
        self.assertTrue(OFFLINE_SAMPLES_FILE.is_file())

    def test_模型id不会覆盖html_renderer标识(self) -> None:
        config = load_project_json(OFFLINE_CONFIG)
        spec = get_model_spec(config, "code2world")
        with TemporaryDirectory() as output_root:
            adapter = create_adapter(spec, output_root, "WM-Markov")
        self.assertEqual(adapter.name, "code2world")
        self.assertEqual(adapter.render_name, "code2world")

    def test_shell环境变量优先于paths_env(self) -> None:
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / "paths.env"
            env_file.write_text(
                "OUTPUT_ROOT=/from-file\n"
                "OPENAI_API_KEY=from-file-key\n"
                "HF_HOME=/from-file-hf\n"
                "CALLER_EMPTY=from-file-empty\n"
                "CALLER_VALUE=from-file-value\n"
                "UNSET_VALUE=from-file-unset\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment.update({
                "GUI_CC_ENV_FILE": str(env_file),
                "OUTPUT_ROOT": "/from-caller",
                "OPENAI_API_KEY": "",
            })
            environment.pop("HF_HOME", None)
            result = subprocess.run(
                [
                    "/bin/bash",
                    "-c",
                    'CALLER_EMPTY=""; CALLER_VALUE="/from-shell"; '
                    f'source "{REPO_ROOT}/scripts/env.sh"; '
                    'printf "<%s>\\n<%s>\\n<%s>\\n<%s>\\n<%s>\\n<%s>\\n" '
                    '"$OUTPUT_ROOT" "$OPENAI_API_KEY" "$HF_HOME" '
                    '"$CALLER_EMPTY" "$CALLER_VALUE" "$UNSET_VALUE"',
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "</from-caller>", "<>", "</from-file-hf>",
                "<>", "</from-shell>", "<from-file-unset>",
            ],
        )


if __name__ == "__main__":
    unittest.main()
