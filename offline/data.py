"""Paths and loaders for the offline dataset."""
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_DATA_ROOT = REPO_ROOT / "data" / "offline_data"
OFFLINE_SAMPLES_FILE = REPO_ROOT / "data" / "offline_samples.jsonl"
OFFLINE_PREDICTIONS_ROOT = REPO_ROOT / "outputs" / "offline" / "predictions"
OFFLINE_EVALUATION_ROOT = REPO_ROOT / "outputs" / "offline" / "evaluation"


def load_samples(samples_file: str | Path = OFFLINE_SAMPLES_FILE) -> dict[str, dict]:
    """按行序加载 offline_samples.jsonl，返回 sample_id -> 元数据。"""
    rows = [
        json.loads(line)
        for line in Path(samples_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {row["sample_id"]: row for row in rows}


def load_sample_ids(samples_file: str | Path = OFFLINE_SAMPLES_FILE) -> list[str]:
    """样本集合与顺序由 samples jsonl 的行序定义。"""
    return list(load_samples(samples_file))


def load_reference(sample_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (sample_dir / "reference_trajectory.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def gt_after_path(sample_dir: Path, step: int) -> Path:
    return sample_dir / f"step_{step:03d}_after.png"


def gt_before_path(sample_dir: Path, step: int) -> Path:
    """step 0 的 before 即 initial.png；其余等于上一步的 after。"""
    return sample_dir / "initial.png" if step == 0 else gt_after_path(sample_dir, step - 1)


def pred_after_path(pred_dir: Path, step: int) -> Path:
    return pred_dir / f"step_{step:03d}" / "pred.png"
